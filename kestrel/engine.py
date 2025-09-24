"""Async coordination layer for Moondream inference."""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from kestrel.config import RuntimeConfig
from kestrel.models import MoondreamTextRuntime
from kestrel.scheduler import GenerationScheduler, SchedulerResult


@dataclass(slots=True)
class EngineMetrics:
    """Timing and token accounting for a single request."""

    prompt_tokens: int
    decode_tokens: int
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.decode_tokens


@dataclass(slots=True)
class EngineResult:
    """Inference output returned to callers."""

    request_id: int
    prompt: str
    text: str
    tokens: List[int]
    finish_reason: str
    metrics: EngineMetrics


@dataclass(slots=True)
class _PendingRequest:
    request_id: int
    prompt: str
    prompt_tokens: torch.Tensor
    prompt_length: int
    max_new_tokens: int
    submitted_at: float
    future: asyncio.Future[EngineResult]


class InferenceEngine:
    """Orchestrates batched inference over a shared runtime and scheduler."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        *,
        batch_timeout_s: float = 0.02,
    ) -> None:
        self._runtime_cfg = runtime_cfg
        self._batch_timeout_s = batch_timeout_s

        self._runtime: MoondreamTextRuntime | None = None
        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._request_ids = itertools.count()
        self._shutdown = False

    @property
    def runtime(self) -> MoondreamTextRuntime:
        if self._runtime is None:
            raise RuntimeError("InferenceEngine has not been started")
        return self._runtime

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @classmethod
    async def create(
        cls,
        runtime_cfg: RuntimeConfig,
        *,
        batch_timeout_s: float = 0.02,
    ) -> "InferenceEngine":
        engine = cls(runtime_cfg, batch_timeout_s=batch_timeout_s)
        await engine._initialize()
        return engine

    async def _initialize(self) -> None:
        if self._runtime is not None:
            return
        loop = asyncio.get_running_loop()
        self._runtime = await loop.run_in_executor(None, MoondreamTextRuntime, self._runtime_cfg)
        await loop.run_in_executor(None, self._warmup)
        self._worker_task = asyncio.create_task(self._worker_loop())

    def _warmup(self) -> None:
        assert self._runtime is not None
        runtime = self._runtime
        prompt = "Warmup prompt."
        tokens = runtime.build_prompt_tokens(prompt)
        state, logits = runtime.start_sequence(prompt_tokens=tokens, max_new_tokens=1)
        try:
            next_token = torch.argmax(logits, dim=-1)
            runtime.decode(state, next_token.view(-1))
        finally:
            runtime.release_sequence(state)

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self._queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
        self._worker_task = None

    async def submit(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> EngineResult:
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        await self._ensure_started()

        req_id = next(self._request_ids)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EngineResult] = loop.create_future()

        if prompt_tokens is None:
            tokens = self.runtime.build_prompt_tokens(prompt)
        else:
            tokens = prompt_tokens

        tokens_cpu = tokens.to("cpu")
        payload = _PendingRequest(
            request_id=req_id,
            prompt=prompt,
            prompt_tokens=tokens_cpu,
            prompt_length=tokens_cpu.shape[1],
            max_new_tokens=max_new_tokens,
            submitted_at=time.perf_counter(),
            future=future,
        )
        await self._queue.put(payload)
        return await future

    async def _ensure_started(self) -> None:
        if self._runtime is None:
            await self._initialize()

    async def _worker_loop(self) -> None:
        assert self._runtime is not None
        max_batch = self._runtime.max_batch_size
        loop = asyncio.get_running_loop()

        while True:
            batch: List[_PendingRequest] = []

            request = await self._queue.get()
            if request is None:
                break
            batch.append(request)

            timeout = self._batch_timeout_s
            while len(batch) < max_batch:
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if req is None:
                    # Drain existing work before stopping.
                    self._shutdown = True
                    break
                batch.append(req)
            if not batch:
                continue

            try:
                results = await loop.run_in_executor(None, self._run_batch, batch)
            except Exception as exc:
                for req in batch:
                    future = req.future
                    if future and not future.done():
                        future.set_exception(exc)

                self._shutdown = True

                # Best-effort cleanup of any sequences that may have been admitted.
                try:
                    runtime_sequences = list(self.runtime.active_sequences.values())
                except Exception:  # pragma: no cover - defensive cleanup
                    runtime_sequences = []
                for state in runtime_sequences:
                    try:
                        self.runtime.release_sequence(state)
                    except Exception:
                        pass

                # Propagate the same failure to anything still queued.
                try:
                    while True:
                        pending = self._queue.get_nowait()
                        if pending is None:
                            continue
                        future = pending.future
                        if future and not future.done():
                            future.set_exception(exc)
                except asyncio.QueueEmpty:
                    pass

                break

            completed_at = time.perf_counter()
            for req in batch:
                future = req.future
                try:
                    result = results[req.request_id]
                except KeyError:
                    future.set_exception(
                        RuntimeError(f"Request {req.request_id} missing from scheduler results")
                    )
                    continue
                metrics = EngineMetrics(
                    prompt_tokens=req.prompt_length,
                    decode_tokens=len(result.tokens),
                    latency_s=max(0.0, completed_at - req.submitted_at),
                )
                engine_result = EngineResult(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    text=result.text,
                    tokens=result.tokens,
                    finish_reason=result.finish_reason,
                    metrics=metrics,
                )
                if not future.done():
                    future.set_result(engine_result)

            if self._shutdown:
                break

        # Cancel any pending futures in the queue.
        while not self._queue.empty():
            pending = self._queue.get_nowait()
            if pending and pending.future and not pending.future.done():
                pending.future.set_exception(RuntimeError("Engine shut down"))

    def _run_batch(self, batch: Iterable[_PendingRequest]) -> dict[int, SchedulerResult]:
        scheduler = GenerationScheduler(self.runtime)
        id_to_request: dict[int, _PendingRequest] = {}
        for req in batch:
            id_to_request[req.request_id] = req
            scheduler.submit(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                prompt_tokens=req.prompt_tokens.clone(),
                request_id=req.request_id,
            )
        results = scheduler.run()
        return {result.request_id: result for result in results}


__all__ = ["InferenceEngine", "EngineResult", "EngineMetrics"]
