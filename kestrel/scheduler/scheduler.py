"""Flexible batching scheduler for Moondream text inference."""

from __future__ import annotations

from typing import Iterable, List, Optional

import time

import torch
from torch import Tensor

from kestrel.models import MoondreamTextRuntime

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestMetrics,
    ScheduledSequence,
    SchedulerResult,
    StreamCallback,
    StreamUpdate,
)
from .sampling import sample_tokens
from kestrel.utils.image import ImageArray


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(
        self,
        runtime: MoondreamTextRuntime,
        *,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
    ) -> None:
        self.runtime = runtime
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[ScheduledSequence] = RunningQueue()
        self.completed: list[ScheduledSequence] = []
        self._next_request_id = 0
        self._default_temperature = max(float(default_temperature), 0.0)
        self._default_top_p = float(default_top_p)
        if not (0.0 < self._default_top_p <= 1.0):
            raise ValueError("default_top_p must be in the range (0, 1]")

    # ------------------------------------------------------------------
    # Submission

    def submit(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[Tensor] = None,
        request_id: Optional[int] = None,
        image: Optional[ImageArray] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream_callback: Optional[StreamCallback] = None,
    ) -> int:
        """Queue a new prompt for generation."""

        if prompt_tokens is None:
            prompt_tokens = self.runtime.build_prompt_tokens(prompt)
        if image is not None and not hasattr(self.runtime.model, "vision"):
            raise ValueError("Runtime does not support image inputs")
        image_length = (
            self.runtime.image_prefix_length if image is not None else 0
        )
        request = GenerationRequest(
            request_id=request_id if request_id is not None else self._issue_request_id(),
            prompt=prompt,
            prompt_tokens=prompt_tokens.detach().clone().to("cpu"),
            max_new_tokens=max_new_tokens,
            temperature=self._resolve_temperature(temperature),
            top_p=self._resolve_top_p(top_p),
            stream_callback=stream_callback,
            image=image,
            image_length=image_length,
        )
        self.waiting.push(request)
        return request.request_id

    def submit_many(
        self,
        prompts: Iterable[str],
        *,
        max_new_tokens: int,
        images: Optional[Iterable[Optional[ImageArray]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream_callback: Optional[StreamCallback] = None,
    ) -> List[int]:
        ids: List[int] = []
        image_iter = iter(images) if images is not None else None
        for prompt in prompts:
            image = next(image_iter) if image_iter is not None else None
            ids.append(
                self.submit(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    image=image,
                    temperature=temperature,
                    top_p=top_p,
                    stream_callback=stream_callback,
                )
            )
        if image_iter is not None:
            try:
                next(image_iter)
            except StopIteration:
                pass
            else:
                raise ValueError("Number of images does not match number of prompts")
        return ids

    # ------------------------------------------------------------------
    # Execution

    def run(self) -> List[SchedulerResult]:
        """Process queued requests until completion and return their outputs."""

        while len(self.waiting) or len(self.running):
            progressed = False
            progressed |= self._try_prefill()
            progressed |= self._decode_step()

            if not progressed:
                stalled = self.waiting.peek()
                if stalled is not None and not self.runtime.can_reserve(stalled.target_length):
                    raise RuntimeError(
                        "Scheduler stalled: insufficient KV cache capacity for request "
                        f"{stalled.request_id} (needs {stalled.target_length} tokens)."
                    )
                # No work left that we can process right now.
                break

        return [self._build_result(seq) for seq in self.completed]

    # ------------------------------------------------------------------
    # Internal helpers

    def _issue_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _try_prefill(self) -> bool:
        progress = False
        while (
            len(self.waiting)
            and len(self.running) < self.runtime.max_batch_size
        ):
            request = self.waiting.peek()
            if request is None:
                break
            if not self.runtime.can_reserve(request.target_length):
                break

            request = self.waiting.pop()
            tokens = request.prompt_tokens.view(1, -1).to(
                device=self.runtime.device, dtype=torch.long
            )
            prefill_start = time.perf_counter()
            state, logits = self.runtime.start_sequence(
                prompt_tokens=tokens,
                image=request.image,
                max_new_tokens=request.max_new_tokens,
            )
            seq = ScheduledSequence(request=request, state=state)
            seq.started_at = prefill_start

            if request.max_new_tokens <= 0:
                seq.first_token_time = prefill_start
                self._finalize_sequence(seq, "max_new_tokens")
                progress = True
                continue

            first_logits = logits.squeeze(0)
            sampled = self._sample_batch(
                first_logits.unsqueeze(0), [seq.request]
            )
            token_value = int(sampled[0])
            seq.stage_token(token_value, first_logits)
            self._emit_stream(seq)

            if self._mark_finished_if_needed(seq):
                progress = True
                continue

            self.running.push(seq)
            progress = True
        return progress

    def _decode_step(self) -> bool:
        if not len(self.running):
            return False

        active: list[ScheduledSequence] = []
        idle: list[ScheduledSequence] = []
        for seq in self.running.take_all():
            if seq.state.at_capacity():
                self._finalize_sequence(seq, "length")
                continue
            if seq.needs_decode():
                active.append(seq)
            else:
                idle.append(seq)

        if not active:
            self.running.extend(idle)
            return False

        token_ids = torch.tensor(
            [seq.pending_token for seq in active], dtype=torch.long
        )
        logits = self.runtime.decode_batch([seq.state for seq in active], token_ids)

        sampled_tokens = self._sample_batch(
            logits, [seq.request for seq in active]
        )
        sampled_cpu = sampled_tokens.cpu().tolist()

        for seq, row, token_value in zip(active, logits, sampled_cpu):
            seq.state.advance()
            seq.stage_token(int(token_value), row)
            self._emit_stream(seq)
            if not self._mark_finished_if_needed(seq):
                idle.append(seq)

        self.running.extend(idle)
        return True

    def _sample_batch(
        self, logits: Tensor, requests: List[GenerationRequest]
    ) -> Tensor:
        batch = len(requests)
        if batch == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)

        if all(req.temperature <= 0.0 for req in requests) and all(
            req.top_p >= 1.0 for req in requests
        ):
            return torch.argmax(logits, dim=-1)

        temps = torch.empty(batch, dtype=torch.float32)
        top_ps = torch.empty(batch, dtype=torch.float32)
        for i, req in enumerate(requests):
            temps[i] = req.temperature
            top_ps[i] = req.top_p
        temps = temps.to(device=logits.device)
        top_ps = top_ps.to(device=logits.device)
        return sample_tokens(logits, temps, top_ps)

    def _mark_finished_if_needed(self, seq: ScheduledSequence) -> bool:
        last_token = seq.last_token
        eos_id = self.runtime.config.tokenizer.eos_id
        eos_hit = last_token == eos_id
        max_new_hit = len(seq.generated_tokens) >= seq.request.max_new_tokens
        max_len_hit = seq.total_length >= seq.state.max_length

        if not (eos_hit or max_new_hit or max_len_hit):
            return False

        if eos_hit:
            reason = "eos"
        elif max_new_hit:
            reason = "max_new_tokens"
        else:
            reason = "length"

        self._finalize_sequence(seq, reason)
        return True

    def _finalize_sequence(self, seq: ScheduledSequence, reason: str) -> None:
        if seq.finished:
            return
        seq.finished = True
        seq.finish_reason = reason
        seq.completed_at = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = seq.completed_at
        if seq.state.batch_idx in self.runtime.active_sequences:
            self.runtime.release_sequence(seq.state)
        self.completed.append(seq)

    def _emit_stream(self, seq: ScheduledSequence) -> None:
        callback = seq.request.stream_callback
        if callback is None:
            return
        if seq.stream_offset >= len(seq.generated_tokens):
            return

        token_index = len(seq.generated_tokens) - 1
        token_id = seq.generated_tokens[token_index]
        new_tokens = seq.generated_tokens[seq.stream_offset :]
        text = self.runtime.tokenizer.decode(new_tokens) if new_tokens else ""
        update = StreamUpdate(
            request_id=seq.request.request_id,
            token=token_id,
            text=text,
            token_index=token_index,
        )
        callback(update)
        seq.stream_offset = len(seq.generated_tokens)

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return self._default_temperature
        return max(float(temperature), 0.0)

    def _resolve_top_p(self, top_p: Optional[float]) -> float:
        value = self._default_top_p if top_p is None else float(top_p)
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        return value

    def _build_result(self, seq: ScheduledSequence) -> SchedulerResult:
        text = self.runtime.tokenizer.decode(seq.generated_tokens) if seq.generated_tokens else ""
        prompt_tokens = seq.state.prompt_length
        decode_tokens = len(seq.generated_tokens)
        started_at = seq.started_at
        completed_at = seq.completed_at or time.perf_counter()
        first_token_time = seq.first_token_time or completed_at
        processing_latency = max(completed_at - started_at, 0.0)
        ttft = max(first_token_time - started_at, 0.0)
        decode_latency = max(completed_at - first_token_time, 0.0)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            processing_latency_s=processing_latency,
            ttft_s=ttft,
            decode_latency_s=decode_latency,
        )
        return SchedulerResult(
            request_id=seq.request.request_id,
            prompt=seq.request.prompt,
            tokens=list(seq.generated_tokens),
            text=text,
            finish_reason=seq.finish_reason or "unknown",
            metrics=metrics,
        )
