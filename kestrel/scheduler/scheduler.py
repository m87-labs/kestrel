"""Flexible batching scheduler for Moondream text inference."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence

import time
import logging

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import (
    MoondreamRuntime,
    TextToken,
    CoordToken,
    SizeToken,
    Token,
)
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.skills import (
    QuerySkill,
    SegmentRequest,
    SkillRegistry,
    SkillSpec,
    SkillState,
)

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestMetrics,
    ScheduledSequence,
    SchedulerResult,
    StreamCallback,
)
from .sampling import sample_tokens
from kestrel.utils.buffers import CpuGpuBuffer


_LOGGER = logging.getLogger(__name__)


def _prompt_with_spatial_tokens(
    prompt_tokens: Tensor,
    coord_id: int,
    size_id: int,
    spatial_refs: Sequence[Sequence[float]],
) -> list[Token]:
    """Replace coord/size placeholder ids in ``prompt_tokens`` with typed tokens.

    - 2-value refs are treated as points: ``[x, y]``.
    - 4-value refs are treated strictly as bounding boxes in
      ``[x_min, y_min, x_max, y_max]`` format.
    """
    if prompt_tokens.ndim != 1:
        tokens_1d = prompt_tokens.view(-1)
    else:
        tokens_1d = prompt_tokens
    ids = tokens_1d.cpu().tolist()

    # Precompute expected placeholder counts
    coord_placeholders = sum(1 for t in ids if t == coord_id)
    size_placeholders = sum(1 for t in ids if t == size_id)

    # Build coord and size lists from spatial refs
    coord_vals: list[float] = []
    size_vals: list[tuple[float, float]] = []
    for ref in spatial_refs:
        n = len(ref)
        if n == 2:
            x, y = float(ref[0]), float(ref[1])
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            coord_vals.extend([x, y])
        elif n == 4:
            x_min, y_min, x_max, y_max = map(float, ref)
            if not (0.0 <= x_min <= x_max <= 1.0 and 0.0 <= y_min <= y_max <= 1.0):
                raise ValueError(
                    "bbox spatial_ref must satisfy 0<=x_min<=x_max<=1 and 0<=y_min<=y_max<=1"
                )
            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            coord_vals.extend([x_c, y_c])
            size_vals.append((width, height))
        else:
            raise ValueError(
                "Each spatial_ref must contain 2 (point) or 4 (bbox) values"
            )

    expected_coords = 2 * len(spatial_refs)
    expected_sizes = sum(1 for r in spatial_refs if len(r) == 4)
    if coord_placeholders != expected_coords or size_placeholders != expected_sizes:
        raise ValueError(
            "Mismatch between spatial_refs and placeholder tokens: "
            f"prompt has {coord_placeholders} coord and {size_placeholders} size placeholders, "
            f"but refs require {expected_coords} coord and {expected_sizes} size placeholders."
        )

    # Replace placeholders in order of appearance
    coord_iter = iter(coord_vals)
    size_iter = iter(size_vals)
    out: list[Token] = []
    for tid in ids:
        if tid == coord_id:
            try:
                pos = next(coord_iter)
            except StopIteration as exc:
                raise ValueError("Insufficient coord placeholders for spatial_refs") from exc
            out.append(CoordToken(pos=float(pos)))
        elif tid == size_id:
            try:
                w, h = next(size_iter)
            except StopIteration as exc:
                raise ValueError("Insufficient size placeholders for bbox spatial_refs") from exc
            # Clamp sizes to [0, 1]
            w = min(max(float(w), 0.0), 1.0)
            h = min(max(float(h), 0.0), 1.0)
            out.append(SizeToken(width=w, height=h))
        else:
            out.append(TextToken(token_id=int(tid)))

    # Ensure all refs were consumed
    try:
        next(coord_iter)
        raise ValueError("Unconsumed coord values after placeholder replacement")
    except StopIteration:
        pass
    try:
        next(size_iter)
        raise ValueError("Unconsumed size values after placeholder replacement")
    except StopIteration:
        pass

    return out

class _SampleBuffer:
    """Pinned host buffer for sampled token ids with optional async copy."""

    def __init__(self, max_batch: int, device: torch.device) -> None:
        self._buffer = CpuGpuBuffer(
            max_batch,
            dtype=torch.long,
            device=device,
            pin_memory=True,
        )
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event(enable_timing=False, blocking=False)

    class TransferHandle:
        def __init__(self, event: torch.cuda.Event, cpu_view: Tensor, count: int) -> None:
            self._event = event
            self._cpu_view = cpu_view
            self._count = count

        def wait(self) -> Tensor:
            if self._count == 0:
                return self._cpu_view[:0]
            self._event.synchronize()
            return self._cpu_view[: self._count]

    def transfer(self, tensor: Tensor) -> "_SampleBuffer.TransferHandle":
        count = int(tensor.shape[0])
        if count == 0:
            return _SampleBuffer.TransferHandle(self._event, self._buffer.cpu, 0)

        current = torch.cuda.current_stream(device=self._device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(current)
            self._buffer.cpu[:count].copy_(tensor, non_blocking=True)
            self._event.record(self._stream)
        return _SampleBuffer.TransferHandle(self._event, self._buffer.cpu, count)


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(
        self,
        runtime: MoondreamRuntime,
        *,
        default_temperature: float = 0.2,
        default_top_p: float = 0.9,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.runtime = runtime
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[ScheduledSequence] = RunningQueue()
        self._completed: Deque[SchedulerResult] = deque()
        self._next_request_id = 0
        self._default_temperature = max(float(default_temperature), 0.0)
        self._default_top_p = float(default_top_p)
        if not (0.0 < self._default_top_p <= 1.0):
            raise ValueError("default_top_p must be in the range (0, 1]")
        self._skills = skill_registry or SkillRegistry([QuerySkill()])
        self._pinned_token_buffer = torch.empty(
            runtime.max_batch_size, dtype=torch.long, device="cpu"
        ).pin_memory()
        self._pinned_token_np = self._pinned_token_buffer.numpy()
        self._sample_buffer = _SampleBuffer(runtime.max_batch_size, runtime.device)

    # ------------------------------------------------------------------
    # Submission

    def enqueue_request(
        self,
        request: GenerationRequest,
        skill_state: SkillState,
    ) -> None:
        """Insert a fully constructed request/skill state into the waiting queue."""

        if request.skill_state is not None and request.skill_state is not skill_state:
            raise ValueError("GenerationRequest already has an associated SkillState")
        request.skill_state = skill_state
        self.waiting.push(request)

    # ------------------------------------------------------------------
    # Execution

    def has_pending_work(self) -> bool:
        """Return ``True`` if there is anything left to prefill or decode."""

        return len(self.waiting) > 0 or len(self.running) > 0

    def advance(self) -> bool:
        """Attempt to make progress by running prefill/decode once.

        Returns ``True`` if any state changed (e.g. tokens decoded, new
        sequences admitted). Callers can keep invoking ``advance`` while it
        returns ``True`` to drain ready work before sleeping.
        """

        progressed = False
        progressed |= self._try_prefill()
        progressed |= self._decode_step()

        if not progressed:
            stalled = self.waiting.peek()
            if stalled is not None and not self.runtime.can_reserve(
                stalled.target_length
            ):
                raise RuntimeError(
                    "Scheduler stalled: insufficient KV cache capacity for request "
                    f"{stalled.request_id} (needs {stalled.target_length} tokens)."
                )
        return progressed

    def pop_completed(self) -> List[SchedulerResult]:
        """Retrieve all completed results accumulated so far."""

        if not self._completed:
            return []
        items = list(self._completed)
        self._completed.clear()
        return items

    # ------------------------------------------------------------------
    # Internal helpers

    def _issue_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _try_prefill(self) -> bool:
        progress = False
        while len(self.waiting) and len(self.running) < self.runtime.max_batch_size:
            request = self.waiting.peek()
            if request is None:
                break
            if not self.runtime.can_reserve(request.target_length):
                break

            request = self.waiting.pop()
            # If this is a segmentation request with spatial refs, convert
            # placeholder coord/size ids into typed CoordToken/SizeToken
            # so the runtime embeds region features during prefill.
            prompt_inputs: Tensor | list[Token]
            ctx = request.request_context
            if isinstance(ctx, SegmentRequest) and ctx.spatial_refs:
                coord_id = self.runtime.config.tokenizer.coord_id
                size_id = self.runtime.config.tokenizer.size_id
                prompt_inputs = _prompt_with_spatial_tokens(
                    request.prompt_tokens,
                    coord_id,
                    size_id,
                    ctx.spatial_refs,
                )
            else:
                tokens = request.prompt_tokens.view(1, -1).to(
                    device=self.runtime.device, dtype=torch.long
                )
                prompt_inputs = tokens
            prefill_start = time.perf_counter()
            state, logits = self.runtime.start_sequence(
                prompt_tokens=prompt_inputs,
                image=request.image,
                image_crops=request.image_crops,
                max_new_tokens=request.max_new_tokens,
                adapter=request.adapter,
            )
            skill_state = request.skill_state
            if skill_state is None:
                skill_state = request.skill.create_state(
                    self.runtime, request, request.request_context
                )
            request.skill_state = skill_state
            seq = ScheduledSequence(
                request=request,
                state=state,
                skill_state=skill_state,
            )
            seq.skill_state.on_prefill(self.runtime)
            seq.started_at = request.submitted_at
            seq.prefill_started_at = prefill_start

            if request.max_new_tokens <= 0:
                seq.first_token_time = prefill_start
                self._finalize_sequence(seq, "length")
                progress = True
                continue

            first_logits = logits.squeeze(0)
            sampled = self._sample_batch(first_logits.unsqueeze(0), [seq.request])
            transfer = self._sample_buffer.transfer(sampled)
            sampled_host = transfer.wait()
            token_value = int(sampled_host[0].item())
            seq.stage_token(self.runtime, token_value)

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

        pending_tokens: list[Token] = []
        all_text = True
        for seq in active:
            token = seq.pending_token
            if token is None:
                raise RuntimeError(
                    "ScheduledSequence has no pending token during decode"
                )
            pending_tokens.append(token)
            if not isinstance(token, TextToken):
                all_text = False

        if all_text:
            count = len(pending_tokens)
            self._pinned_token_np[:count] = [token.token_id for token in pending_tokens]
            token_input = self._pinned_token_buffer[:count]
        else:
            token_input = pending_tokens

        logits = self.runtime.decode_batch([seq.state for seq in active], token_input)

        sampled_tokens = self._sample_batch(logits, [seq.request for seq in active])
        transfer = self._sample_buffer.transfer(sampled_tokens)

        # Overlap advances/length bookkeeping with the host copy.
        for seq in active:
            seq.state.advance()

        sampled_cpu = transfer.wait()
        sampled_np = sampled_cpu.numpy().reshape(-1)

        for seq, token_value in zip(active, sampled_np):
            seq.stage_token(self.runtime, int(token_value))
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

        allowed_tokens: list[Optional[Sequence[int]]] = []
        restrict = False
        for req in requests:
            state = req.skill_state
            if state is None:
                allowed_tokens.append(None)
                continue
            allowed = state.allowed_token_ids(self.runtime)
            allowed_tokens.append(allowed)
            if allowed:
                restrict = True

        if restrict:
            for i, allowed in enumerate(allowed_tokens):
                if not allowed:
                    continue
                idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
                row = logits[i]
                pruned = torch.full_like(row, float("-inf"))
                pruned[idx] = row[idx]
                logits[i] = pruned

        if all(req.temperature <= 0.0 for req in requests):
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
        eos_hit = isinstance(last_token, TextToken) and last_token.token_id == eos_id
        max_new_hit = seq.skill_state.token_count >= seq.request.max_new_tokens
        max_len_hit = seq.total_length >= seq.state.max_length

        if not (eos_hit or max_new_hit or max_len_hit):
            return False

        if eos_hit:
            reason = "stop"
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
        self._completed.append(self._build_result(seq))

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
        finish_reason = seq.finish_reason or "unknown"

        # Finalization can raise (e.g., malformed tokens during decode). Catch
        # and package the error so only the offending request fails.
        try:
            finalize = seq.skill_state.finalize(
                self.runtime, reason=finish_reason
            )
            tokens = finalize.tokens
            output = finalize.output
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("Failed to finalize sequence %s", seq.request.request_id)
            finish_reason = "error"
            tokens = []
            output = {"error": str(exc)}

        prompt_tokens = seq.state.prompt_length
        decode_tokens = len(tokens) if tokens else len(seq.skill_state.tokens)
        queued_at = seq.request.submitted_at
        prefill_started_at = seq.prefill_started_at or queued_at
        completed_at = seq.completed_at or time.perf_counter()
        first_token_time = seq.first_token_time or completed_at
        prefill_time_ms = max((first_token_time - prefill_started_at) * 1000.0, 0.0)
        ttft_ms = max((first_token_time - queued_at) * 1000.0, 0.0)
        decode_time_ms = max((completed_at - first_token_time) * 1000.0, 0.0)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_time_ms=prefill_time_ms,
            ttft_ms=ttft_ms,
            decode_time_ms=decode_time_ms,
        )
        return SchedulerResult(
            request_id=seq.request.request_id,
            tokens=tokens,
            finish_reason=finish_reason,
            metrics=metrics,
            output=output,
        )
