"""Flexible batching scheduler for Moondream text inference."""


from collections import deque
from typing import Deque, List, Optional

import time
import logging

import torch
from torch import Tensor

from kestrel.moondream.runtime import (
    MoondreamRuntime,
    TextToken,
)
from kestrel.moondream.lora import AdapterProvider
from kestrel.utils.buffers import CpuGpuBuffer
from kestrel.skills import (
    QuerySkill,
    SegmentRequest,
    SkillRegistry,
    SkillState,
)

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestMetrics,
    ScheduledSequence,
    SchedulerResult,
)
from .sampling import sample_tokens
from .transfer import RenderBuffer
from .tokens import prompt_with_spatial_tokens, render_tokens_from_packed
from .spatial import compute_spatial_values


_LOGGER = logging.getLogger(__name__)


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(
        self,
        runtime: MoondreamRuntime,
        *,
        default_temperature: float = 0.2,
        default_top_p: float = 0.9,
        skill_registry: Optional[SkillRegistry] = None,
        adapter_provider: Optional[AdapterProvider] = None,
    ) -> None:
        self.runtime = runtime
        self._adapter_provider = adapter_provider
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[ScheduledSequence] = RunningQueue()
        self._completed: Deque[SchedulerResult] = deque()
        self._next_request_id = 0
        self._default_temperature = max(float(default_temperature), 0.0)
        self._default_top_p = float(default_top_p)
        if not (0.0 < self._default_top_p <= 1.0):
            raise ValueError("default_top_p must be in the range (0, 1]")
        self._skills = skill_registry or SkillRegistry([QuerySkill()])
        self._coord_id = runtime.config.tokenizer.coord_id
        self._size_id = runtime.config.tokenizer.size_id
        coord_dtype = runtime.region.coord_features.dtype
        size_dtype = runtime.region.size_features.dtype
        self._pending_token_ids = torch.zeros(
            (runtime.max_batch_size,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._pending_coord_values = torch.zeros(
            (runtime.max_batch_size, 1),
            dtype=coord_dtype,
            device=runtime.device,
        )
        self._pending_size_values = torch.zeros(
            (runtime.max_batch_size, 2),
            dtype=size_dtype,
            device=runtime.device,
        )
        self._render_buffer = RenderBuffer(
            runtime.max_batch_size,
            runtime.device,
            coord_dtype=coord_dtype,
            size_dtype=size_dtype,
        )
        # Preallocated staging buffers for gathering the packed decode inputs
        # from the pending per-sequence slots (avoids per-step allocations).
        self._decode_token_ids = torch.empty(
            (runtime.max_batch_size,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._decode_coord_values = torch.empty(
            (runtime.max_batch_size, 1),
            dtype=coord_dtype,
            device=runtime.device,
        )
        self._decode_size_values = torch.empty(
            (runtime.max_batch_size, 2),
            dtype=size_dtype,
            device=runtime.device,
        )
        self._sampled_token_ids = torch.empty(
            (runtime.max_batch_size,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._decode_batch_idx = CpuGpuBuffer(
            runtime.max_batch_size,
            dtype=torch.long,
            device=runtime.device,
            pin_memory=True,
        )
        self._sampling_rng = torch.Generator(device=runtime.device)
        self._sampling_rng.manual_seed(torch.seed())

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

    def _acquire_adapter_slot(self, adapter_id: Optional[str]) -> int:
        """Acquire a LoRA slot for an adapter at admission time.

        Args:
            adapter_id: The adapter identifier, or None for no LoRA.

        Returns:
            The slot index (0 for no LoRA, >0 for active adapter).
        """
        if self._adapter_provider is None or adapter_id is None:
            return 0
        adapter = self._adapter_provider.get(adapter_id)
        return self.runtime.acquire_adapter_slot(adapter_id, adapter)

    def _fail_request_early(self, request: GenerationRequest, exc: Exception) -> None:
        """Fail a request that couldn't be admitted (e.g., adapter load failure)."""
        _LOGGER.exception(
            "Failed to admit request %s: %s", request.request_id, exc
        )
        now = time.perf_counter()
        metrics = RequestMetrics(
            prompt_tokens=request.prompt_length,
            decode_tokens=0,
            prefill_time_ms=0.0,
            ttft_ms=max((now - request.submitted_at) * 1000.0, 0.0),
            decode_time_ms=0.0,
        )
        result = SchedulerResult(
            request_id=request.request_id,
            tokens=[],
            finish_reason="error",
            metrics=metrics,
            output={"error": "Request failed during admission"},
        )
        self._completed.append(result)

    def _try_prefill(self) -> bool:
        progress = False
        while len(self.waiting) and len(self.running) < self.runtime.max_batch_size:
            request = self.waiting.peek()
            if request is None:
                break
            if not self.runtime.can_reserve(request.target_length):
                break

            request = self.waiting.pop()

            # Acquire adapter slot at admission time (not earlier)
            try:
                lora_slot = self._acquire_adapter_slot(request.adapter)
            except Exception as exc:
                self._fail_request_early(request, exc)
                progress = True
                continue
            request.lora_slot = lora_slot

            # If this is a segmentation request with spatial refs, convert
            # placeholder coord/size ids into typed CoordToken/SizeToken
            # so the runtime embeds region features during prefill.
            try:
                prompt_inputs: Tensor | list[Token]
                ctx = request.request_context
                if isinstance(ctx, SegmentRequest) and ctx.spatial_refs:
                    prompt_inputs = prompt_with_spatial_tokens(
                        request.prompt_tokens,
                        self._coord_id,
                        self._size_id,
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
                    lora_slot=request.lora_slot,
                )
            except Exception as exc:
                # Release slot on failure to prevent leak
                self.runtime.release_adapter_slot(lora_slot)
                self._fail_request_early(request, exc)
                progress = True
                continue
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
            sampled_ids, temps, top_ps = self._sample_batch(
                first_logits.unsqueeze(0), [seq.request]
            )
            hidden_last = seq.state.last_hidden
            if hidden_last is None:  # pragma: no cover - defensive
                raise RuntimeError("Missing last_hidden after prefill")
            coord_out = self._decode_coord_values[:1]
            size_out = self._decode_size_values[:1]
            coord_decode, size_decode = compute_spatial_values(
                sampled_ids.view(-1),
                hidden_last,
                [seq.request],
                self.runtime.spatial_tables,
                temperatures=temps,
                top_ps=top_ps,
                out_coord=coord_out,
                out_size=size_out,
                rng=self._sampling_rng,
            )
            batch_idx = seq.state.batch_idx
            self._pending_token_ids[batch_idx].copy_(sampled_ids.view(-1)[0])
            self._pending_coord_values[batch_idx].copy_(coord_decode[0])
            self._pending_size_values[batch_idx].copy_(size_decode[0])

            transfer = self._render_buffer.transfer(sampled_ids, coord_decode, size_decode)
            token_ids_cpu, coord_cpu, size_cpu = transfer.wait()
            token = render_tokens_from_packed(
                token_ids_cpu, coord_cpu, size_cpu,
                coord_id=self._coord_id, size_id=self._size_id,
            )[0]
            seq.stage_token(self.runtime, token)

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

        batch_size = len(active)
        idx_np = self._decode_batch_idx.np
        for i, seq in enumerate(active):
            idx_np[i] = seq.state.batch_idx
        batch_idx = self._decode_batch_idx.copy_to_gpu(batch_size)
        token_ids = self._decode_token_ids[:batch_size]
        coord_values = self._decode_coord_values[:batch_size]
        size_values = self._decode_size_values[:batch_size]
        torch.index_select(self._pending_token_ids, 0, batch_idx, out=token_ids)
        torch.index_select(self._pending_coord_values, 0, batch_idx, out=coord_values)
        torch.index_select(self._pending_size_values, 0, batch_idx, out=size_values)

        logits, hidden_last = self.runtime.decode_batch(
            [seq.state for seq in active],
            token_ids,
            coord_values,
            size_values,
        )

        requests = [seq.request for seq in active]
        sampled_ids, temps, top_ps = self._sample_batch(logits, requests)
        coord_decode, size_decode = compute_spatial_values(
            sampled_ids,
            hidden_last,
            requests,
            self.runtime.spatial_tables,
            temperatures=temps,
            top_ps=top_ps,
            out_coord=coord_values,
            out_size=size_values,
            rng=self._sampling_rng,
        )
        self._pending_token_ids.index_copy_(0, batch_idx, sampled_ids)
        self._pending_coord_values.index_copy_(0, batch_idx, coord_decode)
        self._pending_size_values.index_copy_(0, batch_idx, size_decode)
        transfer = self._render_buffer.transfer(sampled_ids, coord_decode, size_decode)

        # Overlap advances/length bookkeeping with the host copy.
        for seq in active:
            seq.state.advance()

        token_ids_cpu, coord_cpu, size_cpu = transfer.wait()
        tokens = render_tokens_from_packed(
            token_ids_cpu, coord_cpu, size_cpu,
            coord_id=self._coord_id, size_id=self._size_id,
        )

        for seq, token in zip(active, tokens):
            seq.stage_token(self.runtime, token)
            if not self._mark_finished_if_needed(seq):
                idle.append(seq)

        self.running.extend(idle)
        return True

    def _sample_batch(
        self, logits: Tensor, requests: List[GenerationRequest]
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch = len(requests)
        if batch == 0:
            empty = torch.empty(0, dtype=torch.long, device=logits.device)
            return empty, None, None

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
            return torch.argmax(logits, dim=-1), None, None

        temps_cpu = torch.empty(batch, dtype=torch.float32)
        top_ps_cpu = torch.empty(batch, dtype=torch.float32)
        for i, req in enumerate(requests):
            temps_cpu[i] = req.temperature
            top_ps_cpu[i] = req.top_p
        temps = temps_cpu.to(device=logits.device)
        top_ps = top_ps_cpu.to(device=logits.device)
        sampled_raw = sample_tokens(logits, temps, top_ps, generator=self._sampling_rng)
        if sampled_raw.dtype == torch.long:
            return sampled_raw, temps, top_ps

        sampled = self._sampled_token_ids[:batch]
        sampled.copy_(sampled_raw, non_blocking=True)
        return sampled, temps, top_ps

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
