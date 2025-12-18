"""Flexible batching scheduler for Moondream text inference."""


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
from kestrel.moondream.lora import AdapterProvider
from kestrel.moondream.image_crops import OverlapCropOutput
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
    StreamCallback,
)
from .sampling import sample_tokens


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

class _RenderBuffer:
    """Pinned host buffers for sampled ids + decoded coord/size values."""

    def __init__(
        self,
        max_batch: int,
        device: torch.device,
        *,
        coord_dtype: torch.dtype,
        size_dtype: torch.dtype,
    ) -> None:
        self._token_ids = torch.empty(
            (max_batch,),
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        self._coord_values = torch.empty(
            (max_batch, 1),
            dtype=coord_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._size_values = torch.empty(
            (max_batch, 2),
            dtype=size_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event(enable_timing=False, blocking=False)

    class _TransferHandle:
        def __init__(
            self,
            event: torch.cuda.Event,
            token_ids: Tensor,
            coord_values: Tensor,
            size_values: Tensor,
            count: int,
        ) -> None:
            self._event = event
            self._token_ids = token_ids
            self._coord_values = coord_values
            self._size_values = size_values
            self._count = count

        def wait(self) -> tuple[Tensor, Tensor, Tensor]:
            if self._count == 0:
                empty = self._token_ids[:0]
                return empty, self._coord_values[:0], self._size_values[:0]
            self._event.synchronize()
            return (
                self._token_ids[: self._count],
                self._coord_values[: self._count],
                self._size_values[: self._count],
            )

    def transfer(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
    ) -> "_RenderBuffer._TransferHandle":
        count = int(token_ids.shape[0])
        if count == 0:
            return _RenderBuffer._TransferHandle(
                self._event,
                self._token_ids,
                self._coord_values,
                self._size_values,
                0,
            )

        current = torch.cuda.current_stream(device=self._device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(current)
            self._token_ids[:count].copy_(token_ids, non_blocking=True)
            self._coord_values[:count].copy_(coord_values, non_blocking=True)
            self._size_values[:count].copy_(size_values, non_blocking=True)
            self._event.record(self._stream)
        return _RenderBuffer._TransferHandle(
            self._event, self._token_ids, self._coord_values, self._size_values, count
        )


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
        self._render_buffer = _RenderBuffer(
            runtime.max_batch_size,
            runtime.device,
            coord_dtype=coord_dtype,
            size_dtype=size_dtype,
        )
        coord_bins = runtime.config.region.coord_out_dim
        size_bins = runtime.config.region.size_out_dim // 2
        self._coord_value_lut = torch.linspace(
            0.0,
            1.0,
            coord_bins,
            device=runtime.device,
            dtype=torch.float32,
        ).to(dtype=coord_dtype)
        size_exponents = torch.linspace(
            -10.0,
            0.0,
            size_bins,
            device=runtime.device,
            dtype=torch.float32,
        )
        self._size_value_lut = torch.exp2(size_exponents).to(dtype=size_dtype)
        coord_decoder = runtime.region["coord_decoder"]
        size_decoder = runtime.region["size_decoder"]
        self._coord_logits_dim = int(coord_decoder.out_features)
        self._size_logits_dim = int(size_decoder.out_features)
        self._spatial_decode_weight = torch.cat(
            (coord_decoder.weight, size_decoder.weight), dim=0
        ).contiguous()
        self._spatial_decode_bias = torch.cat(
            (coord_decoder.bias, size_decoder.bias), dim=0
        ).contiguous()
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
        self._coord_bin_ids = torch.empty(
            (runtime.max_batch_size,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._size_bin_ids = torch.empty(
            (runtime.max_batch_size * 2,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._decode_batch_idx = CpuGpuBuffer(
            runtime.max_batch_size,
            dtype=torch.long,
            device=runtime.device,
            pin_memory=True,
        )
        self._flashinfer_rng = torch.Generator(device=runtime.device)
        self._flashinfer_rng.manual_seed(torch.seed())
        self._spatial_rng = torch.Generator(device=runtime.device)
        self._spatial_rng.manual_seed(torch.seed())

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
            coord_decode, size_decode = self._compute_spatial_values(
                sampled_ids.view(-1),
                hidden_last,
                [seq.request],
                temperatures=temps,
                top_ps=top_ps,
                out_coord=coord_out,
                out_size=size_out,
            )
            batch_idx = seq.state.batch_idx
            self._pending_token_ids[batch_idx].copy_(sampled_ids.view(-1)[0])
            self._pending_coord_values[batch_idx].copy_(coord_decode[0])
            self._pending_size_values[batch_idx].copy_(size_decode[0])

            transfer = self._render_buffer.transfer(sampled_ids, coord_decode, size_decode)
            token_ids_cpu, coord_cpu, size_cpu = transfer.wait()
            token = self._render_tokens_from_packed(token_ids_cpu, coord_cpu, size_cpu)[0]
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
        coord_decode, size_decode = self._compute_spatial_values(
            sampled_ids,
            hidden_last,
            requests,
            temperatures=temps,
            top_ps=top_ps,
            out_coord=coord_values,
            out_size=size_values,
        )
        self._pending_token_ids.index_copy_(0, batch_idx, sampled_ids)
        self._pending_coord_values.index_copy_(0, batch_idx, coord_decode)
        self._pending_size_values.index_copy_(0, batch_idx, size_decode)
        transfer = self._render_buffer.transfer(sampled_ids, coord_decode, size_decode)

        # Overlap advances/length bookkeeping with the host copy.
        for seq in active:
            seq.state.advance()

        token_ids_cpu, coord_cpu, size_cpu = transfer.wait()
        tokens = self._render_tokens_from_packed(token_ids_cpu, coord_cpu, size_cpu)

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
        sampled_raw = sample_tokens(logits, temps, top_ps, generator=self._flashinfer_rng)
        if sampled_raw.dtype == torch.long:
            return sampled_raw, temps, top_ps

        sampled = self._sampled_token_ids[:batch]
        sampled.copy_(sampled_raw, non_blocking=True)
        return sampled, temps, top_ps

    def _compute_spatial_values(
        self,
        token_ids: Tensor,
        hidden_last: Tensor,
        requests: List[GenerationRequest],
        *,
        temperatures: Tensor | None = None,
        top_ps: Tensor | None = None,
        out_coord: Tensor,
        out_size: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode coord/size token values from hidden states on GPU."""

        if token_ids.ndim != 1:
            token_ids = token_ids.view(-1)
        batch = int(token_ids.shape[0])
        if batch == 0:
            device = hidden_last.device
            coord_decode = torch.empty((0, 1), device=device, dtype=out_coord.dtype)
            size_decode = torch.empty((0, 2), device=device, dtype=out_size.dtype)
            return coord_decode, size_decode

        hidden = hidden_last.unsqueeze(0) if hidden_last.ndim == 1 else hidden_last

        do_sample = not all(req.temperature <= 0.0 for req in requests)
        if do_sample and (temperatures is None or top_ps is None):
            temps_cpu = torch.tensor(
                [req.temperature for req in requests],
                dtype=torch.float32,
            )
            top_ps_cpu = torch.tensor(
                [req.top_p for req in requests],
                dtype=torch.float32,
            )
            temperatures = temps_cpu.to(device=hidden.device)
            top_ps = top_ps_cpu.to(device=hidden.device)

        spatial_logits = torch.nn.functional.linear(
            hidden, self._spatial_decode_weight, self._spatial_decode_bias
        )
        coord_logits = spatial_logits[:, : self._coord_logits_dim]
        size_flat = spatial_logits[:, self._coord_logits_dim :]
        bins_size = int(size_flat.shape[-1] // 2)
        width_logits = size_flat[:, :bins_size]
        height_logits = size_flat[:, bins_size:]

        if not do_sample:
            coord_bins = torch.argmax(coord_logits, dim=-1)
            width_bins = torch.argmax(width_logits, dim=-1)
            height_bins = torch.argmax(height_logits, dim=-1)
        else:
            if temperatures is None or top_ps is None:  # pragma: no cover - defensive
                raise RuntimeError("Missing sampling parameters for spatial decode")
            coord_bins_raw = sample_tokens(
                coord_logits, temperatures, top_ps, generator=self._spatial_rng
            )
            if coord_bins_raw.dtype == torch.long:
                coord_bins = coord_bins_raw
            else:
                coord_bins = self._coord_bin_ids[:batch]
                coord_bins.copy_(coord_bins_raw, non_blocking=True)
            logits_2 = torch.cat((width_logits, height_logits), dim=0)
            bins_2_raw = sample_tokens(
                logits_2,
                temperatures.repeat(2),
                top_ps.repeat(2),
                generator=self._spatial_rng,
            )
            if bins_2_raw.dtype == torch.long:
                bins_2 = bins_2_raw
            else:
                bins_2 = self._size_bin_ids[: 2 * batch]
                bins_2.copy_(bins_2_raw, non_blocking=True)
            width_bins = bins_2[:batch]
            height_bins = bins_2[batch:]

        coord_out = out_coord[:batch].view(-1)
        size_out = out_size[:batch]
        torch.index_select(self._coord_value_lut, 0, coord_bins, out=coord_out)
        torch.index_select(self._size_value_lut, 0, width_bins, out=size_out[:, 0])
        torch.index_select(self._size_value_lut, 0, height_bins, out=size_out[:, 1])
        return out_coord[:batch], out_size[:batch]

    def _render_tokens_from_packed(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
    ) -> list[Token]:
        """Materialise sampled ids + value tensors into typed tokens on host."""

        ids = token_ids.view(-1).tolist()
        batch = len(ids)
        if batch == 0:
            return []

        coord_id = self.runtime.config.tokenizer.coord_id
        size_id = self.runtime.config.tokenizer.size_id

        out: list[Token] = []
        for i, token_id in enumerate(ids):
            if token_id == coord_id:
                out.append(CoordToken(pos=float(coord_values[i, 0].item())))
            elif token_id == size_id:
                out.append(
                    SizeToken(
                        width=float(size_values[i, 0].item()),
                        height=float(size_values[i, 1].item()),
                    )
                )
            else:
                out.append(TextToken(token_id=token_id))
        return out

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
