"""Flexible batching scheduler for Moondream text inference."""


from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import time
import logging

import torch
from torch import Tensor

from kestrel.moondream.runtime import (
    MoondreamRuntime,
    PreparedSequence,
    PrefillSlot,
    Token,
    TextToken,
)
from kestrel.moondream.lora import AdapterProvider
from kestrel.skills import (
    QuerySkill,
    SkillRegistry,
    SkillState,
)

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestLifecycle,
    RequestMetrics,
    RequestPhase,
    SchedulerResult,
    StepPlan,
)
from .pipeline import (
    DecodeLaunch,
    DecodePendingCommit,
    LaunchHandle,
    PendingCommit,
    PipelineState,
    PrefillLaunch,
    PrefillPendingCommit,
)
from .sampling import sample_tokens
from .transfer import RenderBuffer
from .tokens import render_tokens_from_packed
from .spatial import compute_spatial_values


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PrefillStaging:
    """Per-prefill staging buffers and RenderBuffer for D2H."""

    sampled_ids: Tensor
    coord_staging: Tensor
    size_staging: Tensor
    render: RenderBuffer


@dataclass(slots=True)
class PreparedPrefill:
    """CPU-prepared prefill that is ready for GPU launch.

    Holds the request plus any runtime-side prepared state (KV allocation,
    prefix-cache lookup/locks, etc.) so the actual pipeline "launch" can be
    limited to GPU enqueue work (embed/image encode/prefill forward).
    """

    request: GenerationRequest
    prepared: PreparedSequence
    acquired_lora: bool


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
        self.running: RunningQueue[RequestLifecycle] = RunningQueue()
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
            (runtime.max_batch_slots,),
            dtype=torch.long,
            device=runtime.device,
        )
        self._pending_coord_values = torch.zeros(
            (runtime.max_batch_slots, 1),
            dtype=coord_dtype,
            device=runtime.device,
        )
        self._pending_size_values = torch.zeros(
            (runtime.max_batch_slots, 2),
            dtype=size_dtype,
            device=runtime.device,
        )
        # Persistent per-batch sampling parameters on GPU to avoid per-step
        # CPU->GPU copies during sampling/spatial decode.
        self._sampling_temps_by_batch = torch.zeros(
            (runtime.max_batch_slots,),
            dtype=torch.float32,
            device=runtime.device,
        )
        self._sampling_top_ps_by_batch = torch.ones(
            (runtime.max_batch_slots,),
            dtype=torch.float32,
            device=runtime.device,
        )
        # Scratch buffers for per-step gathers (max effective batch size).
        self._sampling_temps = torch.empty(
            (runtime.max_batch_size,),
            dtype=torch.float32,
            device=runtime.device,
        )
        self._sampling_top_ps = torch.empty(
            (runtime.max_batch_size,),
            dtype=torch.float32,
            device=runtime.device,
        )
        # Prefill staging pool (size matches max in-flight prefills).
        self._prefill_staging_pool: Deque[PrefillStaging] = deque(
            [
                PrefillStaging(
                    sampled_ids=torch.empty(
                        (1,), dtype=torch.long, device=runtime.device
                    ),
                    coord_staging=torch.empty(
                        (1, 1), dtype=coord_dtype, device=runtime.device
                    ),
                    size_staging=torch.empty(
                        (1, 2), dtype=size_dtype, device=runtime.device
                    ),
                    render=RenderBuffer(
                        1,
                        runtime.device,
                        coord_dtype=coord_dtype,
                        size_dtype=size_dtype,
                        copy_stream=runtime.copy_stream,
                    ),
                )
                for _ in range(len(runtime.prefill_slots))
            ]
        )
        self._sampling_rng = torch.Generator(device=runtime.device)
        self._sampling_rng.manual_seed(torch.seed())
        self._pipeline = PipelineState()
        self._prepared_prefill: PreparedPrefill | None = None

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
        """Return ``True`` if there is anything left to prefill, decode, or complete."""
        return (
            len(self.waiting) > 0
            or len(self.running) > 0
            or self._prepared_prefill is not None
            or not self._pipeline.is_empty()
        )

    def advance(self) -> bool:
        """Attempt to make progress using the pipelined decode loop.

        Returns ``True`` if any state changed (e.g. tokens decoded, new
        sequences admitted). Callers can keep invoking ``advance`` while it
        returns ``True`` to drain ready work before sleeping.

        Phase 1 ordering (design doc §5):
        1. Launch forward (if none in-flight) - forward doesn't need mask
        2. Commit previous step (updates skill state via consume_step)
        3. Finalize sampling (compute mask with updated skill state)

        This is "commit-before-finalize" which ensures constrained decoding
        sees the correct skill state. Forward t+1 runs on GPU while CPU
        commits step t, achieving overlap.
        """
        progressed = False
        pipeline = self._pipeline
        prefill_admissible = self._is_prefill_admissible()
        has_prepared = self._prepared_prefill is not None

        has_launch = pipeline.has_launch_in_flight()
        has_queued = pipeline.queue_depth() > 0

        # Only enter stream context if there's GPU work to do.
        if has_launch or has_queued or len(self.running) > 0 or prefill_admissible or has_prepared:
            with torch.cuda.stream(self.runtime.primary_stream):
                # 1. Launch forward if none in-flight (forward doesn't need mask)
                if pipeline.can_launch():
                    launched_forward = False
                    if prefill_admissible or has_prepared:
                        progressed |= self._launch_prefill_step(pipeline)
                        launched_forward = pipeline.has_launch_in_flight()
                    if not launched_forward:
                        plan = self.schedule_decode_step()
                        if plan is not None:
                            slot_id = pipeline.free_slot_id()
                            if slot_id is None:  # pragma: no cover - defensive
                                raise AssertionError(
                                    "Pipeline reported can_launch but no free slot_id"
                                )
                            handle = self.launch_forward_async(plan, slot_id)
                            pipeline.on_launch(handle)
                            progressed = True

                # Prepare next prefill while the GPU forward is in flight.
                if pipeline.has_launch_in_flight():
                    progressed |= self._maybe_prepare_prefill(pipeline)

                # 2. Commit previous step (updates skill state for mask computation)
                # GPU runs forward in parallel while CPU blocks on D2H.
                oldest = pipeline.pop_oldest()
                if oldest is not None:
                    self.commit_step(oldest)
                    pipeline.on_step_completed()
                    progressed = True

                # 3. Finalize sampling (now skill state is updated for mask)
                handle = pipeline.launch_handle
                if handle is not None:
                    step = self.finalize_sampling(handle)
                    pipeline.on_pending_commit(step)
                    progressed = True

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

    def _drain_pipeline(self) -> None:
        """Drain the pipeline before prefill - complete all in-flight work.

        Respects Phase 1 commit-before-finalize ordering: complete all queued
        steps before finalizing any in-flight forward. This ensures grammar
        state is updated before computing masks for constrained decoding.
        """
        pipeline = self._pipeline

        # 1. Complete all queued steps first (commit-before-finalize)
        while True:
            step = pipeline.pop_oldest()
            if step is None:
                break
            self.commit_step(step)
            pipeline.on_step_completed()

        # 2. Finalize any in-flight forward (now safe after all commits)
        if pipeline.has_launch_in_flight():
            handle = pipeline.launch_handle
            step = self.finalize_sampling(handle)
            pipeline.on_pending_commit(step)

            # 3. Complete the final step
            step = pipeline.pop_oldest()
            if step is not None:
                self.commit_step(step)
                pipeline.on_step_completed()

        # Drop any CPU-prepared prefill so pause/shutdown sees a clean runtime state.
        self._abort_prepared_prefill(requeue=True)

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
        lifecycle = request.lifecycle
        lifecycle.finish_reason = "error"
        lifecycle.error = exc
        lifecycle.finished = True
        lifecycle.finalized = True
        lifecycle.transition(RequestPhase.COMPLETED)
        metrics = lifecycle.build_metrics(decode_tokens=0, cached_tokens=0)
        result = SchedulerResult(
            request_id=request.request_id,
            tokens=[],
            finish_reason="error",
            metrics=metrics,
            output={"error": "Request failed during admission"},
        )
        self._completed.append(result)

    def _is_prefill_admissible(self) -> bool:
        """Check if the next pending prefill can be admitted (full check).

        Returns True if:
        - There's a waiting request
        - Batch has room (accounting for zombies via active_sequences)
        - KV cache can reserve space for the request

        This is used to gate pipeline draining per design doc §4.5:
        only drain when prefill would actually proceed.
        """
        request = self.waiting.peek()
        if request is None:
            return False
        lifecycle = request.lifecycle
        if lifecycle.phase not in (
            RequestPhase.WAITING_RESOURCES,
            RequestPhase.READY_FOR_PREFILL,
        ):
            return False
        if (
            lifecycle.has_image
            and not lifecycle.prefix_cache_hit
            and not lifecycle.crops_ready
        ):
            return False
        reserve_length = request.target_length
        if lifecycle.has_image and lifecycle.prefix_cache_hit:
            # Prefix-cache hits are guaranteed (by runtime.check_prefix_cache) to cover at
            # least BOS + the full image prefix, so admission can be less pessimistic
            # than reserving the entire target length from scratch.
            reserve_length = max(
                reserve_length - (1 + self.runtime.image_prefix_length),
                1,
            )
        if not self.runtime.can_reserve(reserve_length):
            return False
        return True

    def _can_prepare_prefill(self) -> bool:
        """Check if we can prepare the next prefill.

        Returns True if:
        - There's a waiting request
        - Request is in the right phase
        - Image crops are ready (if needed)
        - KV cache has enough pages and a batch slot is available

        Note: We allow preparing one prefill ahead even at max decode capacity.
        The +1 extra batch slot (max_batch_slots = max_batch_size + 2) enables this.
        """
        request = self.waiting.peek()
        if request is None:
            return False
        lifecycle = request.lifecycle
        if lifecycle.phase not in (
            RequestPhase.WAITING_RESOURCES,
            RequestPhase.READY_FOR_PREFILL,
        ):
            return False
        if (
            lifecycle.has_image
            and not lifecycle.prefix_cache_hit
            and not lifecycle.crops_ready
        ):
            return False
        reserve_length = request.target_length
        if lifecycle.has_image and lifecycle.prefix_cache_hit:
            reserve_length = max(
                reserve_length - (1 + self.runtime.image_prefix_length),
                1,
            )
        # Check pages and batch slot availability (slot allocated during prepare)
        if not self.runtime.can_reserve(reserve_length):
            return False
        return True

    def _acquire_prefill_staging(self) -> PrefillStaging:
        """Acquire a prefill staging bundle."""
        if not self._prefill_staging_pool:
            raise RuntimeError("Prefill staging pool exhausted")
        return self._prefill_staging_pool.popleft()

    def _release_prefill_staging(self, staging: PrefillStaging) -> None:
        """Return a prefill staging bundle to the pool."""
        self._prefill_staging_pool.append(staging)

    def _abort_prepared_prefill(self, *, requeue: bool) -> None:
        prepared_prefill = self._prepared_prefill
        if prepared_prefill is None:
            return
        self._prepared_prefill = None

        request = prepared_prefill.request
        lifecycle = request.lifecycle
        prepared_seq = prepared_prefill.prepared
        # Note: No prefill_slot to release here - slots are acquired at launch time,
        # not at prepare time. PreparedSequence is decoupled from PrefillSlot.
        try:
            self.runtime.abort_prepared_sequence(prepared_seq)
        finally:
            if prepared_prefill.acquired_lora:
                try:
                    self.runtime.release_adapter_slot(request.lora_slot)
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
                request.lora_slot = 0
                lifecycle.lora_slot_ready = False

        # Reset timing for an aborted attempt.
        lifecycle.prefill_started_at = None
        lifecycle.prefill_completed_at = None

        if requeue:
            phase = (
                RequestPhase.READY_FOR_PREFILL
                if (lifecycle.crops_ready and lifecycle.lora_slot_ready)
                else RequestPhase.WAITING_RESOURCES
            )
            lifecycle.transition(phase)
            self.waiting.push(request)
        else:
            lifecycle.finish_reason = "error"
            lifecycle.error = RuntimeError("Prepared prefill aborted")
            lifecycle.finished = True
            lifecycle.finalized = True
            lifecycle.transition(RequestPhase.COMPLETED)
            metrics = lifecycle.build_metrics(decode_tokens=0, cached_tokens=0)
            result = SchedulerResult(
                request_id=request.request_id,
                tokens=[],
                finish_reason="error",
                metrics=metrics,
                output={"error": "Prepared prefill aborted"},
            )
            self._completed.append(result)

    def _prepare_prefill(self) -> bool:
        """Perform heavy prefill preparation for the next waiting request.

        This does KV allocation/reservation + prefix-cache lookup ahead of time so
        the pipeline "launch" can be limited to GPU enqueue work.

        Note: PrefillSlot is NOT acquired here. It is acquired at launch time,
        allowing preparation to proceed even when all prefill slots are occupied
        by in-flight prefills.

        Note: Uses _can_prepare_prefill() which skips batch slot availability
        check. The page table has an extra slot reserved for preparation, so we
        can always prepare even at max batch size.
        """

        if self._prepared_prefill is not None:
            return False
        if not self._can_prepare_prefill():
            return False

        request = self.waiting.pop()
        lifecycle = request.lifecycle

        acquired_lora = False
        if not lifecycle.lora_slot_ready:
            try:
                lora_slot = self._acquire_adapter_slot(request.adapter)
            except Exception as exc:
                self._fail_request_early(request, exc)
                return True
            request.lora_slot = lora_slot
            lifecycle.lora_slot_ready = True
            acquired_lora = True

        if lifecycle.phase == RequestPhase.WAITING_RESOURCES:
            lifecycle.transition(RequestPhase.READY_FOR_PREFILL)

        try:
            prompt_inputs = request.prompt_tokens
            prefill_start = time.perf_counter()
            lifecycle.prefill_started_at = prefill_start
            lifecycle.transition(RequestPhase.PREFILLING)

            prepared = self.runtime.prepare_sequence(
                prompt_tokens=prompt_inputs,
                image=request.image,
                image_crops=request.image_crops,
                max_new_tokens=request.max_new_tokens,
                lora_slot=request.lora_slot,
                image_hash=request.image_hash,
                adapter_id=request.adapter,
            )
        except Exception as exc:
            if acquired_lora:
                self.runtime.release_adapter_slot(request.lora_slot)
                request.lora_slot = 0
                lifecycle.lora_slot_ready = False
            self._fail_request_early(request, exc)
            return True

        seq = lifecycle
        seq.sequence_state = prepared.state

        # Persist per-sequence sampling parameters for later decode steps.
        batch_idx = prepared.state.batch_idx
        with torch.cuda.stream(self.runtime.primary_stream):
            self._sampling_temps_by_batch[batch_idx] = seq.request.temperature
            self._sampling_top_ps_by_batch[batch_idx] = seq.request.top_p

        self._prepared_prefill = PreparedPrefill(
            request=request,
            prepared=prepared,
            acquired_lora=acquired_lora,
        )
        return True

    def _maybe_prepare_prefill(self, pipeline: PipelineState) -> bool:
        """Prepare a prefill while a GPU forward is in flight.

        Since PreparedSequence is decoupled from PrefillSlot, we can prepare
        even when all prefill slots are occupied by in-flight prefills. The
        slot is acquired at launch time instead.
        """
        if self._prepared_prefill is not None:
            return False
        if not pipeline.has_launch_in_flight():
            return False

        return self._prepare_prefill()

    def _finalize_prefill(self, handle: LaunchHandle) -> PendingCommit:
        """Sample first token + start D2H for a prefill.

        IMPORTANT: Caller must already be on the primary stream.
        """
        if handle.kind != "prefill":
            raise AssertionError("prefill finalize requires a prefill handle")
        prefill_payload = handle.payload
        if not isinstance(prefill_payload, PrefillLaunch):
            raise AssertionError("prefill finalize requires a PrefillLaunch payload")
        staging = prefill_payload.staging
        logits = prefill_payload.logits
        prepared_seq = prefill_payload.prepared
        prefill_slot = prefill_payload.prefill_slot
        seq = handle.sequences[0]

        first_logits = logits.squeeze(0)
        sampled_ids, temps, top_ps = self._sample_batch(
            first_logits.unsqueeze(0),
            [seq],
            staging.sampled_ids,
            batch_idx=prefill_slot.batch_idx,
        )
        hidden_last = seq.state.last_hidden
        if hidden_last is None:  # pragma: no cover - defensive
            raise RuntimeError("Missing last_hidden after prefill")
        coord_out = staging.coord_staging[:1]
        size_out = staging.size_staging[:1]
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
        prefill_slot.step_done_event.record()
        batch_idx = seq.state.batch_idx
        self._pending_token_ids[batch_idx].copy_(sampled_ids.view(-1)[0])
        self._pending_coord_values[batch_idx].copy_(coord_decode[0])
        self._pending_size_values[batch_idx].copy_(size_decode[0])
        prefill_slot.commit_done_event.record()
        seq.packed_pending_ready = True
        seq.uncommitted_prefill_token = True

        # Mark the sequence runnable immediately after the first token is sampled.
        # This allows decode forward to overlap with the CPU commit of token0.
        self.running.push(seq)

        transfer = staging.render.transfer(
            sampled_ids, coord_decode, size_decode, ready_event=prefill_slot.step_done_event
        )
        return PendingCommit(
            kind="prefill",
            sequences=handle.sequences,
            transfer=transfer,
            payload=PrefillPendingCommit(
                staging=staging,
                slot_id=prefill_payload.slot_id,
                prepared=prepared_seq,
                prefill_slot=prefill_slot,
            ),
        )

    def _commit_prefill(self, step: PendingCommit) -> Token:
        """Commit a prefill PendingCommit and return the first token."""
        if step.kind != "prefill":
            raise AssertionError("prefill commit requires a prefill pending commit")
        payload = step.payload
        if not isinstance(payload, PrefillPendingCommit):
            raise AssertionError("prefill commit requires a PrefillPendingCommit payload")
        staging = payload.staging
        prepared_seq = payload.prepared
        prefill_slot = payload.prefill_slot
        try:
            token_ids_cpu, coord_cpu, size_cpu = step.transfer.wait()
            prefill_slot.commit_done_event.synchronize()
            self.runtime.finalize_prepared_sequence_after_prefill(prepared_seq)
            token = render_tokens_from_packed(
                token_ids_cpu,
                coord_cpu,
                size_cpu,
                coord_id=self._coord_id,
                size_id=self._size_id,
            )[0]
        finally:
            self._release_prefill_staging(staging)
            self.runtime.release_prefill_slot(prefill_slot)
        return token

    def _launch_prefill_step(self, pipeline: PipelineState) -> bool:
        """Launch a single prefill forward and enqueue it into the shared pipeline.

        Returns True if any progress was made (request admitted, failed early,
        or prefill launched), False otherwise.
        """
        if pipeline.has_launch_in_flight():
            return False
        if pipeline.committing_step is not None:
            return False

        # Need a free pipeline slot to launch
        slot_id = pipeline.free_slot_id()
        if slot_id is None:
            return False

        # Don't launch if decode batch is at capacity
        if len(self.running) >= self.runtime.max_batch_size:
            return False

        # Get or create a prepared prefill
        prepared_prefill = self._prepared_prefill
        if prepared_prefill is None:
            progressed = self._prepare_prefill()
            if not progressed:
                return False
            prepared_prefill = self._prepared_prefill
            if prepared_prefill is None:
                # Preparation may have failed the request early; that's still progress.
                return True

        assert prepared_prefill is not None
        request = prepared_prefill.request
        lifecycle = request.lifecycle
        seq = lifecycle
        prepared_seq = prepared_prefill.prepared
        acquired_lora = prepared_prefill.acquired_lora

        # Acquire prefill slot at launch time (decoupled from prepare)
        prefill_slot = self.runtime.acquire_prefill_slot(slot_id)

        def fail_prepared(
            exc: Exception,
            *,
            staging: PrefillStaging | None = None,
            release_slot: bool = True,
        ) -> bool:
            if staging is not None:
                self._release_prefill_staging(staging)
            try:
                self.runtime.abort_prepared_sequence(prepared_seq)
            finally:
                if release_slot:
                    try:
                        self.runtime.release_prefill_slot(prefill_slot)
                    except Exception:
                        pass
                if acquired_lora:
                    try:
                        self.runtime.release_adapter_slot(request.lora_slot)
                    except Exception:
                        pass
                    request.lora_slot = 0
                    lifecycle.lora_slot_ready = False
                self._prepared_prefill = None
            self._fail_request_early(request, exc)
            return True

        # For 0-length generation, we don't need token0 sampling/D2H staging.
        if request.max_new_tokens <= 0:
            try:
                self.runtime.launch_prepared_sequence(
                    prepared_seq,
                    prefill_slot,
                    image=request.image,
                    image_crops=request.image_crops,
                )
                lifecycle.prefill_completed_at = time.perf_counter()
                # Ensure prefill forward completes before cache finalize + release.
                with torch.cuda.stream(self.runtime.primary_stream):
                    prefill_slot.commit_done_event.record()
                prefill_slot.commit_done_event.synchronize()
                self.runtime.finalize_prepared_sequence_after_prefill(prepared_seq)
            except Exception as exc:
                return fail_prepared(exc)

            self._prepared_prefill = None
            seq.first_token_time = lifecycle.prefill_started_at or time.perf_counter()
            self._finalize_sequence(seq, "length")
            self.runtime.release_prefill_slot(prefill_slot)
            return True

        try:
            staging = self._acquire_prefill_staging()
        except Exception:
            return False

        try:
            logits = self.runtime.launch_prepared_sequence(
                prepared_seq,
                prefill_slot,
                image=request.image,
                image_crops=request.image_crops,
            )
            lifecycle.prefill_completed_at = time.perf_counter()
        except Exception as exc:
            return fail_prepared(exc, staging=staging)

        # Prefill is enqueued; mark prepared slot consumed.
        self._prepared_prefill = None

        # Prefill has completed (KV/logits enqueued); notify skill.
        seq.skill_state.on_prefill(self.runtime)

        handle = LaunchHandle(
            kind="prefill",
            sequences=[seq],
            payload=PrefillLaunch(
                staging=staging,
                slot_id=slot_id,
                logits=logits,
                prepared=prepared_seq,
                prefill_slot=prefill_slot,
            ),
        )
        pipeline.on_launch(handle)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Split decode API (Phase 1 pipelining)
    # ──────────────────────────────────────────────────────────────────────────

    def _can_dispatch(self, seq: RequestLifecycle) -> bool:
        """Check if a sequence can be included in the next decode step.

        A sequence is dispatchable if:
        - Not already finalized (EOS/length cap reached)
        - Has fewer than 2 in-flight references (pipelining limit)
        - Won't exceed its length budget if dispatched

        This is a pure predicate - it does not mutate any state.
        """
        if seq.finalized:
            return False
        if seq.inflight_refs >= 2:
            return False
        # Absolute max length (includes prompt)
        if seq.state.length >= seq.state.max_length:
            return False
        # Max new tokens budget - account for in-flight steps
        if seq.request.max_new_tokens is not None:
            committed = seq.skill_state.token_count
            if (
                committed
                + seq.inflight_refs
                + (1 if seq.uncommitted_prefill_token else 0)
                >= seq.request.max_new_tokens
            ):
                return False
        return True

    def schedule_decode_step(self) -> Optional[StepPlan]:
        """Select sequences for the next decode step.

        This is a pure selector that examines the running queue and returns a
        StepPlan containing sequences ready for decoding, or None if no work.

        Per design doc §4.7: This method does NOT finalize sequences based on
        GPU-progress (seq.state.length). Finalization happens in commit_step()
        after the token is committed, using committed counts. The _can_dispatch()
        predicate uses budgeted counts to exclude sequences that would exceed
        their limits if dispatched.
        """
        if not len(self.running):
            return None

        active: list[RequestLifecycle] = []
        for seq in self.running:
            if not seq.needs_decode():
                continue
            if not self._can_dispatch(seq):
                continue
            active.append(seq)

        if not active:
            return None

        return StepPlan(sequences=active)

    def launch_forward_async(
        self, plan: StepPlan, slot_id: int
    ) -> LaunchHandle:
        """Launch the forward pass for a decode step (with stream context).

        Wrapper that enters the compute stream context before calling
        _launch_forward_on_stream. Use this when calling from outside advance().
        """
        slot = self.runtime.decode_slots[slot_id]
        with torch.cuda.stream(slot.compute_stream):
            return self._launch_forward_on_stream(plan, slot_id)

    def _launch_forward_on_stream(
        self, plan: StepPlan, slot_id: int
    ) -> LaunchHandle:
        """Launch the forward pass for a decode step.

        IMPORTANT: Caller must already be on the compute stream.

        This increments inflight_refs for each sequence (committing them to this
        step), gathers inputs, and runs the model forward pass. The forward
        outputs (logits, hidden_last) are stored in the DecodeSlot's buffers
        for later retrieval by finalize_sampling.

        Returns a LaunchHandle that can be passed to finalize_sampling.
        """
        sequences = plan.sequences
        batch_size = len(sequences)
        slot = self.runtime.decode_slots[slot_id]

        # Commit sequences to this step (rollback if forward fails)
        for seq in sequences:
            seq.inflight_refs += 1
            seq.packed_pending_ready = False

        try:
            # Prepare all CPU metadata buffers
            idx_np = slot.meta.batch_idx.np
            pos_np = slot.meta.input_pos.np
            lora_np = slot.meta.lora_slot_ids.np
            for i, seq in enumerate(sequences):
                idx_np[i] = seq.state.batch_idx
                pos_np[i] = seq.state.length
                lora_np[i] = seq.state.lora_slot

            # H2D copies for all metadata
            batch_idx = slot.meta.batch_idx.copy_to_gpu(batch_size)
            slot.meta.input_pos.copy_to_gpu(batch_size)
            slot.meta.lora_slot_ids.copy_to_gpu(batch_size)

            # Gather decode inputs from _pending_* into slot staging buffers
            token_ids = slot.decode_token_ids[:batch_size]
            coord_values = slot.decode_coord_values[:batch_size]
            size_values = slot.decode_size_values[:batch_size]
            torch.index_select(self._pending_token_ids, 0, batch_idx, out=token_ids)
            torch.index_select(self._pending_coord_values, 0, batch_idx, out=coord_values)
            torch.index_select(self._pending_size_values, 0, batch_idx, out=size_values)

            # Commit page table for all batch indices before forward pass (deferred H2D sync)
            batch_indices_list = [seq.state.batch_idx for seq in sequences]
            self.runtime.page_table.commit_block_table(batch_indices_list)

            # Run forward pass - writes to slot.logits and slot.hidden_last
            self.runtime.decode_with_slot(slot, batch_size)
        except Exception:
            # Rollback inflight_refs on failure
            for seq in sequences:
                seq.inflight_refs -= 1
                seq.packed_pending_ready = True
            raise

        # Advance sequence states (KV length) immediately after forward dispatch.
        # Per design doc §4.5: length tracks GPU progress, not CPU commit.
        for seq in sequences:
            seq.state.advance()

        return LaunchHandle(
            kind="decode",
            sequences=sequences,
            payload=DecodeLaunch(slot_id=slot_id),
        )

    def finalize_sampling(
        self, handle: LaunchHandle, mask: Optional[Tensor] = None
    ) -> PendingCommit:
        """Finalize sampling for a forward pass (with stream context).

        Wrapper that enters the compute stream context before calling
        _finalize_sampling_on_stream. Use this when calling from outside advance().
        """
        if handle.kind == "decode":
            slot = self.runtime.decode_slots[handle.slot_id]
            with torch.cuda.stream(slot.compute_stream):
                return self._finalize_sampling_on_stream(handle, mask)
        if handle.kind == "prefill":
            if mask is not None:
                raise AssertionError("Prefill finalize does not support masks")
            with torch.cuda.stream(self.runtime.primary_stream):
                return self._finalize_prefill(handle)
        raise AssertionError(f"Unsupported handle kind {handle.kind!r}")

    def _finalize_sampling_on_stream(
        self, handle: LaunchHandle, mask: Optional[Tensor] = None
    ) -> PendingCommit:
        """Finalize sampling for a forward pass and start D2H transfer.

        IMPORTANT: Caller must already be on the compute stream.

        Takes the forward outputs from the DecodeSlot's buffers, applies optional
        token mask, samples tokens, computes spatial values, writes to pending
        buffers, and kicks off async D2H via the slot's RenderBuffer.

        The mask parameter is for future constrained decoding support. Currently
        masking is computed inline from skill_state.allowed_token_ids.

        Returns a PendingCommit that can be passed to commit_step.
        """
        sequences = handle.sequences
        batch_size = len(sequences)
        slot = self.runtime.decode_slots[handle.slot_id]

        # Read forward outputs from slot's buffers
        logits = slot.logits[:batch_size]
        hidden_last = slot.hidden_last[:batch_size]

        # Reuse batch indices already copied in launch_forward_async
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]

        # Sample tokens directly into per-slot staging buffer for D2H.
        # This prevents race with next step's sampling writing to shared buffer.
        sampled_ids, temps, top_ps = self._sample_batch(
            logits, sequences, slot.sampled_ids, batch_idx=batch_idx
        )

        # Compute spatial values into slot's staging buffers
        coord_staging = slot.coord_staging[:batch_size]
        size_staging = slot.size_staging[:batch_size]
        coord_decode, size_decode = compute_spatial_values(
            sampled_ids,
            hidden_last,
            [seq.request for seq in sequences],
            self.runtime.spatial_tables,
            temperatures=temps,
            top_ps=top_ps,
            out_coord=coord_staging,
            out_size=size_staging,
            rng=self._sampling_rng,
        )

        # Record event after staging buffers are ready.
        #
        # The copy stream waits on this event, so D2H can overlap with the
        # writes to `_pending_*` below (these writes are not needed for D2H).
        slot.step_done_event.record()

        # Write to shared pending buffers for next step's input gathering.
        self._pending_token_ids.index_copy_(0, batch_idx, sampled_ids)
        self._pending_coord_values.index_copy_(0, batch_idx, coord_decode)
        self._pending_size_values.index_copy_(0, batch_idx, size_decode)

        # Record a second event after pending writes so we can safely release/reuse
        # batch indices (e.g. finalize a sequence and admit a new one into the same
        # batch slot) without racing the `_pending_*` updates.
        slot.commit_done_event.record()
        for seq in sequences:
            seq.packed_pending_ready = True

        # Start async D2H transfer from per-slot staging buffers.
        # Pass the step_done_event so copy stream waits only on staging writes.
        transfer = slot.render.transfer(
            slot.sampled_ids[:batch_size],
            coord_staging,
            size_staging,
            ready_event=slot.step_done_event,
        )

        return PendingCommit(
            kind="decode",
            sequences=sequences,
            transfer=transfer,
            payload=DecodePendingCommit(slot_id=handle.slot_id),
        )

    def commit_step(self, step: PendingCommit) -> None:
        """Complete a pending step (prefill or decode).

        Blocks until the D2H transfer completes, then materializes tokens and
        commits them to each sequence (calls consume_step, emits streaming).
        Checks for EOS termination and updates finalized state.

        Sequences that become finalized AND have no remaining in-flight refs
        are released immediately. Zombies (finalized with refs > 0) are skipped
        at commit time and released when their last step completes.
        """
        if step.kind == "prefill":
            if len(step.sequences) != 1:
                raise AssertionError("Prefill commit expects exactly one sequence")
            seq = step.sequences[0]
            token = self._commit_prefill(step)
            seq.stage_token(self.runtime, token)
            seq.uncommitted_prefill_token = False
            if self._mark_finished_if_needed(seq):
                seq.finalized = True
                # Prefill sequences are enqueued into `running` immediately after the
                # first token is sampled, so remove on termination here.
                self.running.remove(seq)
                if seq.inflight_refs == 0:
                    seq.transition(RequestPhase.COMPLETED)
                    self._release_sequence(seq)
            return

        if step.kind != "decode":
            raise AssertionError(f"Unsupported commit kind {step.kind!r}")

        # Wait for D2H transfer
        token_ids_cpu, coord_cpu, size_cpu = step.transfer.wait()

        # Ensure `_pending_*` writes for this step have completed before we release
        # or reuse any batch indices (these writes intentionally do not gate D2H).
        slot = self.runtime.decode_slots[step.slot_id]
        slot.commit_done_event.synchronize()

        # Render typed tokens from packed tensors
        tokens = render_tokens_from_packed(
            token_ids_cpu, coord_cpu, size_cpu,
            coord_id=self._coord_id, size_id=self._size_id,
        )

        # Commit each sequence
        for seq, token in zip(step.sequences, tokens):
            seq.inflight_refs -= 1

            # Skip zombies (already finalized in a previous step)
            if seq.finalized:
                if seq.inflight_refs == 0:
                    seq.transition(RequestPhase.COMPLETED)
                    self._release_sequence(seq)
                continue

            # Stage token (calls consume_step, emits streaming)
            seq.stage_token(self.runtime, token)

            # Check for termination
            if self._mark_finished_if_needed(seq):
                seq.finalized = True
                # Remove from running queue
                self.running.remove(seq)
                if seq.inflight_refs == 0:
                    seq.transition(RequestPhase.COMPLETED)
                    self._release_sequence(seq)

    def _release_sequence(self, seq: RequestLifecycle) -> None:
        """Release resources for a finalized sequence with no in-flight refs.

        Called when a zombie's last in-flight reference completes. The sequence
        was finalized earlier (in _mark_finished_if_needed or schedule_decode_step)
        but resource release was deferred because inflight_refs > 0.
        """
        if seq.state.batch_idx in self.runtime.active_sequences:
            self.runtime.release_sequence(seq.state)

    def _sample_batch(
        self,
        logits: Tensor,
        sequences: List[RequestLifecycle],
        out: Tensor,
        *,
        batch_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Sample tokens from logits into the provided output buffer.

        Args:
            logits: Logits tensor of shape [batch, vocab_size].
            sequences: List of scheduled sequences (for temperature/top_p and
                finalized state). Finalized sequences are treated as unconstrained
                to avoid querying skill state after termination.
            out: Pre-allocated output buffer for sampled token IDs. Must be
                shape [batch] or larger, dtype long.

        Returns:
            Tuple of (sampled_ids, temps, top_ps). sampled_ids is a view of
            out[:batch].
        """
        batch = len(sequences)
        if batch == 0:
            return out[:0], None, None

        allowed_tokens: list[Optional[Sequence[int]]] = []
        restrict = False
        for seq in sequences:
            # Skip constraint queries for finalized sequences (zombies).
            # Per design doc §4.6: treat finalized as unconstrained to avoid
            # use-after-finalization bugs.
            if seq.finalized:
                allowed_tokens.append(None)
                continue
            allowed = seq.skill_state.allowed_token_ids(self.runtime)
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

        if all(seq.request.temperature <= 0.0 for seq in sequences):
            torch.argmax(logits, dim=-1, out=out[:batch])
            return out[:batch], None, None

        if batch_idx is None:
            raise AssertionError("batch_idx is required for non-greedy sampling")
        batch_idx = batch_idx.view(-1)[:batch]
        temps = self._sampling_temps[:batch]
        top_ps = self._sampling_top_ps[:batch]
        torch.index_select(self._sampling_temps_by_batch, 0, batch_idx, out=temps)
        torch.index_select(self._sampling_top_ps_by_batch, 0, batch_idx, out=top_ps)
        sampled_raw = sample_tokens(logits, temps, top_ps, generator=self._sampling_rng)
        out[:batch].copy_(sampled_raw)
        return out[:batch], temps, top_ps

    def _mark_finished_if_needed(self, seq: RequestLifecycle) -> bool:
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

    def _finalize_sequence(self, seq: RequestLifecycle, reason: str) -> None:
        """Mark a sequence as finished and prepare its result.

        This marks both `finished` (for result building) and `finalized` (for
        pipelining). Resources are NOT released here if inflight_refs > 0;
        release happens in commit_step() when the last in-flight reference
        completes. This prevents releasing KV cache pages while a zombie step
        is still reading them.
        """
        if seq.finished:
            return
        seq.finished = True
        seq.finalized = True
        seq.finish_reason = reason
        seq.completed_at = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = seq.completed_at

        # Only release immediately if no in-flight steps reference this sequence.
        # Otherwise, release is deferred to commit_step() when inflight_refs hits 0.
        if seq.inflight_refs == 0:
            seq.transition(RequestPhase.COMPLETED)
            if seq.state.batch_idx in self.runtime.active_sequences:
                self.runtime.release_sequence(seq.state)
        else:
            seq.transition(RequestPhase.FINALIZING)

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

    def _build_result(self, seq: RequestLifecycle) -> SchedulerResult:
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

        decode_tokens = len(tokens) if tokens else len(seq.skill_state.tokens)
        metrics = seq.build_metrics(decode_tokens=decode_tokens)
        return SchedulerResult(
            request_id=seq.request.request_id,
            tokens=tokens,
            finish_reason=finish_reason,
            metrics=metrics,
            output=output,
        )
