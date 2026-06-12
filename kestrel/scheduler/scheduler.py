"""Flexible batching scheduler for Moondream text inference."""


from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import time
import logging

import torch
from torch import Tensor

from kestrel.device import NoopEvent, stream_context
from kestrel.runtime import (
    PrefillClassification,
    PreparedSequence,
    AutoregressiveRuntime,
    TextToken,
    Token,
)
from kestrel.models.moondream.lora import AdapterProvider
from kestrel.skills import (
    SkillRegistry,
    SkillState,
)

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestLifecycle,
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
from kestrel_kernels.sampling import sample_step_from_logits
from .transfer import RenderBuffer
from kestrel.runtime.sampling import SamplingHooks


_LOGGER = logging.getLogger(__name__)
# Greedily build a prefill batch until aggregate query tokens reaches this floor.
_MIN_PREFILL_LAUNCH_TOKENS = 2048
_SAMPLING_EPS = 1e-6
# Prefer seeding miss-frontier work only when launch capacity is large enough to
# absorb the short-term hit latency tradeoff.
_MISS_FRONTIER_MIN_CAPACITY = 32


@dataclass(slots=True)
class _MaskPlan:
    """Constrained-decode mask for one decode sampling step.

    The constraint comes from skill_state (no logits needed), so it is built
    right after commit and uploaded async on the slot's copy stream while the
    forward runs. ``disallow`` is the staged per-row boolean mask over the
    vocabulary (True => force that token's logit to -inf); sampling waits on
    ``event`` then applies it with a single ``masked_fill_``. Both are None when
    no row constrains this step (the mask build and H2D are skipped entirely).

    ``suppress_rows`` (request-level ``suppress_next_token_ids``, which is
    logprobs-baseline sensitive and only fires on a request's first generated
    step) is carried through and applied inline in the sampling tail.
    """

    disallow: Tensor | None  # staged GPU bool mask [batch, vocab], or None
    event: torch.cuda.Event | NoopEvent | None  # None when unconstrained
    suppress_rows: list
    all_greedy: bool
    any_return_logprobs: bool


@dataclass(slots=True)
class PrefillStaging:
    """Per-prefill staging buffers and RenderBuffer for D2H."""

    sampled_ids: Tensor
    sampled_logprobs: Tensor
    sampling_temps_cpu: Tensor
    sampling_temps_gpu: Tensor
    sampling_top_ps_cpu: Tensor
    sampling_top_ps_gpu: Tensor
    render: RenderBuffer


@dataclass(slots=True)
class _PrefillCandidate:
    """Fresh launch-time view of how a request would prefill right now."""

    request: GenerationRequest
    classification: PrefillClassification
    reserve_length: int
    pages_needed: int
    cohort_key: tuple[Optional[str], bytes] | None

    @property
    def can_reuse(self) -> bool:
        return self.classification.can_reuse

    @property
    def use_prefix_attn(self) -> bool:
        return self.classification.use_prefix_attn

    @property
    def query_len(self) -> int:
        return self.classification.query_length


@dataclass(slots=True)
class _BoundPrefill:
    """Launch-ready prefill produced just-in-time for a selected request."""

    candidate: _PrefillCandidate
    prepared: PreparedSequence
    acquired_lora: bool


def _plan_prefill_launch_batch(
    candidates: Sequence[_PrefillCandidate],
    *,
    capacity_remaining: int,
    slot_budget: int,
    page_budget: int,
    token_floor: int,
) -> list[_PrefillCandidate]:
    """Select a launch batch from fresh launch-time candidates.

    Policy:
    - seed prefix-attn misses first when launch capacity is large enough
    - otherwise harvest non-prefix work (`can_reuse=True` first)
    - otherwise seed one uncached request per image cohort
    - keep all requests in a batch on the same prefill mode
    """

    if not candidates or capacity_remaining <= 0 or slot_budget <= 0 or page_budget <= 0:
        return []

    non_prefix_candidates = [candidate for candidate in candidates if not candidate.use_prefix_attn]
    prefix_candidates = [candidate for candidate in candidates if candidate.use_prefix_attn]

    miss_cohort_sizes = Counter(
        candidate.cohort_key
        for candidate in prefix_candidates
        if candidate.cohort_key is not None
    )

    prefer_prefix_frontier = bool(prefix_candidates) and (
        not non_prefix_candidates or capacity_remaining >= _MISS_FRONTIER_MIN_CAPACITY
    )

    if prefer_prefix_frontier:
        compatible = sorted(
            prefix_candidates,
            key=lambda candidate: (
                -miss_cohort_sizes.get(candidate.cohort_key, 1),
                candidate.request.submitted_at,
                candidate.request.request_id,
            )
        )
    else:
        compatible = sorted(
            non_prefix_candidates,
            key=lambda candidate: (
                not candidate.can_reuse,
                candidate.query_len,
                candidate.request.submitted_at,
                candidate.request.request_id,
            ),
        )

    batch: list[_PrefillCandidate] = []
    chosen_cohorts: set[tuple[Optional[str], bytes]] = set()
    pages_used = 0
    launch_tokens = 0
    max_batch = min(capacity_remaining, slot_budget)

    for candidate in compatible:
        request = candidate.request
        if request.max_new_tokens <= 0 and batch:
            continue
        if request.adapter is not None and batch:
            continue
        if batch and batch[0].request.adapter is not None:
            break
        if len(batch) >= max_batch:
            break
        if pages_used + candidate.pages_needed > page_budget:
            continue
        if (
            candidate.use_prefix_attn
            and candidate.cohort_key is not None
            and candidate.cohort_key in chosen_cohorts
        ):
            continue

        batch.append(candidate)
        pages_used += candidate.pages_needed
        launch_tokens += candidate.query_len
        if candidate.use_prefix_attn and candidate.cohort_key is not None:
            chosen_cohorts.add(candidate.cohort_key)
        if request.max_new_tokens <= 0:
            break
        if launch_tokens >= token_floor:
            break

    return batch


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(
        self,
        runtime: AutoregressiveRuntime,
        *,
        compute_stream: object,
        skill_registry: SkillRegistry,
        adapter_provider: Optional[AdapterProvider] = None,
    ) -> None:
        self.runtime = runtime
        self._compute_stream = compute_stream
        self._adapter_provider = adapter_provider
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[RequestLifecycle] = RunningQueue()
        self._completed: Deque[SchedulerResult] = deque()
        self._next_request_id = 0
        # The model's skills, supplied by the engine. Required: the kernel
        # holds no model-specific default.
        self._skills = skill_registry
        # Per-step sampling contract. Three optional hooks the runtime
        # can implement:
        #   * post_sample — run any model-specific GPU work alongside
        #     sampling (Moondream: coord/size decode from hidden states),
        #     return an opaque handle.
        #   * materialize_tokens — turn CPU-side sampled ids + the
        #     handle into typed Tokens (default: plain TextTokens).
        #   * prepare_decode_inputs — populate any model-specific decode
        #     inputs on the slot (Moondream: coord/size values for the
        #     next forward).
        # The runtime owns all storage and D2H for its side-channel
        # values; the scheduler only manages sampled token ids +
        # logprobs.
        self._hooks: SamplingHooks = getattr(runtime, "sampling_hooks", None) or SamplingHooks()
        self._pending_token_ids = torch.zeros(
            (runtime.max_batch_slots,),
            dtype=torch.long,
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
        pin_prefill_staging = runtime.device.type == "cuda"
        self._prefill_staging_pool: Deque[PrefillStaging] = deque(
            [
                PrefillStaging(
                    sampled_ids=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    sampled_logprobs=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.float32,
                        device=runtime.device,
                    ),
                    sampling_temps_cpu=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=pin_prefill_staging,
                    ),
                    sampling_temps_gpu=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.float32,
                        device=runtime.device,
                    ),
                    sampling_top_ps_cpu=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=pin_prefill_staging,
                    ),
                    sampling_top_ps_gpu=torch.empty(
                        (runtime.max_batch_size,),
                        dtype=torch.float32,
                        device=runtime.device,
                    ),
                    render=RenderBuffer(
                        runtime.max_batch_size,
                        runtime.device,
                        copy_stream=runtime.copy_stream,
                    ),
                )
                for _ in range(len(runtime.prefill_slots))
            ]
        )
        self._sampling_rng = torch.Generator(device=runtime.device)
        self._sampling_rng.manual_seed(torch.seed())
        self._pipeline = PipelineState()
        self._last_deferred_request_id: int | None = None

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

    def _materialize_tokens(
        self,
        token_ids_cpu: Tensor,
        sequences: Sequence,
        batch_idx: Tensor,
        runtime_step: object,
    ) -> list[Token]:
        """Default TextToken-only materialisation, runtime hook if defined."""
        hook = self._hooks.materialize_tokens
        if hook is None:
            return [TextToken(token_id=int(t)) for t in token_ids_cpu.view(-1).tolist()]
        return hook(token_ids_cpu, sequences, batch_idx, runtime_step)

    # ------------------------------------------------------------------
    # Execution

    def has_pending_work(self) -> bool:
        """Return ``True`` if there is anything left to prefill, decode, or complete."""
        return (
            len(self.waiting) > 0
            or len(self.running) > 0
            or not self._pipeline.is_empty()
        )

    def advance(self) -> bool:
        """Attempt to make progress using the pipelined decode loop.

        Returns ``True`` if any state changed (e.g. tokens decoded, new
        sequences admitted). Callers can keep invoking ``advance`` while it
        returns ``True`` to drain ready work before sleeping.

        Decode-tick ordering: commit -> finalize -> launch.
        1. Commit the oldest finalized step (updates skill state via
           consume_step).
        2. Finalize sampling for the in-flight forward (computes the mask with
           the now-updated skill state).
        3. Launch the next forward (last), so it enqueues behind this step's
           sampling and runs on the GPU while the *next* tick does its
           commit+finalize. This overlaps the commit and launch CPU work with
           the forward instead of paying launch on the critical path between
           forwards.

        Two invariants the ordering preserves:
        * commit-before-finalize: commit advances skill state (consume_step),
          so the forward we finalize samples under the correct
          constrained-decoding mask. Commit stays before finalize.
        * depth-1 speculation (zombies): the next forward is launched only
          after committing the step two behind it, so a sequence that ends
          rides exactly one in-flight step — no extra zombie waste.
        """
        progressed = False
        pipeline = self._pipeline
        with stream_context(self._compute_stream):
            has_launch = pipeline.has_launch_in_flight()
            has_queued = pipeline.queue_depth() > 0

            # Only run pipeline work if there is pending GPU or schedulable work.
            if (
                has_launch
                or has_queued
                or len(self.running) > 0
                or len(self.waiting) > 0
            ):
                # 1. Commit the oldest finalized step (updates skill state for
                # mask computation). The in-flight forward (launched last on the
                # previous tick) runs on the GPU in parallel while CPU blocks on D2H.
                oldest = pipeline.pop_oldest()
                if oldest is not None:
                    self.commit_step(oldest)
                    pipeline.on_step_completed()
                    progressed = True

                # 2. Finalize sampling (now skill state is updated for mask).
                # Build the constrained-decode mask for the in-flight decode
                # forward now -- after commit (so skill_state is current), while
                # the forward runs -- and stage it async on the copy stream, so
                # finalize only waits on the slot's mask_ready event and applies
                # a single masked_fill_, keeping the per-step mask H2D off the
                # compute stream's critical path.
                handle = pipeline.launch_handle
                if handle is not None:
                    plan = None
                    if handle.kind == "decode":
                        slot = self.runtime.decode_slots[handle.slot_id]
                        plan = self._build_mask(handle.sequences, slot)
                    step = self.finalize_sampling(handle, plan)
                    pipeline.on_pending_commit(step)
                    progressed = True

                # 3. Launch the next forward last (forward doesn't need mask), so
                # it overlaps the next tick's commit+finalize instead of paying
                # the launch on the critical path between forwards.
                if pipeline.can_launch():
                    launched_forward = False
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

        if not progressed:
            stalled = next((request for request in self.waiting if self._is_launchable_request(request)), None)
            if stalled is not None and not self.runtime.can_reserve(stalled.target_length):
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

    def _is_launchable_request(self, request: GenerationRequest) -> bool:
        lifecycle = request.lifecycle
        if lifecycle.phase not in (
            RequestPhase.WAITING_RESOURCES,
            RequestPhase.READY_FOR_PREFILL,
        ):
            return False
        if lifecycle.has_image and not lifecycle.crops_ready:
            return False
        return True

    def _make_prefill_candidate(
        self,
        request: GenerationRequest,
    ) -> _PrefillCandidate | None:
        if not self._is_launchable_request(request):
            return None

        lifecycle = request.lifecycle
        classification = self.runtime.classify_prefill(
            request.prefill_tokens,
            has_image=lifecycle.has_image,
            image_hash=request.image_hash,
            adapter_id=request.adapter,
        )
        reserve_length = max(request.target_length - classification.skip_positions, 1)
        if not self.runtime.can_reserve(reserve_length):
            return None

        page_size = self.runtime.page_table.page_size
        cohort_key = None
        if lifecycle.has_image and request.image_hash is not None:
            cohort_key = (request.adapter, request.image_hash)

        return _PrefillCandidate(
            request=request,
            classification=classification,
            reserve_length=reserve_length,
            pages_needed=(reserve_length + page_size - 1) // page_size,
            cohort_key=cohort_key,
        )

    def _select_prefill_batch(self, capacity_remaining: int) -> list[_PrefillCandidate]:
        page_budget, slot_budget = self.runtime.prefill_budget()
        if capacity_remaining <= 0 or slot_budget <= 0 or page_budget <= 0:
            return []

        candidates = []
        for request in self.waiting:
            candidate = self._make_prefill_candidate(request)
            if candidate is not None:
                candidates.append(candidate)

        return _plan_prefill_launch_batch(
            candidates,
            capacity_remaining=capacity_remaining,
            slot_budget=slot_budget,
            page_budget=page_budget,
            token_floor=_MIN_PREFILL_LAUNCH_TOKENS,
        )

    def _acquire_prefill_staging(self) -> PrefillStaging:
        """Acquire a prefill staging bundle."""
        if not self._prefill_staging_pool:
            raise RuntimeError("Prefill staging pool exhausted")
        return self._prefill_staging_pool.popleft()

    def _release_prefill_staging(self, staging: PrefillStaging) -> None:
        """Return a prefill staging bundle to the pool."""
        self._prefill_staging_pool.append(staging)

    def _stage_prefill_sampling_params(
        self,
        staging: PrefillStaging,
        requests: Sequence[GenerationRequest],
        batch_idx: Tensor,
    ) -> None:
        """Stage prefill sampling params without pageable scalar H2D copies."""

        batch = len(requests)
        temps_cpu = staging.sampling_temps_cpu[:batch]
        top_ps_cpu = staging.sampling_top_ps_cpu[:batch]
        for row, request in enumerate(requests):
            temps_cpu[row] = float(request.temperature)
            top_ps_cpu[row] = float(request.top_p)

        temps_gpu = staging.sampling_temps_gpu[:batch]
        top_ps_gpu = staging.sampling_top_ps_gpu[:batch]
        temps_gpu.copy_(temps_cpu, non_blocking=True)
        top_ps_gpu.copy_(top_ps_cpu, non_blocking=True)

        batch_idx = batch_idx.view(-1)[:batch]
        self._sampling_temps_by_batch.index_copy_(0, batch_idx, temps_gpu)
        self._sampling_top_ps_by_batch.index_copy_(0, batch_idx, top_ps_gpu)

    def _finalize_prefill(self, handle: LaunchHandle) -> PendingCommit:
        """Sample first token + start D2H for a prefill.

        IMPORTANT: Caller must already be on the compute stream.
        """
        if handle.kind != "prefill":
            raise AssertionError("prefill finalize requires a prefill handle")
        prefill_payload = handle.payload
        if not isinstance(prefill_payload, PrefillLaunch):
            raise AssertionError("prefill finalize requires a PrefillLaunch payload")
        staging = prefill_payload.staging
        logits = prefill_payload.logits
        prepared_sequences = prefill_payload.prepared_sequences
        prefill_slot = prefill_payload.prefill_slot
        sequences = handle.sequences
        batch_size = len(sequences)
        if batch_size == 0:
            raise AssertionError("Prefill finalize requires at least one sequence")

        sampled_ids, temps, top_ps, sampled_logprobs = self._sample_batch(
            logits,
            sequences,
            staging.sampled_ids,
            batch_idx=prefill_slot.batch_idx[:batch_size],
            logprobs_out=staging.sampled_logprobs[:batch_size],
        )
        hidden_rows: list[Tensor] = []
        for seq in sequences:
            hidden_last = seq.state.last_hidden
            if hidden_last is None:  # pragma: no cover - defensive
                raise RuntimeError("Missing last_hidden after prefill")
            hidden_rows.append(hidden_last)
        hidden_last = torch.stack(hidden_rows, dim=0)
        prefill_slot.step_done_event.record()
        batch_idx = prefill_slot.batch_idx[:batch_size].view(-1)
        # Prefill's first token is text for plain query, but skills like
        # point/detect can constrain it to coord/size — so post_sample
        # has to run for prefill too. The runtime owns the staging.
        runtime_step = None
        if self._hooks.post_sample is not None:
            runtime_step = self._hooks.post_sample(
                prefill_slot,
                sampled_ids=sampled_ids.view(-1),
                hidden_last=hidden_last,
                sequences=sequences,
                batch_idx=batch_idx,
                temperatures=temps,
                top_ps=top_ps,
                token_logprobs=sampled_logprobs,
                ready_event=prefill_slot.step_done_event,
            )
        self._pending_token_ids.index_copy_(0, batch_idx, sampled_ids.view(-1))
        prefill_slot.commit_done_event.record()
        for seq in sequences:
            seq.packed_pending_ready = True
            seq.uncommitted_prefill_token = True
            # Mark the sequence runnable immediately after token0 is sampled.
            self.running.push(seq)

        transfer = staging.render.transfer(
            sampled_ids,
            ready_event=prefill_slot.step_done_event,
            logprobs=sampled_logprobs,
        )
        return PendingCommit(
            kind="prefill",
            sequences=sequences,
            transfer=transfer,
            payload=PrefillPendingCommit(
                staging=staging,
                slot_id=prefill_payload.slot_id,
                prepared_sequences=prepared_sequences,
                prefill_slot=prefill_slot,
                runtime_step=runtime_step,
            ),
        )

    def _commit_prefill(
        self, step: PendingCommit
    ) -> tuple[list[Token], Tensor | None]:
        """Commit a prefill PendingCommit and return first tokens/logprobs."""
        if step.kind != "prefill":
            raise AssertionError("prefill commit requires a prefill pending commit")
        payload = step.payload
        if not isinstance(payload, PrefillPendingCommit):
            raise AssertionError("prefill commit requires a PrefillPendingCommit payload")
        staging = payload.staging
        prepared_sequences = payload.prepared_sequences
        prefill_slot = payload.prefill_slot
        try:
            token_ids_cpu, logprobs_cpu = step.transfer.wait()
            prefill_slot.commit_done_event.synchronize()
            for prepared in prepared_sequences:
                self.runtime.finalize_prepared_sequence_after_prefill(prepared)
            tokens = self._materialize_tokens(
                token_ids_cpu,
                step.sequences,
                payload.prefill_slot.batch_idx[: len(step.sequences)].view(-1),
                payload.runtime_step,
            )
        finally:
            self._release_prefill_staging(staging)
            self.runtime.release_prefill_slot(prefill_slot)
        return tokens, logprobs_cpu

    def _launch_prefill_step(self, pipeline: PipelineState) -> bool:
        """Launch a prefill forward and enqueue it into the shared pipeline.

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

        max_active = self.runtime.max_batch_slots - 1
        if len(self.running) > max_active:
            raise AssertionError(
                f"running queue exceeded active slot cap ({len(self.running)} > {max_active})"
            )

        # Capacity is bounded by both active-slot headroom and per-forward microbatch.
        capacity_remaining = min(
            self.runtime.max_batch_size,
            max_active - len(self.running),
        )
        if capacity_remaining <= 0:
            return False

        launch_candidates = self._select_prefill_batch(capacity_remaining)
        if not launch_candidates:
            return False

        # Acquire prefill slot immediately before launch-time binding.
        try:
            prefill_slot = self.runtime.acquire_prefill_slot(slot_id)
        except Exception:
            return False

        bound_batch: list[_BoundPrefill] = []
        progress = False
        for candidate in launch_candidates:
            request = candidate.request
            lifecycle = request.lifecycle
            acquired_lora = False
            if not lifecycle.lora_slot_ready:
                try:
                    request.lora_slot = self._acquire_adapter_slot(request.adapter)
                    lifecycle.lora_slot_ready = True
                    acquired_lora = True
                except Exception as exc:
                    self.waiting.remove(request)
                    self._fail_request_early(request, exc)
                    progress = True
                    continue

            try:
                prefill_start = time.perf_counter()
                lifecycle.prefill_started_at = prefill_start
                lifecycle.transition(RequestPhase.PREFILLING)
                prepared = self.runtime.prepare_sequence(
                    prompt_tokens=request.prefill_tokens,
                    image=request.image,
                    image_crops=request.image_crops,
                    max_new_tokens=request.remaining_new_tokens,
                    lora_slot=request.lora_slot,
                    image_hash=request.image_hash,
                    adapter_id=request.adapter,
                )
            except Exception as exc:
                lifecycle.prefill_started_at = None
                lifecycle.prefill_completed_at = None
                if acquired_lora:
                    self.runtime.release_adapter_slot(request.lora_slot)
                    request.lora_slot = 0
                    lifecycle.lora_slot_ready = False
                lifecycle.transition(
                    RequestPhase.READY_FOR_PREFILL
                    if (lifecycle.crops_ready and lifecycle.lora_slot_ready)
                    else RequestPhase.WAITING_RESOURCES
                )
                if isinstance(exc, RuntimeError) and "Cannot reserve" in str(exc):
                    continue
                self.waiting.remove(request)
                self._fail_request_early(request, exc)
                progress = True
                continue

            lifecycle.sequence_state = prepared.state

            self.waiting.remove(request)
            bound_batch.append(
                _BoundPrefill(
                    candidate=candidate,
                    prepared=prepared,
                    acquired_lora=acquired_lora,
                )
            )

        if not bound_batch:
            self.runtime.release_prefill_slot(prefill_slot)
            return progress

        requests = [item.candidate.request for item in bound_batch]
        lifecycles = [request.lifecycle for request in requests]
        sequences = lifecycles
        prepared_sequences = [item.prepared for item in bound_batch]

        def fail_bound_batch(
            exc: Exception,
            *,
            staging: PrefillStaging | None = None,
        ) -> bool:
            if staging is not None:
                self._release_prefill_staging(staging)
            try:
                for item in bound_batch:
                    self.runtime.abort_prepared_sequence(item.prepared)
            finally:
                try:
                    self.runtime.release_prefill_slot(prefill_slot)
                except Exception:
                    pass
                for item in bound_batch:
                    request = item.candidate.request
                    lifecycle = request.lifecycle
                    lifecycle.prefill_started_at = None
                    lifecycle.prefill_completed_at = None
                    if item.acquired_lora:
                        try:
                            self.runtime.release_adapter_slot(request.lora_slot)
                        except Exception:
                            pass
                        request.lora_slot = 0
                        lifecycle.lora_slot_ready = False
            for request in requests:
                self._fail_request_early(request, exc)
            return True

        # For 0-length generation, we don't need token0 sampling/D2H staging.
        first_request = requests[0]
        if first_request.max_new_tokens <= 0:
            request = first_request
            lifecycle = request.lifecycle
            seq = lifecycle
            prepared_seq = prepared_sequences[0]
            try:
                with torch.inference_mode():
                    self.runtime.launch_prepared_batch(
                        [prepared_seq],
                        prefill_slot,
                        images=[request.image],
                        image_crops_list=[request.image_crops],
                    )
                lifecycle.prefill_completed_at = time.perf_counter()
                # Ensure prefill forward completes before cache finalize + release.
                with stream_context(self._compute_stream):
                    prefill_slot.commit_done_event.record()
                prefill_slot.commit_done_event.synchronize()
                self.runtime.finalize_prepared_sequence_after_prefill(prepared_seq)
            except Exception as exc:
                return fail_bound_batch(exc)

            seq.first_token_time = lifecycle.prefill_started_at or time.perf_counter()
            self._finalize_sequence(seq, "length")
            self.runtime.release_prefill_slot(prefill_slot)
            return True

        try:
            staging = self._acquire_prefill_staging()
        except Exception:
            for item in bound_batch:
                self.runtime.abort_prepared_sequence(item.prepared)
                if item.acquired_lora:
                    request = item.candidate.request
                    self.runtime.release_adapter_slot(request.lora_slot)
                    request.lora_slot = 0
                    request.lifecycle.lora_slot_ready = False
            self.runtime.release_prefill_slot(prefill_slot)
            raise

        try:
            with torch.inference_mode():
                logits = self.runtime.launch_prepared_batch(
                    prepared_sequences,
                    prefill_slot,
                    images=[request.image for request in requests],
                    image_crops_list=[request.image_crops for request in requests],
                )
                with stream_context(self._compute_stream):
                    self._stage_prefill_sampling_params(
                        staging,
                        requests,
                        prefill_slot.batch_idx[: len(requests)],
                    )
            prefill_completed = time.perf_counter()
            for lifecycle in lifecycles:
                lifecycle.prefill_completed_at = prefill_completed
        except Exception as exc:
            return fail_bound_batch(exc, staging=staging)

        # Prefill has completed (KV/logits enqueued); notify skill.
        for seq in sequences:
            seq.skill_state.on_prefill(self.runtime)

        handle = LaunchHandle(
            kind="prefill",
            sequences=sequences,
            payload=PrefillLaunch(
                staging=staging,
                slot_id=slot_id,
                logits=logits,
                prepared_sequences=prepared_sequences,
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

    def _cap_decode_dispatch(
        self,
        dispatchable: List[RequestLifecycle],
        limit: int,
    ) -> List[RequestLifecycle]:
        """Cap decode dispatch size while keeping deterministic, fair ordering."""
        if len(dispatchable) <= limit:
            self._last_deferred_request_id = None
            return dispatchable

        last_deferred = self._last_deferred_request_id
        defer_idx: int | None = None
        for idx in range(len(dispatchable) - 1, -1, -1):
            if dispatchable[idx].request.request_id != last_deferred:
                defer_idx = idx
                break
        if defer_idx is None:
            defer_idx = len(dispatchable) - 1

        deferred = dispatchable.pop(defer_idx)
        self._last_deferred_request_id = deferred.request.request_id
        return dispatchable[:limit]

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

        decode_limit = self.runtime.max_batch_size
        if len(active) > decode_limit:
            active = self._cap_decode_dispatch(active, decode_limit)
        else:
            self._last_deferred_request_id = None

        return StepPlan(sequences=active)

    def launch_forward_async(
        self, plan: StepPlan, slot_id: int
    ) -> LaunchHandle:
        """Launch the forward pass for a decode step (with stream context).

        Wrapper that enters the compute stream context before calling
        _launch_forward_on_stream. Use this when calling from outside advance().
        """
        slot = self.runtime.decode_slots[slot_id]
        with stream_context(slot.compute_stream):
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

            # Gather decode inputs from _pending_token_ids into slot staging.
            token_ids = slot.decode_token_ids[:batch_size]
            torch.index_select(self._pending_token_ids, 0, batch_idx, out=token_ids)
            # Runtime gathers any of its own per-step state into the slot
            # (Moondream: coord/size values for the next decode forward).
            if self._hooks.prepare_decode_inputs is not None:
                self._hooks.prepare_decode_inputs(slot, batch_idx, batch_size)

            # Commit page table for all batch indices before forward pass (deferred H2D sync)
            batch_indices_list = [seq.state.batch_idx for seq in sequences]
            self.runtime.page_table.commit_block_table(batch_indices_list)

            # Run forward pass - writes to slot.logits and slot.hidden_last.
            # ``inference_mode`` is applied centrally here (not left to each
            # runtime) so eager model paths can't accidentally retain
            # autograd graphs in their KV caches across decode steps.
            with torch.inference_mode():
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
        self, handle: LaunchHandle, plan: "_MaskPlan | None" = None
    ) -> PendingCommit:
        """Finalize sampling for a forward pass (with stream context).

        Wrapper that enters the compute stream context before calling
        _finalize_sampling_on_stream. Use this when calling from outside advance().
        """
        if handle.kind == "decode":
            slot = self.runtime.decode_slots[handle.slot_id]
            with stream_context(slot.compute_stream):
                return self._finalize_sampling_on_stream(handle, plan)
        if handle.kind == "prefill":
            if plan is not None:
                raise AssertionError("Prefill finalize does not support masks")
            with stream_context(self._compute_stream):
                return self._finalize_prefill(handle)
        raise AssertionError(f"Unsupported handle kind {handle.kind!r}")

    def _finalize_sampling_on_stream(
        self, handle: LaunchHandle, plan: "_MaskPlan | None" = None
    ) -> PendingCommit:
        """Finalize sampling for a forward pass and start D2H transfer.

        IMPORTANT: Caller must already be on the compute stream.

        Takes the forward outputs from the DecodeSlot's buffers, applies the
        constrained-decode mask, samples tokens, runs the runtime's post-sample
        hook (Moondream uses it for coord/size decode; default is a no-op),
        writes to pending buffers, and kicks off async D2H via the slot's
        RenderBuffer.

        ``plan`` is the ``_MaskPlan`` built by ``_build_mask`` after commit and
        staged async on the slot's copy stream; sampling waits on its event and
        applies it. It is None only for prefill, which builds its mask inline.

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
        sampled_ids, temps, top_ps, sampled_logprobs = self._sample_batch(
            logits,
            sequences,
            slot.sampled_ids,
            batch_idx=batch_idx,
            logprobs_out=slot.sampled_logprobs[:batch_size],
            plan=plan,
        )

        # Record event once sampled_ids staging is ready. The copy stream
        # waits on this for the D2H below, and the runtime's post_sample
        # hook can re-use the same event to anchor its own copy waits.
        slot.step_done_event.record()

        runtime_step = None
        if self._hooks.post_sample is not None:
            runtime_step = self._hooks.post_sample(
                slot,
                sampled_ids=sampled_ids,
                hidden_last=hidden_last,
                sequences=sequences,
                batch_idx=batch_idx,
                temperatures=temps,
                top_ps=top_ps,
                token_logprobs=sampled_logprobs,
                ready_event=slot.step_done_event,
            )

        # Write to shared pending buffer for next step's input gathering.
        self._pending_token_ids.index_copy_(0, batch_idx, sampled_ids)

        # Record a second event after pending writes so we can safely release/reuse
        # batch indices (e.g. finalize a sequence and admit a new one into the same
        # batch slot) without racing the `_pending_*` updates.
        slot.commit_done_event.record()
        for seq in sequences:
            seq.packed_pending_ready = True

        # Start async D2H transfer of sampled ids + logprobs from per-slot
        # staging. Pass step_done_event so the copy stream waits only on the
        # staging writes, not the pending-buffer index_copy_.
        transfer = slot.render.transfer(
            slot.sampled_ids[:batch_size],
            ready_event=slot.step_done_event,
            logprobs=sampled_logprobs,
        )

        return PendingCommit(
            kind="decode",
            sequences=sequences,
            transfer=transfer,
            payload=DecodePendingCommit(
                slot_id=handle.slot_id,
                runtime_step=runtime_step,
            ),
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
            tokens, logprobs_cpu = self._commit_prefill(step)
            if len(tokens) != len(step.sequences):
                raise RuntimeError(
                    "Prefill token count mismatch: "
                    f"{len(tokens)} token(s) for {len(step.sequences)} sequence(s)"
                )
            logprobs = self._logprobs_for_sequences(step.sequences, logprobs_cpu)
            for seq, token, logprob in zip(step.sequences, tokens, logprobs):
                seq.stage_token(self.runtime, token, logprob=logprob)
                seq.uncommitted_prefill_token = False
                if self._mark_finished_if_needed(seq):
                    seq.finalized = True
                    # Prefill sequences are enqueued into `running` immediately
                    # after token0 is sampled, so remove on termination here.
                    self.running.remove(seq)
                    if seq.inflight_refs == 0:
                        seq.transition(RequestPhase.COMPLETED)
                        self._release_sequence(seq)
            return

        if step.kind != "decode":
            raise AssertionError(f"Unsupported commit kind {step.kind!r}")

        # Wait for D2H transfer
        token_ids_cpu, logprobs_cpu = step.transfer.wait()

        # Ensure `_pending_token_ids` writes for this step have completed before
        # we release or reuse any batch indices (these writes intentionally do
        # not gate the D2H above).
        slot = self.runtime.decode_slots[step.slot_id]
        slot.commit_done_event.synchronize()

        # Materialise typed tokens. Runtime threads its own per-step state via
        # ``runtime_step`` (e.g. CPU-side aux values, if it owns them).
        payload: DecodePendingCommit = step.payload
        batch_idx_cpu = slot.meta.batch_idx.cpu[: len(step.sequences)]
        tokens = self._materialize_tokens(
            token_ids_cpu,
            step.sequences,
            batch_idx_cpu,
            payload.runtime_step,
        )
        logprobs = self._logprobs_for_sequences(step.sequences, logprobs_cpu)

        # Commit each sequence
        for seq, token, logprob in zip(step.sequences, tokens, logprobs):
            seq.inflight_refs -= 1

            # Skip zombies (already finalized in a previous step)
            if seq.finalized:
                if seq.inflight_refs == 0:
                    seq.transition(RequestPhase.COMPLETED)
                    self._release_sequence(seq)
                continue

            # Stage token (calls consume_step, emits streaming)
            seq.stage_token(self.runtime, token, logprob=logprob)

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
            try:
                # The generated prefix was part of prefill, so only tokens
                # decoded after prefill should be retained as generated suffix.
                generated_tokens = seq.skill_state.tokens[
                    seq.request.generated_prefix_length:
                ]
                self.runtime.retain_sequence_prefix(
                    seq.state,
                    generated_tokens,
                    adapter_id=seq.request.adapter,
                    image_hash=seq.request.image_hash,
                )
            finally:
                # Retention is part of the cache path, but release is resource
                # ownership. If retention hits an invariant bug, still release
                # the batch slot/pages/adapter, then let the original exception
                # propagate through the scheduler-fatal path.
                self.runtime.release_sequence(seq.state)

    def _build_mask_spec(self, sequences: List[RequestLifecycle]) -> tuple:
        """Per-sequence sampling-mask inputs (skill_state/request only, no logits).

        Hoisting this out of _sample_batch lets it run concurrently with the
        in-flight forward instead of on the post-forward critical path.
        """
        allowed_tokens: list[Optional[Sequence[int]]] = []
        suppressed_tokens: list[Optional[Sequence[int]]] = []
        restrict = False
        for seq in sequences:
            # Treat finalized sequences (zombies) as unconstrained.
            if seq.finalized:
                allowed_tokens.append(None)
                suppressed_tokens.append(None)
                continue
            allowed = seq.skill_state.allowed_token_ids(self.runtime)
            allowed_tokens.append(allowed)
            if allowed:
                restrict = True
            suppressed_tokens.append(seq.skill_state.suppressed_token_ids(self.runtime))
        suppress_rows: list[tuple[int, tuple[int, ...]]] = []
        for i, seq in enumerate(sequences):
            suppress = seq.request.suppress_next_token_ids
            if (
                seq.finalized
                or not suppress
                or seq.skill_state.token_count != seq.request.generated_prefix_length
            ):
                continue
            suppress_rows.append((i, suppress))
        all_greedy = all(seq.request.temperature <= 0.0 for seq in sequences)
        any_return_logprobs = any(seq.request.return_logprobs is True for seq in sequences)
        return (
            allowed_tokens,
            suppressed_tokens,
            restrict,
            suppress_rows,
            all_greedy,
            any_return_logprobs,
        )

    def _build_mask(
        self, sequences: List[RequestLifecycle], slot
    ) -> "_MaskPlan":
        """Build the decode sampling mask and stage it async on the slot.

        Runs after commit (so skill_state is current) while the forward is in
        flight. Writes a per-row boolean disallow mask into the slot's pinned
        buffer and kicks off the H2D on the copy stream, so the per-step mask
        transfer overlaps the forward instead of blocking the compute stream.
        Sampling waits on ``slot.mask_ready_event`` before applying it.
        """
        (
            allowed_tokens,
            suppressed_tokens,
            restrict,
            suppress_rows,
            all_greedy,
            any_return_logprobs,
        ) = self._build_mask_spec(sequences)
        batch = len(sequences)

        # Nothing constrains this step: skip the mask build + H2D entirely.
        if not restrict and not any(suppressed_tokens):
            return _MaskPlan(
                disallow=None,
                event=None,
                suppress_rows=suppress_rows,
                all_greedy=all_greedy,
                any_return_logprobs=any_return_logprobs,
            )

        # Fill the per-row boolean disallow mask (True => -inf) in pinned host
        # memory. Restrict rows start fully disallowed then re-allow the
        # permitted ids; suppression flips its blacklisted ids on. Unconstrained
        # rows stay all-False (a no-op under masked_fill_). These are NumPy
        # writes to the pinned view -- no device sync.
        disallow_np = slot.disallow_mask.np
        disallow_np[:batch] = False
        for i in range(batch):
            allowed = allowed_tokens[i]
            if allowed:
                disallow_np[i, :] = True
                disallow_np[i, list(allowed)] = False
            suppressed = suppressed_tokens[i]
            if suppressed:
                disallow_np[i, list(suppressed)] = True

        # Async H2D on the copy stream (overlaps the forward); record the event
        # finalize waits on before masked_fill_. The slot's ping-pong lifecycle
        # (reused only after its commit, which fences past mask_ready) keeps the
        # pinned buffer safe to overwrite next time this slot comes around.
        with stream_context(self.runtime.copy_stream):
            disallow_gpu = slot.disallow_mask.copy_to_gpu(batch)
        slot.mask_ready_event.record(self.runtime.copy_stream)

        return _MaskPlan(
            disallow=disallow_gpu,
            event=slot.mask_ready_event,
            suppress_rows=suppress_rows,
            all_greedy=all_greedy,
            any_return_logprobs=any_return_logprobs,
        )

    def _sample_batch(
        self,
        logits: Tensor,
        sequences: List[RequestLifecycle],
        out: Tensor,
        *,
        batch_idx: Tensor | None = None,
        logprobs_out: Tensor | None = None,
        plan: "_MaskPlan | None" = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Sample tokens from logits into the provided output buffer.

        Args:
            logits: Logits tensor of shape [batch, vocab_size].
            sequences: List of scheduled sequences (for temperature/top_p and
                finalized state). Finalized sequences are treated as unconstrained
                to avoid querying skill state after termination.
            out: Pre-allocated output buffer for sampled token IDs. Must be
                shape [batch] or larger, dtype long.
            plan: Pre-built ``_MaskPlan`` whose disallow mask was staged async on
                the slot (the decode hot path). When None (prefill, once per
                request), the mask is built and applied inline here.

        Returns:
            Tuple of (sampled_ids, temps, top_ps, logprobs). sampled_ids is a
            view of out[:batch]. logprobs is a view of logprobs_out[:batch]
            when at least one sequence requested logprobs, otherwise None.
        """
        batch = len(sequences)
        if batch == 0:
            return out[:0], None, None, None

        if plan is not None:
            # Decode hot path: the disallow mask was filled and uploaded on the
            # copy stream during the forward. Wait for that H2D, then apply it as
            # a single vectorized masked_fill_ -- no per-row loop, no sync.
            if plan.event is not None:
                plan.event.wait()
            if plan.disallow is not None:
                logits.masked_fill_(plan.disallow, float("-inf"))
            suppress_rows = plan.suppress_rows
            all_greedy = plan.all_greedy
            any_return_logprobs = plan.any_return_logprobs
        else:
            # Prefill (once per request, off the per-token critical path): build
            # the constraint and apply it inline. The small synchronous H2D here
            # is acceptable because it does not recur per generated token.
            (
                allowed_tokens,
                suppressed_tokens,
                restrict,
                suppress_rows,
                all_greedy,
                any_return_logprobs,
            ) = self._build_mask_spec(sequences)

            if restrict:
                for i, allowed in enumerate(allowed_tokens):
                    if not allowed:
                        continue
                    idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
                    row = logits[i]
                    pruned = torch.full_like(row, float("-inf"))
                    pruned[idx] = row[idx]
                    logits[i] = pruned

            # Apply per-skill token suppression (blacklist).
            for i, suppressed in enumerate(suppressed_tokens):
                if suppressed:
                    idx = torch.tensor(suppressed, device=logits.device, dtype=torch.long)
                    logits[i, idx] = float("-inf")

        want_logprobs = logprobs_out is not None and any_return_logprobs
        logprobs = logprobs_out[:batch] if want_logprobs else None
        out_view = out[:batch]

        baseline_row_idx: Tensor | None = None
        baseline_logits: Tensor | None = None
        if logprobs is not None and suppress_rows:
            rows = [
                i
                for i, _ in suppress_rows
                if sequences[i].request.return_logprobs is True
            ]
            if rows:
                baseline_row_idx = torch.tensor(
                    rows, device=logits.device, dtype=torch.long
                )
                baseline_logits = logits.index_select(0, baseline_row_idx)

        for i, suppress in suppress_rows:
            idx = torch.tensor(suppress, device=logits.device, dtype=torch.long)
            logits[i, idx] = float("-inf")

        if all_greedy:
            if logprobs is None:
                torch.argmax(logits, dim=-1, out=out_view)
                return out_view, None, None, None
            temps = self._sampling_temps[:batch]
            top_ps = self._sampling_top_ps[:batch]
            temps.zero_()
            top_ps.fill_(1.0)
        else:
            if batch_idx is None:
                raise AssertionError("batch_idx is required for non-greedy sampling")
            batch_idx = batch_idx.view(-1)[:batch]
            temps = self._sampling_temps[:batch]
            top_ps = self._sampling_top_ps[:batch]
            torch.index_select(self._sampling_temps_by_batch, 0, batch_idx, out=temps)
            torch.index_select(self._sampling_top_ps_by_batch, 0, batch_idx, out=top_ps)

        sample_kwargs = {
            "out": out_view,
            "generator": self._sampling_rng,
        }
        if logprobs is not None:
            sample_kwargs["logprobs_out"] = logprobs
        sampled_raw = sample_step_from_logits(
            logits,
            temps,
            top_ps,
            **sample_kwargs,
        )
        if sampled_raw is not out_view:
            out_view.copy_(sampled_raw)
        if baseline_logits is not None and baseline_row_idx is not None:
            if temps is None:
                raise AssertionError("sampling temperatures are required for logprobs")
            base_logprobs = self._selected_logprobs_from_logits(
                baseline_logits,
                out_view.index_select(0, baseline_row_idx),
                temps.index_select(0, baseline_row_idx),
            )
            logprobs.index_copy_(0, baseline_row_idx, base_logprobs)
        return out_view, temps, top_ps, logprobs

    @staticmethod
    def _selected_logprobs_from_logits(
        logits: Tensor,
        sampled_ids: Tensor,
        temperatures: Tensor,
    ) -> Tensor:
        temps = torch.clamp(temperatures, min=0.0)
        effective_temp = torch.where(
            temps > _SAMPLING_EPS,
            temps,
            torch.ones_like(temps),
        ).unsqueeze(-1)
        scaled = logits.to(dtype=torch.float32) / effective_temp
        normalizer = torch.logsumexp(scaled, dim=-1)
        force_greedy = (temps <= _SAMPLING_EPS) | ~torch.isfinite(normalizer)
        greedy_ids = logits.argmax(dim=-1)
        selected = scaled.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        softmax_logprobs = selected - normalizer
        greedy_logprobs = torch.where(
            sampled_ids == greedy_ids,
            torch.zeros_like(softmax_logprobs),
            torch.full_like(softmax_logprobs, float("-inf")),
        )
        return torch.where(force_greedy, greedy_logprobs, softmax_logprobs)

    def _logprobs_for_sequences(
        self,
        sequences: Sequence[RequestLifecycle],
        logprobs_cpu: Tensor | None,
    ) -> list[float | None]:
        if logprobs_cpu is None:
            return [None] * len(sequences)
        return [
            float(logprobs_cpu[i].item()) if seq.request.return_logprobs is True else None
            for i, seq in enumerate(sequences)
        ]

    def _mark_finished_if_needed(self, seq: RequestLifecycle) -> bool:
        last_token = seq.last_token
        eos_id = self.runtime.prompt_template.eos_id
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
            self._release_sequence(seq)
        else:
            seq.transition(RequestPhase.FINALIZING)

        self._completed.append(self._build_result(seq))

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
            finalization_failed = False
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("Failed to finalize sequence %s", seq.request.request_id)
            finish_reason = "error"
            tokens = []
            output = {"error": str(exc)}
            finalization_failed = True

        logprobs = None
        if seq.request.return_logprobs is True and not finalization_failed:
            logprobs = list(seq.logprobs)
            if len(logprobs) != len(tokens):
                _LOGGER.error(
                    "Logprob/token count mismatch for request %s: "
                    "%s logprob(s) for %s token(s)",
                    seq.request.request_id,
                    len(logprobs),
                    len(tokens),
                )
                finish_reason = "error"
                tokens = []
                output = {
                    "error": "Internal logprobs/token alignment mismatch"
                }
                logprobs = None

        decode_tokens = max(
            0,
            seq.skill_state.token_count - seq.request.generated_prefix_length,
        )
        metrics = seq.build_metrics(decode_tokens=decode_tokens)
        return SchedulerResult(
            request_id=seq.request.request_id,
            tokens=tokens,
            finish_reason=finish_reason,
            metrics=metrics,
            output=output,
            logprobs=logprobs,
        )
