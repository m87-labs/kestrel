"""Flexible batching scheduler for Moondream text inference."""


from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import time
import logging

import torch
from torch import Tensor

from kestrel.device import NoopEvent, stream_context
from kestrel.utils import CpuGpuBuffer
from kestrel.runtime import (
    PrefillClassification,
    PreparedSequence,
    AutoregressiveRuntime,
    SequenceState,
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
# Marker the LoRA slot manager raises when every adapter slot is in use (see
# ``LoRASlotManager.acquire`` -> "Out of LoRA slots: all N slots are in use").
# Only this RuntimeError is a recoverable, retry-once-a-slot-frees deferral;
# every other RuntimeError from adapter acquisition (e.g. a CUDA load/copy
# failure) is unrecoverable and must fail the request, not loop forever.
_OUT_OF_LORA_SLOTS_MARKER = "Out of LoRA slots"


def _is_out_of_lora_slots(exc: BaseException) -> bool:
    """True only for the recoverable out-of-LoRA-slots signal."""
    return isinstance(exc, RuntimeError) and _OUT_OF_LORA_SLOTS_MARKER in str(exc)


def _min_cap(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Combine two spec ``commit_caps`` (per-row token bounds).

    Each cap is an upper bound on tokens a spec macro-step may commit for a row,
    with ``None`` meaning "no bound" (+inf). The combined cap is the tighter of
    the two: ``min`` when both are present, the present one when only one is, and
    ``None`` when neither bounds the row. Used to fold the stateful-mask cap and
    the remaining-output-budget cap into a single per-row ``commit_caps`` entry.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


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
    sampling_temps: CpuGpuBuffer
    sampling_top_ps: CpuGpuBuffer
    render: RenderBuffer

    def stage_sampling_params(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[Tensor, Tensor]:
        batch = len(requests)
        temps_cpu = self.sampling_temps.cpu[:batch]
        top_ps_cpu = self.sampling_top_ps.cpu[:batch]
        for row, request in enumerate(requests):
            temps_cpu[row] = float(request.temperature)
            top_ps_cpu[row] = float(request.top_p)
        return (
            self.sampling_temps.copy_to_gpu(batch),
            self.sampling_top_ps.copy_to_gpu(batch),
        )


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

        def sampling_param_buffer() -> CpuGpuBuffer:
            return CpuGpuBuffer(
                runtime.max_batch_size,
                dtype=torch.float32,
                device=runtime.device,
                pin_memory=pin_prefill_staging,
                with_numpy=False,
                zero=False,
            )

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
                    sampling_temps=sampling_param_buffer(),
                    sampling_top_ps=sampling_param_buffer(),
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
        # Depth-1 spec overlap: the previously-launched macro-step's
        # (sequences, lazy SpecStepResult) awaiting commit while the GPU runs the
        # next step. None when no spec step is in flight.
        self._pending_spec = None

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
            # Depth-1 spec overlap: a launched-but-uncommitted macro-step keeps
            # work pending even when ``running`` is already empty. The last
            # running sequence can finish in ``_commit_spec`` while the
            # follow-up (zombie) step it launched stays in ``_pending_spec``;
            # without this term the executor stops calling ``advance`` and that
            # step is never committed, so ``decoder.retire`` never runs and the
            # spec row leaks.
            or self._pending_spec is not None
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

        When the runtime advertises speculative decoding (``runtime.spec`` is
        not ``None``) the loop drives the spec macro-step path instead; the
        single-token pipeline below runs unchanged when ``runtime.spec`` is
        ``None`` (byte-for-byte identical behavior).
        """
        if self.runtime.spec is not None:
            return self._advance_spec()
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
        """Drain in-flight work before prefill / before an engine pause.

        Respects Phase 1 commit-before-finalize ordering: complete all queued
        steps before finalizing any in-flight forward. This ensures grammar
        state is updated before computing masks for constrained decoding.

        The engine PAUSE path calls this (``Executor.drain``) before mutating
        runtime state under ``graph_capture_lock`` (e.g. rebuilding CUDA graphs).
        The spec macro-step path commits through ``_pending_spec``, which the
        non-spec pipeline drain below does not touch, so drain spec work first:
        if a pause lands after a macro-step was launched (``_pending_spec`` set)
        and before the next ``advance``, the spec result would otherwise stay
        uncommitted while the runtime/graphs are mutated under an active spec row
        -- corrupting that row and delaying its completion/retirement until
        resume. ``_drain_spec`` commits the in-flight macro-step (no new launch)
        until ``_pending_spec`` is empty. In non-spec mode it is a no-op.
        """
        self._drain_spec()

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

    # ------------------------------------------------------------------
    # Speculative decode path
    #
    # When ``runtime.spec`` is set the runtime exposes a per-macro-step
    # decoder (draft + verify + greedy-accept + commit, see
    # ``kestrel.runtime.spec.SpecDecoder``). One macro-step advances each active
    # sequence by a *variable* amount (a_i + 1 tokens). The runtime owns all
    # device state (persistent pool + reused CUDA graphs + GDN ring buffers); the
    # scheduler only admits/retires sequences and stages the returned tokens.
    #
    # This path is deliberately synchronous (the spec decoder samples + commits
    # on the GPU itself and reads accept counts back to the host each step), so
    # it does not use the single-token depth-1 pipeline above. The non-spec path
    # is untouched.
    # ------------------------------------------------------------------

    def _advance_spec(self) -> bool:
        progressed = False
        with stream_context(self._compute_stream):
            progressed |= self._spec_admit()
            progressed |= self._spec_decode_step()
        return progressed

    def _spec_admit(self) -> bool:
        """Admit waiting requests into free spec rows (prefill + first token)."""
        decoder = self.runtime.spec.decoder
        if decoder is None:
            raise RuntimeError("runtime.spec set without a decoder")
        progressed = False
        # Requests deferred this call because their adapter slot is currently
        # unavailable. Tracking them stops the launchable scan from re-picking
        # the same request (it stays in ``waiting``) while still letting other
        # waiting requests be considered for admission.
        deferred_ids: set[int] = set()
        while decoder.free_slots > 0:
            request = next(
                (
                    r
                    for r in self.waiting
                    if self._is_launchable_request(r)
                    and r.request_id not in deferred_ids
                ),
                None,
            )
            if request is None:
                break
            lifecycle = request.lifecycle

            # Zero-length generation: the non-spec path has a dedicated branch
            # (see the ``max_new_tokens <= 0`` case in ``_launch_prefill_step``)
            # that finalizes the request as ``"length"`` WITHOUT sampling/staging
            # a first token. ``decoder.admit`` has no prefill-only mode -- its
            # contract is to prefill *and* sample/stage token0 -- so routing a
            # 0-token request through it would stream/return one extra token and
            # briefly consume a spec pool row. Match the non-spec contract:
            # finalize with zero generated tokens, no admit, no row. The spec
            # decoder's persistently-reserved pool gains nothing from a prefill it
            # would immediately retire, so skipping admission entirely is both
            # correct and leaves the row free for real work.
            if request.max_new_tokens <= 0:
                self.waiting.remove(request)
                # No spec row, so no ``sequence_state``; ``build_metrics`` then
                # falls back to ``request.prompt_length`` exactly like the
                # non-spec 0-length path (which also leaves ``sequence_state``
                # unset). ``_finalize_sequence`` -> ``_release_sequence``
                # early-returns for a spec runtime before touching the (absent)
                # state, so this is safe with ``sequence_state is None``.
                lifecycle.prefill_started_at = time.perf_counter()
                lifecycle.prefill_completed_at = lifecycle.prefill_started_at
                self._finalize_sequence(lifecycle, "length")
                progressed = True
                continue

            # Acquire the LoRA slot before admission, mirroring the normal
            # prefill path. Without this, adapter requests admitted to the spec
            # path keep ``request.lora_slot == 0`` and silently run on the
            # base-model slot, ignoring the requested finetune. A genuine load
            # failure fails just that request; transient slot exhaustion (all
            # LoRA slots in use) keeps the request WAITING_RESOURCES so it
            # retries once an adapter retires, instead of admitting it wrong.
            if not lifecycle.lora_slot_ready:
                try:
                    request.lora_slot = self._acquire_adapter_slot(request.adapter)
                    lifecycle.lora_slot_ready = True
                except RuntimeError as exc:
                    if not _is_out_of_lora_slots(exc):
                        # Not the recoverable out-of-slots case: a genuine
                        # adapter load/copy failure (e.g. a CUDA error while
                        # loading a new slot) also surfaces as a RuntimeError.
                        # Retrying it would loop forever, so fail this request
                        # cleanly instead of deferring it.
                        self.waiting.remove(request)
                        self._fail_request_early(request, exc)
                        progressed = True
                        continue
                    # Out of LoRA slots: recoverable. The request is genuinely
                    # blocked on a LoRA resource, so put it back in
                    # WAITING_RESOURCES (it stays in the waiting queue) and try
                    # other waiting requests; it retries once a slot frees up.
                    _LOGGER.debug(
                        "Deferring spec admission for request %s: %s",
                        request.request_id,
                        exc,
                    )
                    lifecycle.transition(RequestPhase.WAITING_RESOURCES)
                    deferred_ids.add(request.request_id)
                    continue
                except Exception as exc:
                    # Unrecoverable adapter load/config error (NotImplementedError
                    # for an unconfigured provider, ValueError for a rank/device
                    # mismatch, etc.): fail this request rather than retry it.
                    self.waiting.remove(request)
                    self._fail_request_early(request, exc)
                    progressed = True
                    continue

            # Hand ``admit`` the request's *typed* prefill tokens -- exactly the
            # ``prompt_tokens`` the non-spec ``prepare_sequence`` path receives --
            # not a text-only ``int(t.token_id)`` projection. A launchable prefill
            # can contain non-text tokens: multi-image chat prompts carry
            # ``ImageMarker`` tokens, and a generated prefix (resumed request) can
            # carry ``CoordToken`` / ``SizeToken``; none of those expose
            # ``token_id``, so stripping every token to its id would raise
            # ``AttributeError`` here -- *before* the per-request ``try`` below --
            # and abort the whole scheduler instead of admitting or failing just
            # this request. ``admit`` (like ``prepare_sequence``) expands
            # ``ImageMarker``s with the image KV prefix and reads the typed ids.
            prompt_tokens = list(request.prefill_tokens)
            # Admission capacity is the decoder's free rows: the spec decoder
            # owns a fixed, pre-reserved pool of page-table rows (its captured
            # graphs depend on them), so ``runtime.can_reserve`` -- which gates on
            # the shared page/slot pool those rows have already claimed -- does
            # not apply here.
            self.waiting.remove(request)
            lifecycle.prefill_started_at = time.perf_counter()
            lifecycle.transition(RequestPhase.PREFILLING)
            # ``length`` / ``prompt_length`` must be the row's *KV* prompt length,
            # exactly what the non-spec ``prepare_sequence`` records (it expands
            # image markers / a single-image prefix into the KV prompt length, see
            # ``moondream/runtime.py``). ``len(prompt_tokens)`` counts each
            # ``ImageMarker`` as one token but the image occupies
            # ``image_prefix_length`` KV slots, so the count excludes the image KV
            # prefix. ``request.image_length`` is exactly that prefix
            # (``image_prefix_length`` for a single image; ``num_images *
            # (image_prefix_length - 1)`` for the multi-image marker layout, where
            # each marker is already one token in ``prompt_tokens``), so
            # ``len(prompt_tokens) + request.image_length`` reconstructs the KV
            # prompt length for both layouts and leaves text-only requests
            # (``image_length == 0``) unchanged. Without this, ``build_metrics``
            # (which reports ``sequence_state.prompt_length``) under-reports image
            # prompt tokens and ``output_length`` (``length - prompt_length``) is
            # off by the image prefix. ``batch_idx`` is assigned by ``admit`` (it
            # picks the spec pool row); carry ``image_length`` so downstream
            # KV/total-length accounting on the state stays consistent.
            kv_prompt_length = len(prompt_tokens) + request.image_length
            state = SequenceState(
                batch_idx=-1,
                length=kv_prompt_length,
                max_length=request.target_length,
                prompt_length=kv_prompt_length,
                image_length=request.image_length,
                lora_slot=request.lora_slot,
            )
            # The spec decoder reads ``return_logprobs`` off the state to decide
            # whether ``step`` computes per-committed-token logprobs for this row
            # (matching the non-spec sampler, which gates the logprob gather on
            # the request). ``SequenceState`` is a plain dataclass, so attach it
            # as an extra attribute the decoder picks up via ``getattr``.
            state.return_logprobs = request.return_logprobs is True

            # Run the skill's prefill hook BEFORE building the mask + admitting:
            # ``admit`` prefills *and* samples the first (bonus) token in one
            # call, so the row's skill mask has to be installed before that
            # sample -- and some skills (e.g. query) initialise the ids their
            # ``allowed_token_ids`` depends on inside ``on_prefill``. The hook
            # only reads ``runtime.prompt_template`` (no dependency on the
            # forward having run), so running it here is safe.
            request.skill_state.on_prefill(self.runtime)

            # Per-sequence constrained-decode mask (skill whitelist/blacklist),
            # forwarded into ``admit`` so the spec path constrains the drafter +
            # verify exactly like the non-spec sampler's whitelist-then-blacklist
            # -- no gating to a text-only/unconstrained fallback. ``None`` leaves
            # the row unmasked.
            allowed_token_ids = request.skill_state.allowed_token_ids(self.runtime)
            suppressed_token_ids = request.skill_state.suppressed_token_ids(
                self.runtime
            )
            # Request-level *one-shot* suppression. The non-spec path applies
            # ``suppress_next_token_ids`` to a request's first generated token
            # only (``_build_mask_spec`` gates it on
            # ``token_count == generated_prefix_length``). ``admit`` samples that
            # exact first token, so forward the suppression here -- otherwise a
            # suppressed id can be sampled at admit, staged, and the one-shot
            # window closes (``token_count`` advances) before ``step`` could ever
            # apply it. ``None`` when the request set no one-shot suppression or
            # already generated past its prefix (resumed request).
            suppress_next_token_ids = None
            if (
                request.suppress_next_token_ids
                and request.skill_state.token_count
                == request.generated_prefix_length
            ):
                suppress_next_token_ids = request.suppress_next_token_ids
            try:
                with torch.inference_mode():
                    # Pass the request's image AND its multi-crop tiles
                    # (``image_crops``) -- exactly what the non-spec
                    # ``prepare_sequence`` forwards (it hands both to the vision
                    # encoder, which reads ``image_crops`` as the ``overlap`` so
                    # the high-res crop tiles are encoded, not just the
                    # global/thumbnail image). Forwarding ``image`` alone would
                    # give a multi-crop request an incomplete image prefill on
                    # the spec path and diverge from the non-spec output. Skill
                    # mask, one-shot suppression, and sampling params
                    # (temperature/top_p) likewise go through ``admit`` so image
                    # + constrained + non-greedy requests run on the spec path
                    # with no fallback. ``admit`` returns ``(first_token_id,
                    # first_logprob)``: the real selected-token logprob for a
                    # ``return_logprobs`` request, or ``None`` otherwise.
                    first_token_id, first_logprob = decoder.admit(
                        state,
                        prompt_tokens,
                        image=request.image,
                        image_crops=request.image_crops,
                        allowed_token_ids=allowed_token_ids,
                        suppressed_token_ids=suppressed_token_ids,
                        suppress_next_token_ids=suppress_next_token_ids,
                        temperature=float(request.temperature),
                        top_p=float(request.top_p),
                    )
            except Exception as exc:
                # ``admit`` prefills into a free spec pool row and assigns
                # ``state.batch_idx`` before the prefill's image/CUDA work
                # runs, so a mid-admit failure can leave the row reserved
                # (and ``batch_idx`` set) even though no token was staged.
                # This request has already left ``waiting`` and
                # ``lifecycle.sequence_state`` is not set yet, so no later
                # finish/zombie path will ever call ``decoder.retire`` for
                # it -- the row would leak permanently and repeated failures
                # would drain ``decoder.free_slots`` and stall unrelated spec
                # requests. Retire the row here when ``admit`` got far enough
                # to reserve one (``batch_idx`` left at its ``-1`` sentinel
                # means it never did, so there is nothing to retire).
                #
                # This is NOT ``_retire_spec_row``: that also calls
                # ``release_adapter_slot(state.lora_slot)``, which would
                # double-release the LoRA slot the block below already frees
                # via ``request.lora_slot`` (the same slot). Each resource is
                # released exactly once -- pool row via ``decoder.retire``,
                # adapter slot via the ``release_adapter_slot`` below --
                # mirroring the per-resource cleanup the non-spec
                # admit-failure path performs.
                if state.batch_idx >= 0:
                    try:
                        decoder.retire(state)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    # ``active_sequences[batch_idx]`` is only populated AFTER
                    # the ``try`` (on the success path), so on this failure
                    # path it is normally absent; pop defensively in case a
                    # future reorder registers it earlier, leaving no dangling
                    # entry.
                    self.runtime.active_sequences.pop(state.batch_idx, None)
                if lifecycle.lora_slot_ready and request.lora_slot:
                    # Release the LoRA slot we acquired for this failed admission
                    # so its refcount does not leak.
                    try:
                        self.runtime.release_adapter_slot(request.lora_slot)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    request.lora_slot = 0
                    lifecycle.lora_slot_ready = False
                self._fail_request_early(request, exc)
                progressed = True
                continue

            lifecycle.sequence_state = state
            lifecycle.prefill_completed_at = time.perf_counter()
            self.runtime.active_sequences[state.batch_idx] = state

            # Stage the first (prefill/bonus) token exactly like the non-spec
            # prefill path: consume_step + streaming + finish check.
            #
            # Type the admit token through the runtime hook -- not a hard-coded
            # ``TextToken``. On a spatial runtime the first generated id can be
            # ``coord_id`` / ``size_id`` (point/detect apply the skill mask before
            # this first sample), so it must become a ``CoordToken`` /
            # ``SizeToken`` the way the non-spec prefill path does via
            # ``post_sample`` -> ``materialize_tokens``; otherwise the first
            # coordinate/box component is dropped. ``admit`` surfaces the
            # admit-position side-values (the target's last hidden + sampling
            # knobs the spatial hook needs to decode the coord/size value) on
            # ``state.admit_side_values`` -- a :class:`SpecSideValues`, so
            # ``_materialize_spec_tokens`` routes it through the spec-aware
            # ``materialize_spec_tokens`` hook (not the non-spec one). A text-only
            # runtime leaves it ``None`` and the id materialises as a plain
            # ``TextToken`` (unchanged).
            admit_side_values = getattr(state, "admit_side_values", None)
            (typed_first,) = self._materialize_spec_tokens(
                [lifecycle], [[int(first_token_id)]], admit_side_values
            )
            token = typed_first[0]
            # ``admit`` returns the real first-token logprob for a
            # ``return_logprobs`` request (the sampler's selected-token logprob,
            # matching the non-spec prefill path's transferred ``sampled_logprobs``
            # for token0) and ``None`` otherwise. Stage it through unchanged --
            # no 0.0 greedy-approximation placeholder. The macro-step supplies
            # real per-token logprobs for every subsequently committed token.
            self._spec_stage_token(lifecycle, token, logprob=first_logprob)
            progressed = True
            if self._mark_finished_if_needed(lifecycle):
                lifecycle.finalized = True
                self._retire_spec_row(state)
                continue
            self.running.push(lifecycle)
        return progressed

    def _spec_stage_token(
        self,
        seq: RequestLifecycle,
        token: Token,
        *,
        logprob: float | None,
    ) -> None:
        """Stage one spec-committed token onto ``seq``.

        Mirrors ``RequestLifecycle.stage_token`` but is the single spec-path
        staging point. ``logprob`` is the macro-step's per-committed-token
        logprob (``SpecStepResult.logprobs``) for a ``return_logprobs`` request,
        or ``None`` when the request did not ask for logprobs; it flows through
        to ``stage_token`` exactly like the non-spec single-token path.
        """
        seq.stage_token(self.runtime, token, logprob=logprob)

    def _retire_spec_row(self, state: SequenceState) -> None:
        """Release every resource a finished spec row holds.

        The single spec-path retire point. The spec decoder owns the pool's
        page-table rows, so ``decoder.retire`` (not ``runtime.release_sequence``)
        reclaims the row; ``_release_sequence`` deliberately early-returns for a
        spec runtime to avoid erasing those persistently-reserved pages. But the
        adapter slot is *not* part of that pool -- ``_spec_admit`` acquires it via
        ``_acquire_adapter_slot`` exactly like the non-spec prefill path, which
        releases it in ``runtime.release_sequence`` -> ``release_adapter_slot``.
        Since the spec path skips ``release_sequence``, release the slot here too;
        otherwise every completed adapter spec request leaks its LoRA slot until
        the adapter pool starves. ``release_adapter_slot`` is a no-op for
        ``lora_slot == 0`` (a base-model request), so this is unconditional.
        """
        decoder = self.runtime.spec.decoder
        decoder.retire(state)
        self.runtime.active_sequences.pop(state.batch_idx, None)
        self.runtime.release_adapter_slot(state.lora_slot)

    def _spec_decode_step(self) -> bool:
        """Run one spec macro-step, overlapped depth-1.

        Ordering per tick: snapshot membership -> reserve in-flight refs ->
        commit the previous macro-step -> build this step's masks -> launch.

        Membership is an *optimistic* snapshot taken BEFORE the commit. The
        commit can FINALIZE a sequence (it hit ``max_new_tokens`` / EOS); because
        the snapshot predates the commit, such a row is still in ``active`` but
        must NOT be launched into another macro-step -- its ``state.length`` can
        already equal ``max_length`` and a real decoder that reserves only the
        request budget would advance/write past the row. So after the commit the
        snapshot is filtered: rows the commit finalized are dropped from this
        launch and retired SYNCHRONOUSLY (their reserved ref is released and the
        row retired here, mirroring ``_commit_spec``'s zombie branch), exactly
        like the non-spec launch set, which excludes finalized rows via
        ``_can_dispatch``. Only continuing rows are launched. The depth-1
        ``has_pending_work`` leak-guard (``_pending_spec is not None``) still
        covers a launched-but-uncommitted step for those continuing rows.

        Why the commit happens before the launch (and before the mask build):
        ``_commit_spec`` advances each committed sequence's skill state via
        ``consume_step``, and STATEFUL constrained skills evolve their allowed-
        token set per committed token (e.g. point toggles ``[coord,eos]`` <->
        ``[coord]`` every coordinate; detect cycles x->y->size). The mask handed
        to ``decoder.step`` must therefore be recomputed from the skill state
        *after* the prior run is committed -- exactly the non-spec
        commit-before-finalize invariant (``advance`` docstring), where
        ``_build_mask`` re-queries ``allowed_token_ids`` post-commit.
        Snapshotting once at ``admit`` and reusing it (as the spec path did)
        left stateful skills with a stale mask for every later position.

        ``inflight_refs`` is reserved for this launch BEFORE the commit so a
        sequence present in both the previous step and this one (the common
        continuing case) never drops to zero refs mid-commit and is not retired
        out from under the relaunch; its ref is decremented again when this
        launch is itself committed next tick, keeping the count balanced.
        """
        decoder = self.runtime.spec.decoder
        pending = self._pending_spec
        self._pending_spec = None

        active = [
            seq
            for seq in self.running
            if not seq.finished and seq.sequence_state is not None
        ]
        # Reserve this launch's ref before the commit (see docstring): protects
        # continuing sequences from a premature retire when the commit drops
        # their previous-step ref.
        for seq in active:
            seq.inflight_refs += 1

        # The commit + launch below can raise (e.g. ``decoder.step`` on a CUDA
        # error). Until ``_pending_spec`` is installed those reserved refs are
        # owned by nothing -- no future ``_commit_spec`` will decrement them -- so
        # a finishing row would stay stuck in ``FINALIZING`` and its pool row /
        # adapter slot would leak. Mirror the non-spec launch path
        # (``launch_decode_step``): on any failure before the pending step is
        # installed, roll back exactly the reservation added above, then re-raise.
        try:
            if pending is not None:
                self._commit_spec(pending)

            # The commit above can FINALIZE a row (its last allowed token was
            # ``length``-committed), which sets ``finalized`` and removes it from
            # ``running`` but cannot retire it yet because the ref reserved above
            # keeps ``inflight_refs > 0``. That row is still in ``active`` (the
            # snapshot predates the commit), so it must NOT be launched into
            # another macro-step: its ``SequenceState.length`` can already equal
            # ``max_length``, and ``_build_spec_step_masks`` returns ``None`` caps
            # for a finalized row, so the decoder would be asked to advance/write
            # past a row that has no budget left (the non-spec launch set excludes
            # finalized rows via ``_can_dispatch``). Drop those rows from the
            # launch and turn each one's reserved ref into an immediate retire
            # (mirrors ``_commit_spec``'s zombie branch: COMPLETED then retire when
            # the last ref clears).
            launchable = []
            for seq in active:
                if seq.finalized:
                    seq.inflight_refs -= 1
                    if seq.inflight_refs == 0:
                        seq.transition(RequestPhase.COMPLETED)
                        self._retire_spec_row(seq.state)
                else:
                    launchable.append(seq)
            active = launchable

            if active:
                # Recompute each row's skill mask from the now-current skill
                # state (post-commit), so stateful skills constrain the drafter +
                # verify with the up-to-date allowed/suppressed set this step --
                # not the admit-time snapshot. ``commit_caps`` is the per-row min
                # of two accept-truncations: the STATEFUL-mask cap (a row whose
                # allowed/suppressed set changes per committed token is held to one
                # token this step, since the single per-step mask is only exact for
                # one constraint transition per run -- then re-masked from the
                # now-current skill state next step), and the REMAINING-BUDGET cap
                # (``max_new_tokens - token_count``) so the decoder's KV/proposer
                # state never advances past the request's output limit. Rows
                # finalized by the commit above were already dropped from
                # ``active``. See ``_build_spec_step_masks`` / ``decoder.step``.
                (
                    allowed_token_ids,
                    suppressed_token_ids,
                    commit_caps,
                ) = self._build_spec_step_masks(active)
                with torch.inference_mode():
                    result = decoder.step(
                        [seq.state for seq in active],
                        allowed_token_ids=allowed_token_ids,
                        suppressed_token_ids=suppressed_token_ids,
                        commit_caps=commit_caps,
                    )
                self._pending_spec = (active, result)
        except Exception:
            # Failure before ``_pending_spec`` was installed: undo this launch's
            # reservation so the count stays balanced and no row is left with a
            # phantom ref (mirrors ``launch_decode_step``'s rollback). ``active``
            # is the only set whose refs this method added; ``_commit_spec``
            # already decremented whatever it committed before raising.
            if self._pending_spec is None:
                for seq in active:
                    seq.inflight_refs -= 1
            raise

        return bool(active) or pending is not None

    def _drain_spec(self) -> None:
        """Commit any in-flight spec macro-step, launching no new work.

        ``_spec_decode_step`` overlaps depth-1 by launching the next macro-step
        *before* committing the previous one, so between ticks a launched-but-
        uncommitted step lives in ``_pending_spec``. Unlike ``_spec_decode_step``
        this deliberately does NOT launch a new step -- it only flushes the
        outstanding one -- so it is safe to call from the pause/drain path
        (``_drain_pipeline``) without admitting more spec work while the engine is
        trying to quiesce. ``_commit_spec`` finalizes/retires any sequence that
        ended in the committed run and never repopulates ``_pending_spec``, so a
        single commit empties it; the loop is defensive and converges. No-op when
        the runtime does not speculate (``_pending_spec`` stays ``None``).
        """
        while self._pending_spec is not None:
            pending = self._pending_spec
            self._pending_spec = None
            with stream_context(self._compute_stream):
                with torch.inference_mode():
                    self._commit_spec(pending)

    def _materialize_spec_tokens(
        self,
        active: List[RequestLifecycle],
        runs: Sequence[Sequence[int]],
        side_values: object | None,
    ) -> list[list[Token]]:
        """Type each sequence's committed run via the runtime hook.

        ``runs[i]`` is the committed id list for ``active[i]``.

        Dispatch is on ``side_values`` (the macro-step's :class:`SpecSideValues`),
        which carries the target's per-committed-position final hidden + sampling
        knobs a spatial runtime decodes coord/size ids from. It must NOT be handed
        to the non-spec ``materialize_tokens`` hook: that hook's step-handle is the
        ``post_sample`` aux handle (Moondream unpacks it as ``(slot, batch_size)``
        and synchronises slot-local staging), so feeding it a ``SpecSideValues``
        raises (cannot unpack) or reads the wrong layout. So:

        * ``side_values`` set -> route through the dedicated
          ``materialize_spec_tokens`` hook (the spec analog that decodes coord/size
          from the packed hidden). If the runtime exposes no such hook (text-only,
          or a runtime that does not type spatial tokens on the spec path),
          *guard* by materialising plain ``TextToken``s rather than misusing the
          non-spec hook.
        * ``side_values is None`` -> a text-only macro-step; use the normal
          ``materialize_tokens`` path with a ``None`` handle (its plain-text
          branch), matching the non-spec single-token commit.

        Returns the typed tokens re-grouped per sequence (parallel to ``runs``).
        """
        flat_ids: list[int] = [int(t) for run in runs for t in run]
        if not flat_ids:
            return [[] for _ in runs]
        token_ids_cpu = torch.tensor(flat_ids, dtype=torch.long)
        # ``batch_idx`` parallel to the flattened ids (the spec batch row each
        # committed token belongs to); a spatial hook indexes side-values by the
        # active-sequence order, but keep the row ids aligned for hooks that key
        # off them.
        batch_indices: list[int] = [
            seq.state.batch_idx for seq, run in zip(active, runs) for _ in run
        ]
        batch_idx = torch.tensor(batch_indices, dtype=torch.long)
        if side_values is not None:
            spec_hook = self._hooks.materialize_spec_tokens
            if spec_hook is not None:
                flat_tokens = spec_hook(
                    token_ids_cpu, active, batch_idx, side_values
                )
            else:
                # Guard: a runtime that does not type spatial tokens on the spec
                # path still returned side-values; never pass them to the
                # non-spec hook -- materialise plain text ids.
                flat_tokens = [
                    TextToken(token_id=int(t))
                    for t in token_ids_cpu.view(-1).tolist()
                ]
        else:
            # Text-only macro-step: the non-spec hook's ``step_handle is None``
            # branch (plain text) applies.
            flat_tokens = self._materialize_tokens(
                token_ids_cpu,
                active,
                batch_idx,
                None,
            )
        # Re-group the flat typed tokens back into per-sequence runs.
        grouped: list[list[Token]] = []
        cursor = 0
        for run in runs:
            n = len(run)
            grouped.append(list(flat_tokens[cursor:cursor + n]))
            cursor += n
        return grouped

    def _commit_spec(self, pending) -> None:
        """Commit one previously-launched spec macro-step (depth-1).

        Resolving ``result.tokens`` blocks on the macro-step's deferred D2H, but
        that transfer completed while the GPU ran the step launched after it, so
        the wait is hidden. Mirrors ``commit_step``'s decode commit: decrement
        ``inflight_refs``, skip + retire zombies, else stage the variable run of
        committed tokens (typed via the runtime hook, each with its macro-step
        logprob) and finish/retire on EOS or length cap.
        """
        active, result = pending
        # ``tokens``/``logprobs`` resolve lazily off the (already-landed) D2H;
        # ``side_values`` holds device tensors read at ``step`` time.
        tokens = result.tokens
        logprobs = result.logprobs  # None when nobody requested logprobs
        side_values = result.side_values
        # Type the whole step's committed ids up front (coord/size via the
        # runtime hook using ``side_values``), then stage each typed token below
        # so a mid-run EOS still drops the trailing typed tokens correctly.
        typed = self._materialize_spec_tokens(active, tokens, side_values)
        for i, (seq, typed_run) in enumerate(zip(active, typed)):
            seq.inflight_refs -= 1
            if seq.finalized:
                # Zombie: finished by an earlier commit while still in flight here.
                # When its last in-flight ref drops, mark the lifecycle COMPLETED
                # *before* retiring the row, mirroring the non-spec zombie path in
                # ``commit_step`` (``seq.transition(COMPLETED)`` then release). The
                # finishing commit left it FINALIZING because ``inflight_refs > 0``;
                # without this transition a normally-completed spec request stays
                # stuck in FINALIZING forever after its row is retired.
                if seq.inflight_refs == 0:
                    seq.transition(RequestPhase.COMPLETED)
                    self._retire_spec_row(seq.state)
                continue
            seq_logprobs = logprobs[i] if logprobs is not None else None
            for j, token in enumerate(typed_run):
                seq.state.advance()  # KV length mirrors the committed token
                logprob = seq_logprobs[j] if seq_logprobs is not None else None
                self._spec_stage_token(seq, token, logprob=logprob)
                if self._mark_finished_if_needed(seq):
                    seq.finalized = True
                    self.running.remove(seq)
                    if seq.inflight_refs == 0:
                        self._retire_spec_row(seq.state)
                    break

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
        temps_gpu, top_ps_gpu = staging.stage_sampling_params(requests)
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

            # One H2D copy stages batch_idx/input_pos/lora_slot_ids together
            # (they share a packed buffer) instead of three separate launches.
            slot.meta.copy_inputs_to_gpu()
            batch_idx = slot.meta.batch_idx.gpu[:batch_size]

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
        # Speculative decoding: the spec decoder owns the page-table rows (a
        # fixed, persistently-reserved pool backing its captured CUDA graphs).
        # Releasing the batch_idx here would erase pages out from under those
        # graphs; row reuse is handled by ``decoder.retire`` at finish instead.
        if self.runtime.spec is not None:
            return
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

    def _build_spec_step_masks(
        self, sequences: List[RequestLifecycle]
    ) -> tuple[
        list[Optional[Sequence[int]]],
        list[Optional[Sequence[int]]],
        list[Optional[int]],
    ]:
        """Per-row skill masks (+ commit caps) for a spec macro-step.

        The spec-decode analog of ``_build_mask_spec``'s allowed/suppressed
        re-query. ``_spec_decode_step`` calls this AFTER committing the previous
        macro-step (so ``skill_state`` reflects every token committed through the
        prior run), then threads the result into ``decoder.step`` so the drafter
        + verify constrain to the *current* allowed/suppressed set instead of the
        snapshot taken once at ``admit``. This is what keeps STATEFUL skills
        correct on the spec path (point toggles ``[coord,eos]`` <-> ``[coord]``
        every coordinate; detect cycles x->y->size), mirroring how the non-spec
        ``_build_mask`` refreshes the mask each step.

        Finalized/zombie rows come back unconstrained (``None``), exactly like
        ``_build_mask_spec`` -- their tokens are dropped at commit, so masking
        them is moot.

        Deliberately does NOT re-apply the request-level one-shot
        ``suppress_next_token_ids``: that suppression targets a request's *first*
        generated token only and is applied once, at ``admit`` (which samples
        that token); re-applying it here would wrongly suppress past the one-shot
        window.

        The third return is the per-row ``commit_caps`` (parallel to
        ``sequences``): the per-row upper bound on tokens this macro-step may
        commit, the ``min`` of two independent accept-truncation caps (``None``
        in either slot means that cap is absent / +inf):

        * the STATEFUL-mask cap -- ``1`` for a row whose skill mask changes per
          committed token, else absent. The single per-step mask is exact only
          for one constraint transition per committed run: a macro-step commits a
          *variable* run of ``a_i + 1`` tokens under ONE mask, so a stateful
          row's 2nd..Nth positions would be verified under the stale
          1st-position mask (e.g. a detect run could suppress the required
          ``size_id`` or accept a ``coord`` where ``size`` is required). Capping
          such a row to one committed token forces exactly one transition per run
          -- the regime where the per-step mask IS exact, identical to the
          non-spec one-token-per-step path -- and the next step re-queries the
          mask from the now-current skill state.

        * the REMAINING-BUDGET cap -- ``max_new_tokens - token_count``, the
          number of output tokens the request may still emit. ``_commit_spec``
          already stops *staging* once ``max_new_tokens`` is reached, but the
          decoder advances its KV / proposer state through the *whole* committed
          run regardless, so without this cap the row's pool state (and the
          optimistic depth-1 zombie step launched off it) runs past the request's
          target length even though the extra tokens are never emitted. Capping
          the committed run to the remaining budget makes the decoder finish the
          row at exactly ``max_new_tokens`` -- the same hard stop the non-spec
          path hits -- so this step's commit lands the final token and
          ``_commit_spec`` marks it ``length``-finished. A continuing (not
          finalized) row always has ``token_count < max_new_tokens`` here (else a
          prior commit would have finished it), so the budget is ``>= 1``.

        A non-stateful row with budget to spare stays uncapped (``None``) so it
        keeps the full multi-token speculative accept. Finalized/zombie rows come
        back ``None`` (their tokens are dropped at commit). Returns
        ``(allowed_per_row, suppressed_per_row, commit_caps_per_row)``.
        """
        allowed_tokens: list[Optional[Sequence[int]]] = []
        suppressed_tokens: list[Optional[Sequence[int]]] = []
        commit_caps: list[Optional[int]] = []
        for seq in sequences:
            if seq.finalized:
                allowed_tokens.append(None)
                suppressed_tokens.append(None)
                commit_caps.append(None)
                continue
            allowed = seq.skill_state.allowed_token_ids(self.runtime)
            suppressed = seq.skill_state.suppressed_token_ids(self.runtime)
            allowed_tokens.append(allowed)
            suppressed_tokens.append(suppressed)
            stateful_cap = (
                1 if self._mask_is_stateful(seq, allowed, suppressed) else None
            )
            budget_cap = self._remaining_commit_budget(seq)
            commit_caps.append(_min_cap(stateful_cap, budget_cap))
        return allowed_tokens, suppressed_tokens, commit_caps

    def _remaining_commit_budget(self, seq: RequestLifecycle) -> Optional[int]:
        """Tokens ``seq`` may still emit this macro-step before ``max_new_tokens``.

        The committed run must not advance the decoder past the request's output
        budget (the non-spec path stops sampling exactly at ``max_new_tokens``).
        ``skill_state.token_count`` already reflects every token committed through
        the prior macro-step (this runs post-commit), so the remaining budget is
        ``max_new_tokens - token_count``. Returns ``None`` (uncapped) only when
        ``max_new_tokens`` is unset; otherwise the live remaining count, which is
        ``>= 1`` for any not-yet-finalized row (a row that had already reached the
        limit would have been finished + retired at the prior commit and is not in
        this step's ``active`` set).
        """
        max_new = seq.request.max_new_tokens
        if max_new is None:
            return None
        return max(0, max_new - seq.skill_state.token_count)

    def _mask_is_stateful(
        self,
        seq: RequestLifecycle,
        allowed: Optional[Sequence[int]],
        suppressed: Optional[Sequence[int]],
    ) -> bool:
        """Whether ``seq``'s skill mask can change WITHIN one committed run.

        A spec macro-step commits a variable run of tokens under a single
        per-step mask, so a row whose allowed/suppressed set evolves per
        committed token (point cycles coord/eos; detect cycles x->y->size; query
        injects a fixed post-reasoning prefix one id at a time, and toggles its
        suppression as it leaves reasoning) must be capped to one committed token
        per step -- otherwise the run's later positions verify under the stale
        first-position mask. This decides that cap from the live skill state,
        model-agnostically (the scheduler never imports a model's skills).

        A skill may declare its mask precisely via an optional
        ``mask_is_stateful`` attribute / property on its ``SkillState`` (``True``
        => evolving, ``False`` => provably constant); when present that wins. In
        its absence the conservative behavioural rule is: any ACTIVE constraint
        (a non-empty ``allowed`` whitelist or ``suppressed`` blacklist this step)
        is treated as potentially stateful and capped, because the scheduler
        cannot prove the set is position-independent without model knowledge.
        Correctness-first: over-capping a constant-mask skill only costs that
        row's intra-step speculation (it still advances one token per step), while
        under-capping a stateful one corrupts constrained output. A row with no
        active constraint is never stateful and keeps the full multi-token accept.
        """
        declared = getattr(seq.skill_state, "mask_is_stateful", None)
        if declared is not None:
            return bool(declared)
        return bool(allowed) or bool(suppressed)

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
        is_text = isinstance(last_token, TextToken)
        eos_hit = is_text and last_token.token_id == eos_id
        if is_text and not eos_hit:
            # A skill (e.g. chat) may terminate a turn on its own token — a
            # ChatML-style ``<|im_end|>`` can differ from the model's ``eos_id``.
            stop_ids = seq.skill_state.stop_token_ids(self.runtime)
            if stop_ids is not None and last_token.token_id in stop_ids:
                eos_hit = True
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
