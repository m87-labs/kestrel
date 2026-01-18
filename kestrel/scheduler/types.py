"""Typed containers used by the scheduler."""

import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import pyvips
from kestrel.moondream.runtime import MoondreamRuntime, SequenceState, Token
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.skills import SkillSpec, SkillState, DecodeStep


class RequestPhase(str, Enum):
    NEW = "new"
    WAITING_RESOURCES = "waiting_resources"
    READY_FOR_PREFILL = "ready_for_prefill"
    PREFILLING = "prefilling"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


@dataclass
class RequestLifecycle:
    """Scheduler-owned lifecycle state for a request."""

    request: "GenerationRequest"
    skill_state: SkillState
    sequence_state: Optional[SequenceState] = None
    phase: RequestPhase = RequestPhase.NEW

    # Resource flags
    has_image: bool = False
    crops_ready: bool = False
    lora_slot_ready: bool = False
    prefix_cache_hit: bool = False

    # Decode pipeline state
    packed_pending_ready: bool = False
    uncommitted_prefill_token: bool = False
    inflight_refs: int = 0
    finalized: bool = False
    finished: bool = False

    # Timing
    submitted_at: float = 0.0
    prefill_started_at: Optional[float] = None
    prefill_completed_at: Optional[float] = None
    first_token_time: Optional[float] = None
    completed_at: Optional[float] = None

    # Output
    finish_reason: Optional[str] = None
    error: Optional[BaseException] = None

    def transition(self, phase: RequestPhase) -> None:
        """Update phase with minimal bookkeeping."""
        if phase == self.phase:
            return
        self.phase = phase
        if phase == RequestPhase.COMPLETED and self.completed_at is None:
            self.completed_at = time.perf_counter()

    @property
    def state(self) -> SequenceState:
        if self.sequence_state is None:
            raise RuntimeError("sequence_state is not set")
        return self.sequence_state

    def build_metrics(
        self,
        *,
        decode_tokens: int,
        cached_tokens: Optional[int] = None,
    ) -> "RequestMetrics":
        queued_at = self.submitted_at
        prefill_started_at = self.prefill_started_at or queued_at
        prefill_completed_at = self.prefill_completed_at or prefill_started_at
        completed_at = self.completed_at or time.perf_counter()
        first_token_time = self.first_token_time or completed_at
        if self.sequence_state is not None:
            prompt_tokens = self.sequence_state.prompt_length
            cached = self.sequence_state.reused_page_count
        else:
            prompt_tokens = self.request.prompt_length
            cached = 0
        if cached_tokens is None:
            cached_tokens = cached
        return RequestMetrics(
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_time_ms=max((prefill_completed_at - prefill_started_at) * 1000.0, 0.0),
            ttft_ms=max((first_token_time - queued_at) * 1000.0, 0.0),
            decode_time_ms=max((completed_at - prefill_completed_at) * 1000.0, 0.0),
            cached_tokens=cached_tokens,
        )

    def stage_token(
        self,
        runtime: MoondreamRuntime,
        token: Token,
    ) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        if self.phase not in (
            RequestPhase.RUNNING,
            RequestPhase.FINALIZING,
            RequestPhase.COMPLETED,
        ):
            self.transition(RequestPhase.RUNNING)
        step = DecodeStep(
            token=token,
            position=self.skill_state.token_count,
        )
        self.skill_state.consume_step(runtime, step)
        callback = self.request.stream_callback
        if callback is not None:
            delta = self.skill_state.pop_stream_delta(runtime)
            if delta:
                callback(
                    StreamUpdate(
                        request_id=self.request.request_id,
                        token=token,
                        text=delta,
                        token_index=self.skill_state.token_count - 1,
                    )
                )

    @property
    def total_length(self) -> int:
        return (
            self.request.prompt_length
            + self.request.image_length
            + self.skill_state.token_count
        )

    @property
    def last_token(self) -> Optional[Token]:
        tokens: Sequence[Token] = self.skill_state.tokens
        return tokens[-1] if tokens else None

    def needs_decode(self) -> bool:
        return not self.finished and self.packed_pending_ready


@dataclass
class GenerationRequest:
    """User-facing request tracked by the scheduler."""

    request_id: int
    prompt: str
    prompt_tokens: Sequence[Token]
    max_new_tokens: int
    skill: SkillSpec = field(repr=False)
    request_context: object = field(repr=False)
    temperature: float = 0.0
    top_p: float = 1.0
    stream_callback: Optional["StreamCallback"] = None
    image: Optional[pyvips.Image | np.ndarray] = None
    image_hash: Optional[bytes] = None  # SHA256 hash for prefix caching
    image_crops: Optional[OverlapCropOutput] = None
    image_length: int = 0
    submitted_at: float = 0.0
    skill_state: Optional[SkillState] = field(default=None, repr=False)
    lifecycle: RequestLifecycle = field(init=False, repr=False)
    adapter: Optional[str] = None
    lora_slot: int = 0  # Slot in workspace; 0 = no LoRA

    prompt_length: int = field(init=False)

    def __post_init__(self) -> None:
        tokens = list(self.prompt_tokens)
        self.prompt_tokens = tokens
        # Prompt tokens are already materialized (including BOS if required by
        # the template), so prompt_length is the list length.
        self.prompt_length = len(tokens)
        if self.request_context is None:
            raise ValueError("request_context must be provided")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1]")

    @property
    def target_length(self) -> int:
        return self.prompt_length + self.image_length + self.max_new_tokens


@dataclass
class SchedulerResult:
    """Final materialisation of a completed request."""

    request_id: int
    tokens: List[Token]
    finish_reason: str
    metrics: "RequestMetrics"
    output: Dict[str, object]


@dataclass
class StreamUpdate:
    """Incremental token update emitted while a request is decoding."""

    request_id: int
    token: Token
    text: str
    token_index: int


StreamCallback = Callable[[StreamUpdate], None]


@dataclass
class RequestMetrics:
    prompt_tokens: int
    decode_tokens: int
    prefill_time_ms: float
    ttft_ms: float
    decode_time_ms: float
    cached_tokens: int = 0  # KV positions reused from prefix cache


@dataclass
class StepPlan:
    """Decode step plan returned by schedule_decode_step.

    This is a pure selection of which requests to include in the next decode
    step. It does not mutate any state - the scheduler remains a stateless
    selector. Actual state changes (inflight_refs increment, GPU dispatch)
    happen when the plan is executed via launch_forward_async.
    """

    sequences: List["RequestLifecycle"]
