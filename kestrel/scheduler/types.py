"""Typed containers used by the scheduler."""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import MoondreamRuntime, SequenceState, Token
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.skills import SkillSpec, SkillState, DecodeStep
from kestrel.scheduler.transfer import TransferHandle


@dataclass
class GenerationRequest:
    """User-facing request tracked by the scheduler."""

    request_id: int
    prompt: str
    prompt_tokens: Tensor
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
    adapter: Optional[str] = None
    lora_slot: int = 0  # Slot in workspace; 0 = no LoRA

    prompt_length: int = field(init=False)

    def __post_init__(self) -> None:
        tokens = self.prompt_tokens
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)
        if tokens.ndim != 1:
            raise ValueError(
                f"prompt_tokens must be 1D or shaped (1, L); got {self.prompt_tokens.shape}"
            )
        self.prompt_tokens = tokens.to(dtype=torch.long, device="cpu")
        # Runtime unconditionally prepends a BOS token, so account for it in the
        # prompt length even though individual skills omit it from their token
        # buffers.
        self.prompt_length = int(self.prompt_tokens.shape[0]) + 1
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
class ScheduledSequence:
    """Runtime state for an admitted request."""

    request: GenerationRequest
    state: SequenceState
    skill_state: SkillState
    pending_token: Optional[Token] = None
    finished: bool = False
    finish_reason: Optional[str] = None
    started_at: float = field(default=0.0)
    prefill_started_at: Optional[float] = None
    first_token_time: Optional[float] = None
    completed_at: Optional[float] = None
    # Pipelining state: finalized means EOS/length cap reached, inflight_refs
    # counts how many in-flight steps reference this sequence.
    finalized: bool = False
    inflight_refs: int = 0

    def stage_token(
        self,
        runtime: MoondreamRuntime,
        token: Token,
    ) -> None:
        self.pending_token = token
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
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
        return not self.finished and self.pending_token is not None


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

    This is a pure selection of which sequences to include in the next decode
    step. It does not mutate any state - the scheduler remains a stateless
    selector. Actual state changes (inflight_refs increment, GPU dispatch)
    happen when the plan is executed via launch_forward_async.
    """

    sequences: List["ScheduledSequence"]
