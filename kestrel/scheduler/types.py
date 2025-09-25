"""Typed containers used by the scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor
from PIL import Image

from kestrel.models import SequenceState


@dataclass
class GenerationRequest:
    """User-facing request tracked by the scheduler."""

    request_id: int
    prompt: str
    prompt_tokens: Tensor
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stream_callback: Optional["StreamCallback"] = None
    image: Optional[Image.Image] = None
    image_length: int = 0

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
        self.prompt_length = int(self.prompt_tokens.shape[0])
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
    generated_tokens: List[int] = field(default_factory=list)
    pending_token: Optional[int] = None
    last_logits: Optional[Tensor] = None
    finished: bool = False
    finish_reason: Optional[str] = None
    stream_offset: int = 0
    started_at: float = field(default=0.0)
    first_token_time: Optional[float] = None
    completed_at: Optional[float] = None

    def stage_token(self, token_id: int, logits: Tensor) -> None:
        token = int(token_id)
        self.generated_tokens.append(token)
        self.pending_token = token
        self.last_logits = logits.detach().to("cpu")
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    @property
    def total_length(self) -> int:
        return (
            self.request.prompt_length
            + self.request.image_length
            + len(self.generated_tokens)
        )

    @property
    def last_token(self) -> Optional[int]:
        return self.generated_tokens[-1] if self.generated_tokens else None

    def needs_decode(self) -> bool:
        return not self.finished and self.pending_token is not None


@dataclass
class SchedulerResult:
    """Final materialisation of a completed request."""

    request_id: int
    prompt: str
    tokens: List[int]
    text: str
    finish_reason: str
    metrics: "RequestMetrics"


@dataclass
class StreamUpdate:
    """Incremental token update emitted while a request is decoding."""

    request_id: int
    token: int
    text: str
    token_index: int


StreamCallback = Callable[[StreamUpdate], None]


@dataclass
class RequestMetrics:
    prompt_tokens: int
    decode_tokens: int
    processing_latency_s: float
    ttft_s: float
    decode_latency_s: float
