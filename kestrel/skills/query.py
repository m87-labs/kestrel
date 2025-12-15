"""Query skill leveraging the existing text generation flow."""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyvips
import torch
import numpy as np
from torch import Tensor

from kestrel.moondream.runtime import CoordToken, TextToken

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class QuerySettings:
    """Sampling parameters supplied with a query invocation."""

    temperature: float
    top_p: float
    max_tokens: int


@dataclass(slots=True)
class QueryRequest:
    """Validated query payload aligned with the fal_inference API."""

    question: str
    image: Optional[pyvips.Image | np.ndarray]
    reasoning: bool
    stream: bool
    settings: QuerySettings


class QuerySkill(SkillSpec):
    """Default skill emitting plain text answers."""

    def __init__(self) -> None:
        super().__init__(name="query")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Tensor:
        if not isinstance(request_context, QueryRequest):
            raise ValueError("QuerySkill.build_prompt_tokens requires a QueryRequest")
        prompt = request_context.question
        template = runtime.config.tokenizer.templates["query"]
        prefix: Sequence[int] = template["prefix"]
        suffix: Sequence[int] = template["suffix"]
        encoded = runtime.tokenizer.encode(prompt).ids if prompt else []
        reasoning = request_context.reasoning
        if reasoning:
            thinking_id = runtime.config.tokenizer.thinking_id
            ids = [*prefix, *encoded, thinking_id]
        else:
            ids = [*prefix, *encoded, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: "QueryRequest",
    ) -> "QuerySkillState":
        if not isinstance(request_context, QueryRequest):
            raise ValueError("QuerySkill.create_state requires a QueryRequest context")
        return QuerySkillState(self, request, request_context)


class QuerySkillState(SkillState):
    """Skill state that buffers tokens and exposes plain text outputs."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        query_request: QueryRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = query_request
        self._reasoning_enabled = bool(query_request.reasoning)
        self._collecting_reasoning = self._reasoning_enabled
        self._reasoning_tokens: List[int] = []
        self._answer_tokens: List[int] = []
        self._reasoning_chunks: List[Tuple[List[int], List[Tuple[float, float]]]] = []
        self._current_chunk_tokens: List[int] = []
        self._current_chunk_points: List[Tuple[float, float]] = []
        self._pending_coord: Optional[float] = None
        self._answer_id: Optional[int] = None
        self._start_ground_id: Optional[int] = None
        self._end_ground_id: Optional[int] = None
        self._streaming = bool(query_request.stream)
        self._answer_stream_offset = 0

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        if self._reasoning_enabled:
            self._ensure_token_ids(runtime)
        self.append_token(step.token)

        if not self._reasoning_enabled:
            if isinstance(step.token, TextToken):
                self._answer_tokens.append(step.token.token_id)
            return None

        if self._collecting_reasoning:
            token = step.token
            if isinstance(token, TextToken):
                token_id = token.token_id
                if token_id == self._answer_id:
                    self._collecting_reasoning = False
                    self._flush_current_chunk()
                    self._pending_coord = None
                    self._answer_stream_offset = 0
                    return None
                if token_id == self._start_ground_id or token_id == self._end_ground_id:
                    self._flush_current_chunk()
                    self._pending_coord = None
                    return None
                self._reasoning_tokens.append(token_id)
                self._current_chunk_tokens.append(token_id)
                return None
            if isinstance(token, CoordToken):
                value = float(token.pos)
                if self._pending_coord is None:
                    self._pending_coord = value
                else:
                    self._current_chunk_points.append((self._pending_coord, value))
                    self._pending_coord = None
                return None
            # Ignore other token types during reasoning
            return None

        # Answer phase
        if isinstance(step.token, TextToken):
            self._answer_tokens.append(step.token.token_id)
        return None

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        if not self._streaming:
            return None
        if self._collecting_reasoning:
            return None
        if not self._answer_tokens:
            return None
        text = runtime.tokenizer.decode(self._answer_tokens)
        if len(text) <= self._answer_stream_offset:
            return None
        chunk = text[self._answer_stream_offset :]
        self._answer_stream_offset = len(text)
        if not chunk:
            return None
        return chunk

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        if self._reasoning_enabled:
            self._flush_current_chunk()

        tokenizer = runtime.tokenizer
        answer_text = (
            tokenizer.decode(self._answer_tokens) if self._answer_tokens else ""
        )

        output: Dict[str, Any] = {"answer": answer_text}

        if self._reasoning_enabled:
            reasoning_text = (
                tokenizer.decode(self._reasoning_tokens)
                if self._reasoning_tokens
                else ""
            )
            grounding: List[Dict[str, object]] = []
            cursor = 0
            for tokens, points in self._reasoning_chunks:
                chunk_text = tokenizer.decode(tokens) if tokens else ""
                length = len(chunk_text)
                if points:
                    grounding.append(
                        {
                            "start_idx": cursor,
                            "end_idx": cursor + length,
                            "points": points.copy(),
                        }
                    )
                cursor += length
            output["reasoning"] = {"text": reasoning_text, "grounding": grounding}

        return SkillFinalizeResult(
            text=answer_text,
            tokens=list(self.tokens),
            output=output,
        )

    def on_prefill(self, runtime: "MoondreamRuntime") -> None:
        if not self._reasoning_enabled:
            return None
        self._ensure_token_ids(runtime)
        return None

    def _flush_current_chunk(self) -> None:
        if not self._reasoning_enabled:
            return
        if self._current_chunk_tokens or self._current_chunk_points:
            self._reasoning_chunks.append(
                (
                    list(self._current_chunk_tokens),
                    list(self._current_chunk_points),
                )
            )
        self._current_chunk_tokens.clear()
        self._current_chunk_points.clear()
        self._pending_coord = None

    def _ensure_token_ids(self, runtime: "MoondreamRuntime") -> None:
        if self._answer_id is not None:
            return
        tokenizer_cfg = runtime.config.tokenizer
        self._answer_id = tokenizer_cfg.answer_id
        self._start_ground_id = tokenizer_cfg.start_ground_points_id
        self._end_ground_id = tokenizer_cfg.end_ground_id
