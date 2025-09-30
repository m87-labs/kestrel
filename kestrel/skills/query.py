"""Query skill leveraging the existing text generation flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import TextToken

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class QuerySettings:
    """Sampling parameters supplied with a query invocation."""

    temperature: float
    top_p: float


@dataclass(slots=True)
class QueryRequest:
    """Validated query payload aligned with the fal_inference API."""

    question: str
    image: Optional[pyvips.Image]
    reasoning: bool
    stream: bool
    settings: QuerySettings


@dataclass(slots=True)
class QueryInvocation:
    """Query request plus global defaults resolved for scheduling."""

    request: QueryRequest
    max_new_tokens: int


@dataclass(slots=True)
class QueryDefaults:
    """Default values applied while parsing external query payloads."""

    max_new_tokens: int
    temperature: float
    top_p: float


class QuerySkill(SkillSpec):
    """Default skill emitting plain text answers."""

    def __init__(self) -> None:
        super().__init__(name="query")

    # ------------------------------------------------------------------

    def parse_http_payload(
        self,
        payload: Mapping[str, Any],
        *,
        image_decoder: Callable[[str], pyvips.Image],
        defaults: QueryDefaults,
    ) -> QueryInvocation:
        """Validate an HTTP JSON payload and materialize a `QueryInvocation`.

        Parameters
        ----------
        payload:
            Incoming JSON body from the HTTP server.
        image_decoder:
            Callable that resolves base64 image payloads into `pyvips.Image`.
        defaults:
            Default sampling/max-token values configured for the server.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("Request body must be a JSON object")

        raw_question = payload.get("question")
        if not isinstance(raw_question, str):
            raise ValueError("Field 'question' must be a string")
        question = raw_question.strip()
        if not question:
            raise ValueError("Field 'question' must be a non-empty string")

        raw_reasoning = payload.get("reasoning", False)
        if not isinstance(raw_reasoning, bool):
            raise ValueError("Field 'reasoning' must be a boolean if provided")
        if raw_reasoning:
            raise ValueError("Reasoning mode is not supported for this endpoint")

        raw_stream = payload.get("stream", False)
        if not isinstance(raw_stream, bool):
            raise ValueError("Field 'stream' must be a boolean if provided")
        if raw_stream:
            raise ValueError("Streaming is not supported for this endpoint")

        settings_payload = payload.get("settings")
        if settings_payload is None:
            temperature = defaults.temperature
            top_p = defaults.top_p
        elif isinstance(settings_payload, Mapping):
            if "temperature" in settings_payload:
                temperature = _parse_float(
                    settings_payload["temperature"],
                    field="settings.temperature",
                    minimum=0.0,
                )
            else:
                temperature = defaults.temperature
            if "top_p" in settings_payload:
                top_p = _parse_float(
                    settings_payload["top_p"],
                    field="settings.top_p",
                    minimum_exclusive=0.0,
                    maximum=1.0,
                )
            else:
                top_p = defaults.top_p
        else:
            raise ValueError("Field 'settings' must be an object if provided")

        max_new_tokens = _parse_int(
            payload.get("max_new_tokens", defaults.max_new_tokens),
            field="max_new_tokens",
            minimum=1,
        )

        image_data = payload.get("image_url")
        if image_data is None:
            image: Optional[pyvips.Image] = None
        elif isinstance(image_data, str):
            try:
                image = image_decoder(image_data)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        else:
            raise ValueError("Field 'image_url' must be a string if provided")

        request = QueryRequest(
            question=question,
            image=image,
            reasoning=False,
            stream=False,
            settings=QuerySettings(temperature=temperature, top_p=top_p),
        )
        return QueryInvocation(request=request, max_new_tokens=max_new_tokens)

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
    ) -> Tensor:
        template = runtime.config.tokenizer.templates["query"]
        prefix: Sequence[int] = template["prefix"]
        suffix: Sequence[int] = template["suffix"]
        encoded = runtime.tokenizer.encode(prompt).ids if prompt else []
        ids = [*prefix, *encoded, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        *,
        context: Optional[object] = None,
    ) -> "QuerySkillState":
        if context is None:
            context = QueryRequest(
                question=request.prompt,
                image=request.image,
                reasoning=False,
                stream=False,
                settings=QuerySettings(
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
            )
        if not isinstance(context, QueryRequest):
            raise ValueError("QuerySkill.create_state requires a QueryRequest context")
        return QuerySkillState(self, request, context)


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

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        self.append_token(step.token)
        return None

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        text_tokens = [
            token.token_id for token in self.tokens if isinstance(token, TextToken)
        ]
        text = runtime.tokenizer.decode(text_tokens) if text_tokens else ""
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens),
            extras={"answer": text},
        )


def _parse_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Field '{field}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"Field '{field}' must be >= {minimum}, received {value}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"Field '{field}' must be <= {maximum}, received {value}"
        )
    return value


def _parse_float(
    value: Any,
    *,
    field: str,
    minimum: Optional[float] = None,
    minimum_exclusive: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        raise ValueError(f"Field '{field}' must be a number")
    if minimum is not None and out < minimum:
        raise ValueError(
            f"Field '{field}' must be >= {minimum}, received {out}"
        )
    if minimum_exclusive is not None and out <= minimum_exclusive:
        raise ValueError(
            f"Field '{field}' must be > {minimum_exclusive}, received {out}"
        )
    if maximum is not None and out > maximum:
        raise ValueError(
            f"Field '{field}' must be <= {maximum}, received {out}"
        )
    return out
