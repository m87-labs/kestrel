from __future__ import annotations

from dataclasses import dataclass

import pytest

from kestrel.models.moondream.runtime import TextToken
from kestrel.scheduler.types import (
    MODEL_PREFILL_TIMING_BOUNDARY,
    GenerationRequest,
    RequestLifecycle,
)
from kestrel.skills import DecodeStep, SkillFinalizeResult, SkillState


@dataclass(frozen=True)
class _SkillSpec:
    name: str = "metrics"


class _SkillState(SkillState):
    def __init__(self, request: GenerationRequest) -> None:
        super().__init__(_SkillSpec(), request)  # type: ignore[arg-type]

    def consume_step(self, runtime: object, step: DecodeStep) -> None:
        self.append_token(step.token)

    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        return SkillFinalizeResult(text="", tokens=list(self.tokens), output={})


def _lifecycle() -> RequestLifecycle:
    request = GenerationRequest(
        request_id=7,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=_SkillSpec(),  # type: ignore[arg-type]
        request_context=object(),
    )
    skill_state = _SkillState(request)
    lifecycle = RequestLifecycle(request=request, skill_state=skill_state)
    request.lifecycle = lifecycle
    return lifecycle


def test_metrics_split_model_prefill_at_first_token_ready() -> None:
    assert MODEL_PREFILL_TIMING_BOUNDARY == "scheduled_to_first_model_token"

    lifecycle = _lifecycle()
    lifecycle.submitted_at = 1.0
    lifecycle.prefill_started_at = 3.0
    lifecycle.prefill_completed_at = 4.0
    lifecycle.first_token_time = 7.0
    lifecycle.completed_at = 11.0

    metrics = lifecycle.build_metrics(decode_tokens=4)

    assert metrics.prefill_time_ms == pytest.approx(4_000.0)
    assert metrics.decode_time_ms == pytest.approx(4_000.0)
    assert metrics.ttft_ms == pytest.approx(6_000.0)


def test_metrics_without_generated_tokens_assigns_work_to_prefill() -> None:
    lifecycle = _lifecycle()
    lifecycle.submitted_at = 1.0
    lifecycle.prefill_started_at = 2.0
    lifecycle.prefill_completed_at = 3.0
    lifecycle.first_token_time = 2.0
    lifecycle.completed_at = 5.0

    metrics = lifecycle.build_metrics(decode_tokens=0)

    assert metrics.prefill_time_ms == pytest.approx(3_000.0)
    assert metrics.decode_time_ms == 0.0
    assert metrics.ttft_ms == pytest.approx(4_000.0)
