from __future__ import annotations

from kestrel.runtime import TextToken
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle


def test_request_metrics_include_end_to_end_request_time() -> None:
    request = GenerationRequest(
        request_id=1,
        prompt="prompt",
        prompt_tokens=[TextToken(1), TextToken(2)],
        max_new_tokens=4,
        skill=object(),
        request_context=object(),
    )
    lifecycle = RequestLifecycle(request=request, skill_state=object())
    request.lifecycle = lifecycle
    lifecycle.submitted_at = 1.0
    lifecycle.prefill_started_at = 2.0
    lifecycle.prefill_completed_at = 3.0
    lifecycle.completed_at = 8.0

    metrics = lifecycle.build_metrics(decode_tokens=2)

    assert metrics.prefill_time_ms == 1000.0
    assert metrics.decode_time_ms == 5000.0
    assert metrics.request_time_ms == 7000.0
