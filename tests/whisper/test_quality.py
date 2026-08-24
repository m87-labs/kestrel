from __future__ import annotations

import pytest

from kestrel.models.whisper.quality import (
    DEFAULT_TEMPERATURES,
    compression_ratio,
    parse_quality_policy,
)


def test_default_quality_policy_uses_fixed_whisper_thresholds() -> None:
    policy = parse_quality_policy(None)

    assert policy.temperatures == DEFAULT_TEMPERATURES
    assert policy.best_of == 5
    assert policy.candidate_count(0.0) == 1
    assert policy.candidate_count(0.2) == 5
    assert policy.needs_fallback(avg_logprob=-1.01, compression_ratio=1.0)
    assert policy.needs_fallback(avg_logprob=-0.5, compression_ratio=2.41)
    assert not policy.needs_fallback(avg_logprob=-1.0, compression_ratio=2.4)
    assert policy.is_silence(avg_logprob=-1.01, no_speech_prob=0.61)
    assert not policy.is_silence(avg_logprob=-1.0, no_speech_prob=0.61)


def test_quality_policy_accepts_explicit_schedule_and_disabled_thresholds() -> None:
    policy = parse_quality_policy(
        {
            "temperature": [0.0, 0.35, 0.8],
            "best_of": 3,
            "compression_ratio_threshold": None,
            "logprob_threshold": None,
            "no_speech_threshold": None,
        }
    )

    assert policy.temperatures == (0.0, 0.35, 0.8)
    assert policy.best_of == 3
    assert not policy.needs_fallback(avg_logprob=-100.0, compression_ratio=100.0)
    assert not policy.is_silence(avg_logprob=-100.0, no_speech_prob=1.0)


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        ({"temperature": []}, "must not be empty"),
        ({"temperature": [0.2, 0.2]}, "strictly increasing"),
        ({"temperature": [-0.1]}, "lie in"),
        ({"temperature": [0.0, float("nan")]}, "lie in"),
        ({"compression_ratio_threshold": -0.1}, "supported range"),
        ({"logprob_threshold": 0.1}, "supported range"),
        ({"no_speech_threshold": 1.1}, "supported range"),
        ({"best_of": 0}, "lie in"),
        ({"best_of": 9}, "lie in"),
        ({"temperature": 0.0, "best_of": 2}, "requires sampling"),
    ),
)
def test_quality_policy_rejects_invalid_values(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_quality_policy(settings)


def test_quality_policy_rejects_non_mapping_settings() -> None:
    with pytest.raises(TypeError, match="mapping"):
        parse_quality_policy([])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="positive integer"):
        parse_quality_policy({"best_of": 2.5})


def test_compression_ratio_is_utf8_and_empty_safe() -> None:
    assert compression_ratio("") == 0.0
    assert compression_ratio("hello hello hello") > 0.0
