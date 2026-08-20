from __future__ import annotations

import pytest
import torch

from kestrel.models.whisper.timestamps import (
    apply_timestamp_rules_cpu,
    parse_timestamp_segments,
    timestamp_mask_plan,
    timestamp_seconds,
)
from kestrel.models.whisper.tokenizer import WhisperControlTokens


@pytest.mark.parametrize(
    "generated",
    [
        [],
        [50365],
        [50365, 100],
        [50365, 100, 50375],
        [50365, 100, 50375, 50375],
        [100],
    ],
)
def test_timestamp_rules_match_transformers(
    generation_config_dict,
    generated,
) -> None:
    transformers = pytest.importorskip("transformers", minversion="4.56.0")
    from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor

    controls = WhisperControlTokens.from_dict(generation_config_dict)
    generation_config = transformers.GenerationConfig.from_dict(generation_config_dict)
    processor = WhisperTimeStampLogitsProcessor(generation_config, begin_index=0)
    generator = torch.Generator().manual_seed(100 + len(generated))
    scores = torch.randn(controls.vocab_size, generator=generator)

    ours = apply_timestamp_rules_cpu(scores, generated, controls)
    oracle = processor(
        torch.tensor([generated], dtype=torch.long),
        scores[None, :],
    )[0]
    torch.testing.assert_close(ours, oracle, rtol=0.0, atol=0.0)


def test_initial_timestamp_plan_has_one_second_bound() -> None:
    controls = WhisperControlTokens(suppress_tokens=())
    plan = timestamp_mask_plan([], controls)
    assert plan.suppress_ranges == ((0, 50365), (50416, 51866))
    assert timestamp_seconds(50365, controls) == 0.0
    assert timestamp_seconds(50415, controls) == 1.0


def test_timestamp_probability_mass_only_masks_text_when_it_wins() -> None:
    controls = WhisperControlTokens(suppress_tokens=())
    history = [50365, 100]

    text_wins = torch.zeros(controls.vocab_size)
    text_wins[10] = 100.0
    assert torch.isfinite(apply_timestamp_rules_cpu(text_wins, history, controls)[10])

    timestamps_win = torch.zeros(controls.vocab_size)
    processed = apply_timestamp_rules_cpu(timestamps_win, history, controls)
    assert torch.isneginf(processed[: controls.timestamp_begin_id]).all()


class _Decoder:
    def decode_text(self, token_ids):
        return "".join({10: " first", 11: " second"}.get(i, "") for i in token_ids)


def test_segment_parser_handles_adjacent_end_start_timestamps() -> None:
    controls = WhisperControlTokens(suppress_tokens=())
    tokens = [50365, 10, 50375, 50375, 11, 50390, controls.eos_id]
    segments = parse_timestamp_segments(tokens, _Decoder(), controls)
    assert [segment.as_dict() for segment in segments] == [
        {"start": 0.0, "end": 0.2, "text": "first"},
        {"start": 0.2, "end": 0.5, "text": "second"},
    ]


def test_unclosed_segment_uses_audio_duration() -> None:
    controls = WhisperControlTokens(suppress_tokens=())
    segments = parse_timestamp_segments(
        [50370, 10, controls.eos_id],
        _Decoder(),
        controls,
        duration_seconds=0.4,
    )
    assert segments[0].start == 0.1
    assert segments[0].end == 0.4
