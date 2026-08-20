"""Whisper timestamp grammar plans and segment parsing.

The shipping scheduler must apply plans with one batched/fused GPU operation.
The CPU applicator here is a correctness oracle and deliberately rejects CUDA
tensors so it cannot accidentally become the serving path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor

from .tokenizer import WhisperControlTokens


@dataclass(frozen=True, slots=True)
class TimestampMaskPlan:
    """Half-open vocabulary ranges to mask before the probability-mass rule."""

    suppress_ranges: tuple[tuple[int, int], ...]
    detect_timestamp_from_logprob: bool = True


def _merged_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    vocab_size: int,
) -> tuple[tuple[int, int], ...]:
    clipped = sorted(
        (max(0, int(start)), min(vocab_size, int(end)))
        for start, end in ranges
        if int(start) < int(end) and int(end) > 0 and int(start) < vocab_size
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def timestamp_mask_plan(
    generated_token_ids: Sequence[int],
    controls: WhisperControlTokens,
    *,
    detect_timestamp_from_logprob: bool = True,
) -> TimestampMaskPlan:
    """Build the history-dependent part of HF's Whisper timestamp processor."""

    sequence = [int(token_id) for token_id in generated_token_ids]
    timestamp_begin = controls.timestamp_begin_id
    ranges: list[tuple[int, int]] = [
        (controls.no_timestamps_id, controls.no_timestamps_id + 1)
    ]

    last_was_timestamp = bool(sequence and sequence[-1] >= timestamp_begin)
    penultimate_was_timestamp = len(sequence) < 2 or sequence[-2] >= timestamp_begin
    if last_was_timestamp:
        if penultimate_was_timestamp:
            ranges.append((timestamp_begin, controls.vocab_size))
        else:
            ranges.append((0, controls.eos_id))

    timestamps = [token_id for token_id in sequence if token_id >= timestamp_begin]
    if timestamps:
        last_timestamp = timestamps[-1]
        first_allowed = (
            last_timestamp
            if last_was_timestamp and not penultimate_was_timestamp
            else last_timestamp + 1
        )
        ranges.append((timestamp_begin, first_allowed))

    if not sequence:
        ranges.append((0, timestamp_begin))
        last_allowed = timestamp_begin + controls.max_initial_timestamp_index
        ranges.append((last_allowed + 1, controls.vocab_size))

    return TimestampMaskPlan(
        suppress_ranges=_merged_ranges(ranges, vocab_size=controls.vocab_size),
        detect_timestamp_from_logprob=bool(detect_timestamp_from_logprob),
    )


def apply_timestamp_rules_cpu(
    scores: Tensor,
    generated_token_ids: Sequence[int],
    controls: WhisperControlTokens,
    *,
    detect_timestamp_from_logprob: bool = True,
) -> Tensor:
    """Apply timestamp rules to one CPU row for tests and reference parity."""

    if scores.device.type != "cpu":
        raise ValueError(
            "apply_timestamp_rules_cpu is an oracle; serving must use the fused batched GPU processor"
        )
    if scores.ndim != 1 or scores.shape[0] != controls.vocab_size:
        raise ValueError(
            f"timestamp scores must have shape ({controls.vocab_size},), got {tuple(scores.shape)}"
        )
    result = scores.clone()
    plan = timestamp_mask_plan(
        generated_token_ids,
        controls,
        detect_timestamp_from_logprob=detect_timestamp_from_logprob,
    )
    for start, end in plan.suppress_ranges:
        result[start:end] = -float("inf")

    if plan.detect_timestamp_from_logprob:
        float_scores = result.float()
        timestamp_mass = torch.logsumexp(
            float_scores[controls.timestamp_begin_id :], dim=0
        )
        best_text = torch.max(float_scores[: controls.timestamp_begin_id])
        # The log-softmax normalizer used by the reference cancels from both
        # sides of this comparison.
        if bool((timestamp_mass > best_text).item()):
            result[: controls.timestamp_begin_id] = -float("inf")
    return result


class TextDecoder(Protocol):
    def decode_text(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    token_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "text": self.text}


def timestamp_seconds(token_id: int, controls: WhisperControlTokens) -> float:
    token = int(token_id)
    if not controls.timestamp_begin_id <= token < controls.vocab_size:
        raise ValueError(f"Token {token_id} is not a timestamp token")
    return round((token - controls.timestamp_begin_id) * 0.02, 2)


def parse_timestamp_segments(
    token_ids: Sequence[int],
    tokenizer: TextDecoder,
    controls: WhisperControlTokens,
    *,
    duration_seconds: float | None = None,
) -> list[TranscriptSegment]:
    """Parse timestamp-paired short-form output into portable segments."""

    segments: list[TranscriptSegment] = []
    start: float | None = None
    text_ids: list[int] = []

    def close(end: float) -> None:
        nonlocal start, text_ids
        if start is None:
            return
        text = tokenizer.decode_text(text_ids).strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(start),
                    end=float(max(start, end)),
                    text=text,
                    token_ids=tuple(text_ids),
                )
            )
        start = None
        text_ids = []

    for raw_token in token_ids:
        token_id = int(raw_token)
        if token_id == controls.eos_id:
            break
        if controls.timestamp_begin_id <= token_id < controls.vocab_size:
            value = timestamp_seconds(token_id, controls)
            if start is None:
                start = value
                text_ids = []
            else:
                close(value)
            continue
        if 0 <= token_id < controls.eos_id and start is not None:
            text_ids.append(token_id)

    if start is not None and text_ids and duration_seconds is not None:
        end = min(30.0, max(start, float(duration_seconds)))
        close(end)
    return segments


__all__ = [
    "TimestampMaskPlan",
    "TranscriptSegment",
    "apply_timestamp_rules_cpu",
    "parse_timestamp_segments",
    "timestamp_mask_plan",
    "timestamp_seconds",
]
