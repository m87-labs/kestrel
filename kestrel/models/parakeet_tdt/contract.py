"""Parakeet-specific transcription validation."""

from __future__ import annotations

from typing import Mapping

from kestrel.models.asr.contract import DecodeSettings, TranscriptionRequest


def parse_request(
    prompt: Mapping[str, object],
    settings: object,
) -> tuple[TranscriptionRequest, DecodeSettings]:
    request = TranscriptionRequest.from_prompt(prompt)
    if request.task != "transcribe":
        raise ValueError("Parakeet TDT does not support speech translation")
    if request.initial_prompt is not None:
        raise ValueError("Parakeet TDT does not support an initial prompt")
    if request.language is not None:
        raise ValueError(
            "Parakeet TDT v3 detects language automatically and does not "
            "support language forcing"
        )
    if not request.condition_on_previous_text:
        raise ValueError("Parakeet TDT does not support cross-window text conditioning")
    decode = DecodeSettings.from_mapping(settings)
    if decode.temperature != 0 or decode.top_p != 1:
        raise ValueError("Parakeet TDT uses deterministic transducer decoding")
    return request, decode


__all__ = ["parse_request"]
