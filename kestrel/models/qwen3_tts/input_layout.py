"""Integer Talker input layouts for CustomVoice synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Qwen3TTSConfig
from .contract import EncodedCustomVoiceRequest


@dataclass(frozen=True, slots=True)
class TalkerTokenLayout:
    text_token_ids: tuple[int, ...]
    codec_token_ids: tuple[int, ...]
    codec_embedding_mask: tuple[bool, ...]
    continuation_text_token_ids: tuple[int, ...]
    text_pad_token_id: int

    @property
    def sequence_length(self) -> int:
        return len(self.text_token_ids)


def _conditioning_prefix(
    config: Qwen3TTSConfig,
    request: EncodedCustomVoiceRequest,
) -> tuple[int, tuple[int, ...]]:
    talker = config.talker
    voice = request.request.voice_key
    try:
        speaker_id = talker.spk_id[voice]
    except KeyError as error:
        raise ValueError(f"unsupported Qwen3-TTS voice: {request.request.voice!r}") from error

    language = request.request.language_key
    if language == "auto":
        language_id = None
    elif not language.endswith("_dialect"):
        try:
            language_id = talker.codec_language_id[language]
        except KeyError as error:
            raise ValueError(
                f"unsupported Qwen3-TTS language: {request.request.language!r}"
            ) from error
    else:
        raise ValueError(f"unsupported Qwen3-TTS language: {request.request.language!r}")

    dialect = talker.spk_is_dialect[voice]
    if language in {"auto", "chinese"} and dialect:
        language_id = talker.codec_language_id[str(dialect)]

    if language_id is None:
        prefix = (
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
        )
    else:
        prefix = (
            talker.codec_think_id,
            talker.codec_think_bos_id,
            language_id,
            talker.codec_think_eos_id,
        )
    return speaker_id, prefix


def build_talker_token_layout(
    config: Qwen3TTSConfig,
    request: EncodedCustomVoiceRequest,
) -> TalkerTokenLayout:
    """Build the official full-text or streaming-text Talker layout."""

    text = request.text_token_ids
    if len(text) < 8:
        raise ValueError("Qwen3-TTS wrapped text is shorter than its eight control tokens")
    speaker_id, codec_prefix = _conditioning_prefix(config, request)
    talker = config.talker
    role = text[:3]
    target = text[3:-5]
    if not target:
        raise ValueError("Qwen3-TTS wrapped text has no target tokens")
    instruction = request.instruction_token_ids

    prefix_codec = (*codec_prefix, speaker_id, talker.codec_pad_id)
    text_ids = (
        *instruction,
        *role,
        *((config.tts_pad_token_id,) * (len(codec_prefix) + 1)),
        config.tts_bos_token_id,
    )
    codec_ids = (
        *((talker.codec_pad_id,) * (len(instruction) + len(role))),
        *prefix_codec,
    )
    mask = (
        *((False,) * (len(instruction) + len(role))),
        *((True,) * len(prefix_codec)),
    )
    if request.request.stream:
        text_ids = (*text_ids, target[0])
        codec_ids = (*codec_ids, talker.codec_bos_id)
        mask = (*mask, True)
        continuation = (*target[1:], config.tts_eos_token_id)
    else:
        text_ids = (
            *text_ids,
            *target,
            config.tts_eos_token_id,
            config.tts_pad_token_id,
        )
        codec_ids = (
            *codec_ids,
            *((talker.codec_pad_id,) * (len(target) + 1)),
            talker.codec_bos_id,
        )
        mask = (*mask, *((True,) * (len(target) + 2)))
        continuation = ()
    if not (len(text_ids) == len(codec_ids) == len(mask)):
        raise RuntimeError("Qwen3-TTS Talker prefill fields have mismatched lengths")
    return TalkerTokenLayout(
        text_token_ids=text_ids,
        codec_token_ids=codec_ids,
        codec_embedding_mask=mask,
        continuation_text_token_ids=continuation,
        text_pad_token_id=config.tts_pad_token_id,
    )


__all__ = ["TalkerTokenLayout", "build_talker_token_layout"]
