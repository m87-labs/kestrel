"""Qwen3-ASR transcription skill and per-request decode state."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Mapping, Optional, Sequence

from kestrel.runtime.tokens import TextToken
from kestrel.skills import SkillRegistry
from kestrel.skills.base import (
    BuiltRequest,
    DecodeStep,
    PreparedSkillPrompt,
    SkillFinalizeResult,
    SkillSpec,
    SkillState,
)
from tokenizers.decoders import DecodeStream

from kestrel.models.asr.contract import (
    DecodeSettings,
    Segment,
    TranscriptionRequest,
    TranscriptionResult,
)

from .runtime import PreparedQwenAudio
from .tokenizer import Qwen3AsrTokenizer, language_code


@dataclass(frozen=True, slots=True)
class QwenTranscribeContext:
    language: str | None
    timestamps: str


class Qwen3AsrTranscribeSkill(SkillSpec):
    def __init__(self) -> None:
        super().__init__(name="transcribe")

    def orchestrator(self) -> Any:
        from .longform import Qwen3AsrLongFormOrchestrator

        return Qwen3AsrLongFormOrchestrator()

    def build_request(
        self,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        if image is not None:
            raise ValueError("transcribe does not accept an image")
        request = TranscriptionRequest.from_prompt(prompt)
        if request.task != "transcribe":
            raise ValueError("Qwen3-ASR does not support speech translation")
        if request.timestamps == "character":
            raise ValueError("Qwen3-ASR does not support character timestamps")
        decode = DecodeSettings.from_mapping(settings)
        return BuiltRequest(
            request_context=request,
            max_new_tokens=decode.max_tokens,
            temperature=decode.temperature,
            top_p=decode.top_p,
            encoder_input=request,
        )

    def prompt_text(self, request_context: object) -> str:
        return (
            "transcribe" if isinstance(request_context, QwenTranscribeContext) else ""
        )

    def build_prompt_tokens(
        self,
        runtime: Any,
        request_context: object,
    ) -> Sequence[TextToken]:
        if not isinstance(request_context, TranscriptionRequest):
            raise TypeError("Qwen3-ASR transcribe requires a TranscriptionRequest")
        tokenizer = _runtime_tokenizer(runtime)
        # The runtime expands this marker after async audio preprocessing reveals
        # the exact number of audio embeddings in the decoder prefix.
        return (TextToken(token_id=tokenizer.audio_token_id),)

    def prepare_prompt(
        self,
        runtime: Any,
        request_context: object,
        max_new_tokens: int,
    ) -> PreparedSkillPrompt:
        if not isinstance(request_context, TranscriptionRequest):
            raise TypeError("Qwen3-ASR transcribe requires a TranscriptionRequest")
        return PreparedSkillPrompt(
            request_context=QwenTranscribeContext(
                request_context.language,
                request_context.timestamps,
            ),
            tokens=tuple(self.build_prompt_tokens(runtime, request_context)),
            max_new_tokens=max_new_tokens,
        )

    def create_state(
        self,
        runtime: Any,
        request: Any,
        request_context: object,
    ) -> "Qwen3AsrTranscribeState":
        if not isinstance(request_context, QwenTranscribeContext):
            raise TypeError("Qwen3-ASR admission requires transcribe context")
        prepared = getattr(request, "encoder_input", None)
        if not isinstance(prepared, PreparedQwenAudio):
            raise TypeError("Qwen3-ASR admission must provide prepared audio")
        return Qwen3AsrTranscribeState(
            self,
            request,
            request_context,
            _runtime_tokenizer(runtime),
            prepared,
        )


def _runtime_tokenizer(runtime: Any) -> Qwen3AsrTokenizer:
    tokenizer = getattr(runtime, "tokenizer", None)
    if not isinstance(tokenizer, Qwen3AsrTokenizer):
        raise TypeError("Qwen3-ASR runtime must expose its tokenizer")
    return tokenizer


class Qwen3AsrTranscribeState(SkillState):
    def __init__(
        self,
        spec: SkillSpec,
        request: Any,
        transcribe_context: QwenTranscribeContext,
        tokenizer: Qwen3AsrTokenizer,
        prepared: PreparedQwenAudio,
    ) -> None:
        super().__init__(spec, request)
        self.language = transcribe_context.language
        self.timestamps = transcribe_context.timestamps
        self.tokenizer = tokenizer
        audio = prepared.audio
        self._duration_seconds = audio.duration_seconds
        self._source_duration_seconds = audio.source_duration_seconds
        self._clip_start_seconds = audio.clip_start_seconds
        self._alignment_audio = audio if self.timestamps == "word" else None
        self._token_ids: list[int] = []
        self._stream_decoder = (
            DecodeStream(skip_special_tokens=False)
            if request.stream_callback is not None
            else None
        )
        if self._stream_decoder is not None:
            self._stream_raw = ""
            self._stream_text = ""

    def consume_step(self, runtime: Any, step: DecodeStep) -> None:
        del runtime
        if not isinstance(step.token, TextToken):
            raise TypeError("Qwen3-ASR decode emitted a non-text token")
        self.append_token(step.token)
        token_id = int(step.token.token_id)
        self._token_ids.append(token_id)
        if self._stream_decoder is not None:
            piece = self._stream_decoder.step(self.tokenizer.backend, token_id)
            if piece:
                self._stream_raw += piece

    def _decoded(self) -> tuple[str, str | None]:
        return self.tokenizer.decode_result(
            self._token_ids,
            forced_language=self.language,
        )

    def pop_stream_delta(self, runtime: Any) -> str | None:
        del runtime
        if self._stream_decoder is None:
            return None
        raw = self._stream_raw
        if self.language is None:
            marker = "<asr_text>"
            if marker not in raw:
                return None
            raw = raw.split(marker, 1)[1]
        for terminal in ("<|im_end|>", "<|endoftext|>"):
            raw = raw.split(terminal, 1)[0]
        text = raw.lstrip()
        if len(text) <= len(self._stream_text):
            return None
        delta = text[len(self._stream_text) :]
        self._stream_text = text
        return delta or None

    def finalize(self, runtime: Any, *, reason: str) -> SkillFinalizeResult:
        del reason
        text, language = self._decoded()
        language_value = language_code(language) if language is not None else None
        segments: tuple[Segment, ...] = ()
        if text and self.timestamps == "segment":
            segments = (
                Segment(
                    text,
                    self._clip_start_seconds,
                    self._clip_start_seconds + self._duration_seconds,
                ),
            )
        elif text and self.timestamps == "word":
            if language is None:
                raise RuntimeError(
                    "Qwen3-ASR did not identify a language for word alignment"
                )
            if self._alignment_audio is None:
                raise RuntimeError("Qwen3-ASR word alignment audio is missing")
            words = tuple(runtime.align_words(self._alignment_audio, text, language))
            if words:
                segments = (Segment(text, words[0].start, words[-1].end, words),)

        output = TranscriptionResult(
            text=text,
            language=language_value,
            duration_seconds=self._duration_seconds,
            source_duration_seconds=self._source_duration_seconds,
            clip_start_seconds=self._clip_start_seconds,
            segments=segments,
        ).as_dict()
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens),
            output=output,
        )


@cache
def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([Qwen3AsrTranscribeSkill()])


__all__ = [
    "Qwen3AsrTranscribeSkill",
    "Qwen3AsrTranscribeState",
    "build_skill_registry",
]
