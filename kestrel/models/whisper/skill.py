"""Whisper's ``transcribe`` capability and per-request decode state."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache
from numbers import Integral, Real
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
    parse_settings,
)

from .alignment import AlignedWord, TranscriptScores
from .audio import PreparedAudio, validate_audio_source
from .longform import WhisperLongFormOrchestrator
from .quality import compression_ratio
from .timestamps import (
    TimestampMaskPlan,
    apply_timestamp_rules_cpu,
    parse_timestamp_segments,
    timestamp_mask_plan,
)
from .tokenizer import (
    WhisperTokenizer,
    normalize_language_code,
)


DEFAULT_TRANSCRIPT_TOKENS = 444
MAX_INITIAL_PROMPT_TOKENS = 128
_PROMPT_FIELDS = frozenset(
    {
        "audio",
        "clip_end_seconds",
        "clip_start_seconds",
        "sample_rate",
        "language",
        "initial_prompt",
        "condition_on_previous_text",
        "_previous_text_token_ids",
        "stream",
        "task",
        "timestamps",
    }
)
_SETTING_FIELDS = frozenset(
    {
        "compression_ratio_threshold",
        "best_of",
        "logprob_threshold",
        "max_tokens",
        "no_speech_threshold",
        "temperature",
        "top_p",
    }
)


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    language: str | None
    timestamps: str
    max_transcript_tokens: int
    temperature: float
    task: str = "transcribe"
    initial_prompt: str | None = None
    previous_text_token_ids: tuple[int, ...] = ()
    max_tokens_explicit: bool = False
    forced_prefix_tail: tuple[int, ...] = ()

    @property
    def automatic_control_token_count(self) -> int:
        if self.language is not None:
            return 0
        return 3 if self.timestamps == "none" else 2

    @property
    def full_control_prefix_length(self) -> int:
        return 4 if self.timestamps == "none" else 3


class WhisperTranscribeSkill(SkillSpec):
    """Timestamp-aware transcription with bounded long-file orchestration."""

    def __init__(self) -> None:
        super().__init__(name="transcribe")

    def orchestrator(self) -> WhisperLongFormOrchestrator:
        return WhisperLongFormOrchestrator()

    def build_request(
        self,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        if image is not None:
            raise ValueError("transcribe does not accept an image")
        if not isinstance(prompt, Mapping):
            raise TypeError("transcribe prompt must be a mapping")
        if any(not isinstance(name, str) for name in prompt):
            raise TypeError("transcribe prompt field names must be strings")
        unknown_prompt = sorted(set(prompt) - _PROMPT_FIELDS)
        if unknown_prompt:
            raise ValueError(
                f"Unsupported transcribe option(s): {', '.join(unknown_prompt)}"
            )
        if "audio" not in prompt:
            raise ValueError("audio must be provided for transcription")

        if settings is not None and not isinstance(settings, Mapping):
            raise TypeError("transcribe settings must be a mapping or None")
        if settings is not None and any(not isinstance(name, str) for name in settings):
            raise TypeError("transcribe setting names must be strings")
        unknown_settings = sorted(set(settings or {}) - _SETTING_FIELDS)
        if unknown_settings:
            raise ValueError(
                f"Unsupported transcribe setting(s): {', '.join(unknown_settings)}"
            )
        if settings is not None and "max_tokens" in settings:
            max_tokens = settings["max_tokens"]
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral):
                raise TypeError("settings.max_tokens must be a positive integer")
        for name in ("temperature", "top_p"):
            if settings is None or name not in settings:
                continue
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"settings.{name} must be a real number")
        decode_settings = {
            name: settings[name]
            for name in ("max_tokens", "temperature", "top_p")
            if settings is not None and name in settings
        }
        resolved = parse_settings(
            decode_settings,
            temperature=0.0,
            top_p=1.0,
            max_tokens=DEFAULT_TRANSCRIPT_TOKENS,
        )
        if not 0.0 <= resolved.temperature <= 1.0:
            raise ValueError("settings.temperature must lie in [0, 1]")

        timestamps = prompt.get("timestamps", "segment")
        if not isinstance(timestamps, str) or timestamps not in {
            "none",
            "segment",
            "word",
        }:
            raise ValueError("timestamps must be 'none', 'segment', or 'word'")
        language_value = prompt.get("language")
        if language_value is None:
            language = None
        else:
            language = normalize_language_code(language_value)  # type: ignore[arg-type]
        task = prompt.get("task", "transcribe")
        if not isinstance(task, str) or task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        if not isinstance(prompt.get("stream", False), bool):
            raise TypeError("stream must be a boolean")
        if not isinstance(prompt.get("condition_on_previous_text", True), bool):
            raise TypeError("condition_on_previous_text must be a boolean")
        initial_prompt_value = prompt.get("initial_prompt")
        if initial_prompt_value is None:
            initial_prompt = None
        elif not isinstance(initial_prompt_value, str):
            raise TypeError("initial_prompt must be a non-empty string or None")
        elif not initial_prompt_value.strip():
            raise ValueError("initial_prompt must be a non-empty string or None")
        else:
            initial_prompt = initial_prompt_value
        previous_text_value = prompt.get("_previous_text_token_ids", ())
        if not isinstance(previous_text_value, tuple) or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < 50257
            for token_id in previous_text_value
        ):
            raise ValueError("previous text context contains invalid token IDs")
        if len(previous_text_value) > MAX_INITIAL_PROMPT_TOKENS:
            raise ValueError(
                f"previous text context exceeds {MAX_INITIAL_PROMPT_TOKENS} tokens"
            )

        source = validate_audio_source(
            prompt["audio"],
            sample_rate=prompt.get("sample_rate"),
            clip_start_seconds=prompt.get("clip_start_seconds", 0.0),
            clip_end_seconds=prompt.get("clip_end_seconds"),
        )
        request = TranscribeRequest(
            language=language,
            task=task,
            timestamps=timestamps,
            max_transcript_tokens=resolved.max_tokens,
            temperature=resolved.temperature,
            initial_prompt=initial_prompt,
            previous_text_token_ids=previous_text_value,
            max_tokens_explicit=settings is not None and "max_tokens" in settings,
        )
        max_transcript_tokens = 448 - request.full_control_prefix_length
        if request.max_transcript_tokens > max_transcript_tokens:
            raise ValueError(
                f"settings.max_tokens={request.max_transcript_tokens} exceeds the "
                f"Whisper target-position budget {max_transcript_tokens} for "
                f"timestamps={timestamps!r}"
            )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=(
                request.max_transcript_tokens + request.automatic_control_token_count
            ),
            temperature=resolved.temperature,
            top_p=resolved.top_p,
            encoder_input=source,
            capture_logprobs=True,
        )

    def prompt_text(self, request_context: object) -> str:
        return "transcribe" if isinstance(request_context, TranscribeRequest) else ""

    def build_prompt_tokens(
        self,
        runtime: Any,
        request_context: object,
    ) -> Sequence[TextToken]:
        if not isinstance(request_context, TranscribeRequest):
            raise ValueError(
                "WhisperTranscribeSkill.build_prompt_tokens requires a TranscribeRequest"
            )
        tokenizer = _runtime_tokenizer(runtime)
        return [
            TextToken(token_id=token_id)
            for token_id in tokenizer.controls.prompt_ids(
                request_context.language,
                timestamps=request_context.timestamps,
                task=request_context.task,
            )
        ]

    def prepare_prompt(
        self,
        runtime: Any,
        request_context: object,
        max_new_tokens: int,
    ) -> PreparedSkillPrompt:
        if not isinstance(request_context, TranscribeRequest):
            raise ValueError(
                "WhisperTranscribeSkill.prepare_prompt requires a TranscribeRequest"
            )
        tokenizer = _runtime_tokenizer(runtime)
        if (
            request_context.initial_prompt is None
            and not request_context.previous_text_token_ids
        ):
            return super().prepare_prompt(runtime, request_context, max_new_tokens)

        initial_ids = (
            ()
            if request_context.initial_prompt is None
            else tokenizer.encode_text(request_context.initial_prompt)
        )
        if len(initial_ids) > MAX_INITIAL_PROMPT_TOKENS:
            raise ValueError(
                f"initial_prompt exceeds the {MAX_INITIAL_PROMPT_TOKENS}-token limit"
            )
        previous_capacity = MAX_INITIAL_PROMPT_TOKENS - len(initial_ids)
        previous_ids = (
            request_context.previous_text_token_ids[-previous_capacity:]
            if previous_capacity
            else ()
        )
        text_ids = (*initial_ids, *previous_ids)
        full_prefix = (
            tokenizer.controls.prev_sot_id,
            *text_ids,
            *tokenizer.controls.prompt_ids(
                request_context.language,
                timestamps=request_context.timestamps,
                task=request_context.task,
            ),
        )
        prompt_tokens = full_prefix[:4]
        forced_tail = full_prefix[4:]
        available_transcript_tokens = (
            tokenizer.controls.max_target_positions
            - len(full_prefix)
            - request_context.automatic_control_token_count
        )
        if available_transcript_tokens <= 0:
            raise ValueError(
                "initial_prompt leaves no target positions for transcription"
            )
        if (
            request_context.max_tokens_explicit
            and request_context.max_transcript_tokens > available_transcript_tokens
        ):
            raise ValueError(
                f"settings.max_tokens={request_context.max_transcript_tokens} exceeds "
                f"the {available_transcript_tokens}-token transcript budget after initial_prompt"
            )
        transcript_tokens = min(
            request_context.max_transcript_tokens,
            available_transcript_tokens,
        )
        resolved = replace(
            request_context,
            max_transcript_tokens=transcript_tokens,
            forced_prefix_tail=tuple(forced_tail),
        )
        return PreparedSkillPrompt(
            request_context=resolved,
            tokens=tuple(TextToken(token_id=token_id) for token_id in prompt_tokens),
            max_new_tokens=(
                len(forced_tail)
                + resolved.automatic_control_token_count
                + transcript_tokens
            ),
        )

    def create_state(
        self,
        runtime: Any,
        request: Any,
        request_context: object,
    ) -> "WhisperTranscribeState":
        if not isinstance(request_context, TranscribeRequest):
            raise ValueError(
                "WhisperTranscribeSkill.create_state requires a TranscribeRequest"
            )
        prepared = getattr(request, "encoder_input", None)
        if not isinstance(prepared, PreparedAudio):
            raise TypeError(
                "Whisper admission must replace AudioSource with PreparedAudio"
            )
        return WhisperTranscribeState(
            self,
            request,
            request_context,
            _runtime_tokenizer(runtime),
            prepared,
        )


def _runtime_tokenizer(runtime: Any) -> WhisperTokenizer:
    tokenizer = getattr(runtime, "tokenizer", None)
    if not isinstance(tokenizer, WhisperTokenizer):
        raise TypeError(
            "Whisper runtime must expose a WhisperTokenizer as runtime.tokenizer"
        )
    return tokenizer


class WhisperTranscribeState(SkillState):
    """Host-side control grammar and transcript accumulator for one request."""

    # Language and forced-control phases make the mask position-dependent.
    mask_is_stateful = True

    def __init__(
        self,
        spec: SkillSpec,
        request: Any,
        transcribe_request: TranscribeRequest,
        tokenizer: WhisperTokenizer,
        prepared_audio: PreparedAudio,
    ) -> None:
        super().__init__(spec, request)
        self.transcribe_request = transcribe_request
        self.tokenizer = tokenizer
        self.controls = tokenizer.controls
        self.language = transcribe_request.language
        self.language_probability: float | None = None
        self._post_prompt_phase = (
            "transcript" if self.language is not None else "language"
        )
        self._phase = (
            "prompt_prefix"
            if transcribe_request.forced_prefix_tail
            else self._post_prompt_phase
        )
        self._forced_prefix_tail = transcribe_request.forced_prefix_tail
        self._forced_prefix_index = 0
        derived_max_new_tokens = (
            len(transcribe_request.forced_prefix_tail)
            + transcribe_request.automatic_control_token_count
            + transcribe_request.max_transcript_tokens
        )
        request_max_new_tokens = getattr(
            request, "max_new_tokens", derived_max_new_tokens
        )
        if (
            isinstance(request_max_new_tokens, bool)
            or not isinstance(request_max_new_tokens, int)
            or request_max_new_tokens <= 0
        ):
            raise ValueError(
                "Whisper request max_new_tokens must be a positive integer"
            )
        self._max_new_tokens = request_max_new_tokens
        self._transcript_token_ids: list[int] = []
        self._transcript_logprobs: list[float] = []
        self._stream_text_offset = 0
        self._aligned_words: tuple[AlignedWord, ...] | None = None
        self._transcript_scores: TranscriptScores | None = None
        self._duration_seconds = prepared_audio.duration_seconds
        self._clip_start_seconds = prepared_audio.clip_start_seconds
        self._source_duration_seconds = (
            prepared_audio.duration_seconds
            if prepared_audio.source_duration_seconds is None
            else prepared_audio.source_duration_seconds
        )

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def transcript_token_ids(self) -> tuple[int, ...]:
        return tuple(self._transcript_token_ids)

    def allowed_token_ids(self, runtime: Any) -> Sequence[int] | None:
        if self._phase == "prompt_prefix":
            return (self._forced_prefix_tail[self._forced_prefix_index],)
        if self._phase == "language":
            return self.controls.language_token_ids
        if self._phase == "task_control":
            return (self.controls.task_id(self.transcribe_request.task),)
        if self._phase == "no_timestamps_control":
            return (self.controls.no_timestamps_id,)
        return None

    def suppressed_token_ids(self, runtime: Any) -> Sequence[int] | None:
        suppressed = set(self.controls.suppress_tokens)
        if self._phase == "transcript" and not self._transcript_token_ids:
            suppressed.update(self.controls.begin_suppress_tokens)
        allowed = self.allowed_token_ids(runtime)
        if allowed is not None:
            # The checkpoint suppresses task controls during normal transcript
            # generation. A control explicitly forced by our phase machine must
            # win over that persistent table.
            suppressed.difference_update(allowed)
        return tuple(sorted(suppressed)) if suppressed else None

    def timestamp_plan(self) -> TimestampMaskPlan | None:
        if self._phase != "transcript" or self.transcribe_request.timestamps not in {
            "segment",
            "word",
        }:
            return None
        return timestamp_mask_plan(self._transcript_token_ids, self.controls)

    def process_logits_cpu(self, scores: Any) -> Any:
        """CPU-only oracle; serving integration must use the fused batched hook."""

        if (
            self.transcribe_request.timestamps not in {"segment", "word"}
            or self._phase != "transcript"
        ):
            return scores.clone()
        return apply_timestamp_rules_cpu(
            scores,
            self._transcript_token_ids,
            self.controls,
        )

    def consume_step(self, runtime: Any, step: DecodeStep) -> None:
        token = step.token
        if not isinstance(token, TextToken):
            raise TypeError("Whisper decode emitted a non-text token")
        token_id = int(token.token_id)
        allowed = self.allowed_token_ids(runtime)
        if allowed is not None and token_id not in allowed:
            raise RuntimeError(
                f"Whisper phase {self._phase!r} received invalid token {token_id}"
            )
        logprob = step.logprob
        if self._phase == "transcript" and (
            logprob is None or not math.isfinite(logprob) or logprob > 1e-6
        ):
            raise RuntimeError(
                "Whisper transcript decoding requires a finite selected-token logprob"
            )
        self.append_token(token)

        if self._phase == "prompt_prefix":
            self._forced_prefix_index += 1
            if self._forced_prefix_index == len(self._forced_prefix_tail):
                self._phase = self._post_prompt_phase
        elif self._phase == "language":
            if logprob is None or not math.isfinite(logprob) or logprob > 1e-6:
                raise RuntimeError(
                    "Whisper language detection requires a finite selected-token "
                    "logprob"
                )
            self.language = self.controls.language_code(token_id)
            self.language_probability = min(1.0, math.exp(logprob))
            self._phase = "task_control"
        elif self._phase == "task_control":
            self._phase = (
                "no_timestamps_control"
                if self.transcribe_request.timestamps == "none"
                else "transcript"
            )
        elif self._phase == "no_timestamps_control":
            self._phase = "transcript"
        else:
            if logprob is None:  # pragma: no cover - validated before mutation above
                raise AssertionError("validated transcript logprob is missing")
            self._transcript_token_ids.append(token_id)
            self._transcript_logprobs.append(logprob)
            terminal = token_id == self.controls.eos_id or (
                self.token_count >= self._max_new_tokens
            )
            if terminal:
                if self.language is None:
                    raise RuntimeError(
                        "Whisper decoder analysis requires a resolved language"
                    )
                analyze = getattr(runtime, "analyze_transcript", None)
                if not callable(analyze):
                    raise RuntimeError(
                        "Whisper runtime does not implement transcript analysis"
                    )
                prompt_tokens = getattr(self.request, "prompt_tokens", None)
                if prompt_tokens is None:
                    prompt_token_ids = self.controls.prompt_ids(
                        self.transcribe_request.language,
                        timestamps=self.transcribe_request.timestamps,
                        task=self.transcribe_request.task,
                    )
                else:
                    if any(not isinstance(value, TextToken) for value in prompt_tokens):
                        raise TypeError("Whisper prompt contains a non-text token")
                    prompt_token_ids = tuple(
                        int(value.token_id) for value in prompt_tokens
                    )
                generated_token_ids = tuple(
                    int(value.token_id) for value in self.tokens
                )
                transcript_count = len(self._transcript_token_ids)
                prefix_token_ids = (
                    *prompt_token_ids,
                    *generated_token_ids[:-transcript_count],
                )
                analysis = analyze(
                    batch_idx=int(self.request.lifecycle.state.batch_idx),
                    language=self.language,
                    task=self.transcribe_request.task,
                    prefix_token_ids=prefix_token_ids,
                    text_token_ids=tuple(
                        value
                        for value in self._transcript_token_ids
                        if 0 <= value < self.controls.eos_id
                    ),
                    avg_logprob=(
                        sum(self._transcript_logprobs) / len(self._transcript_logprobs)
                    ),
                    duration_seconds=self._duration_seconds,
                    include_words=self.transcribe_request.timestamps == "word",
                )
                self._aligned_words = analysis.words
                self._transcript_scores = analysis.scores

    def stop_token_ids(self, runtime: Any) -> Sequence[int]:
        return (self.controls.eos_id,)

    def pop_stream_delta(self, runtime: Any) -> str | None:
        """Return newly decoded transcript text, excluding Whisper controls."""

        text = self.tokenizer.decode_text(self._transcript_token_ids).lstrip()
        if len(text) <= self._stream_text_offset:
            return None
        chunk = text[self._stream_text_offset :]
        self._stream_text_offset = len(text)
        return chunk or None

    def finalize(self, runtime: Any, *, reason: str) -> SkillFinalizeResult:
        if self.language is None:
            raise RuntimeError(
                "Whisper transcription ended before language identification"
            )
        text = self.tokenizer.decode_text(self._transcript_token_ids).strip()
        if self._transcript_scores is None:
            raise RuntimeError("Whisper transcription ended without decoder analysis")
        avg_logprob = self._transcript_scores.avg_logprob
        no_speech_prob = self._transcript_scores.no_speech_prob
        ratio = compression_ratio(text)
        segments = []
        if self.transcribe_request.timestamps in {"segment", "word"}:
            parsed_segments = parse_timestamp_segments(
                self._transcript_token_ids,
                self.tokenizer,
                self.controls,
                duration_seconds=self._duration_seconds,
            )
            aligned_index = 0
            for segment in parsed_segments:
                value = segment.as_dict()
                value["start"] = float(value["start"]) + self._clip_start_seconds
                value["end"] = float(value["end"]) + self._clip_start_seconds
                value.update(
                    temperature=self.transcribe_request.temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=ratio,
                    no_speech_prob=no_speech_prob,
                )
                if self.transcribe_request.timestamps == "word":
                    if self._aligned_words is None:
                        raise RuntimeError(
                            "Whisper transcription ended without word alignment"
                        )
                    saved_tokens = 0
                    previous_end = float(segment.start)
                    words = []
                    while saved_tokens < len(segment.token_ids):
                        if aligned_index >= len(self._aligned_words):
                            raise RuntimeError(
                                "Whisper word alignment ended before its segment tokens"
                            )
                        word = self._aligned_words[aligned_index]
                        aligned_index += 1
                        next_saved = saved_tokens + len(word.token_ids)
                        if next_saved > len(segment.token_ids):
                            raise RuntimeError(
                                "Whisper word alignment crossed a segment boundary"
                            )
                        start = min(
                            float(segment.end),
                            max(previous_end, float(segment.start), word.start),
                        )
                        end = min(
                            float(segment.end),
                            max(start, word.end),
                        )
                        words.append(
                            {
                                "word": word.word,
                                "start": start + self._clip_start_seconds,
                                "end": end + self._clip_start_seconds,
                                "probability": word.probability,
                            }
                        )
                        previous_end = end
                        saved_tokens = next_saved
                    value["words"] = words
                segments.append(value)
            if (
                self.transcribe_request.timestamps == "word"
                and self._aligned_words is not None
                and aligned_index != len(self._aligned_words)
            ):
                raise RuntimeError(
                    "Whisper word alignment contains tokens outside timestamp segments"
                )
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens[len(self._forced_prefix_tail) :]),
            output={
                "text": text,
                "language": self.language,
                "language_probability": self.language_probability,
                "task": self.transcribe_request.task,
                "duration_seconds": self._duration_seconds,
                "source_duration_seconds": self._source_duration_seconds,
                "clip_start_seconds": self._clip_start_seconds,
                "clip_end_seconds": self._clip_start_seconds + self._duration_seconds,
                "temperature": self.transcribe_request.temperature,
                "avg_logprob": avg_logprob,
                "compression_ratio": ratio,
                "no_speech_prob": no_speech_prob,
                "segments": segments,
            },
        )


@cache
def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([WhisperTranscribeSkill()])


__all__ = [
    "DEFAULT_TRANSCRIPT_TOKENS",
    "MAX_INITIAL_PROMPT_TOKENS",
    "TranscribeRequest",
    "WhisperTranscribeSkill",
    "WhisperTranscribeState",
    "build_skill_registry",
]
