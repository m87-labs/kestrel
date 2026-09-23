"""Qwen3-TTS synthesis skill and per-request audio state."""

from __future__ import annotations

from functools import cache
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from kestrel.audio import SpeechOnsetTrimmer
from kestrel.runtime.tokens import TextToken
from kestrel.skills import SkillRegistry
from kestrel.skills.base import (
    BuiltRequest,
    DecodeStep,
    SkillFinalizeResult,
    SkillSpec,
    SkillState,
    parse_settings,
)

from .config import DEFAULT_MAX_TOKENS, SAMPLE_RATE, SAMPLES_PER_FRAME
from .contract import DEFAULT_SAMPLING, CustomVoiceRequest, Qwen3TTSSampling


_PROMPT_MARKER = TextToken(token_id=0)
_PROMPT_KEYS = frozenset({"text", "voice", "language", "instructions", "stream"})
_SETTING_KEYS = frozenset(
    {
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
        "max_tokens",
    }
)


class Qwen3TTSSynthesizeSkill(SkillSpec):
    def __init__(self) -> None:
        super().__init__(name="synthesize")

    def build_request(
        self,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        if image is not None:
            raise ValueError("synthesize does not accept an image")
        if "speaker" in prompt:
            raise ValueError("use voice, not speaker, for speech synthesis")
        unknown_prompt = set(prompt) - _PROMPT_KEYS
        if unknown_prompt:
            raise ValueError(f"unsupported synthesize inputs: {sorted(unknown_prompt)}")
        unknown_settings = set(settings or ()) - _SETTING_KEYS
        if unknown_settings:
            raise ValueError(
                f"unsupported synthesize settings: {sorted(unknown_settings)}"
            )
        sampling = parse_settings(
            settings,
            temperature=DEFAULT_SAMPLING.temperature,
            top_p=DEFAULT_SAMPLING.top_p,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        if sampling.max_tokens < 2:
            raise ValueError("Qwen3-TTS settings.max_tokens must be at least 2")
        values: Mapping[str, Any] = settings or {}
        model_sampling = Qwen3TTSSampling(
            do_sample=values.get("do_sample", DEFAULT_SAMPLING.do_sample),
            temperature=sampling.temperature,
            top_k=values.get("top_k", DEFAULT_SAMPLING.top_k),
            top_p=sampling.top_p,
            repetition_penalty=values.get(
                "repetition_penalty", DEFAULT_SAMPLING.repetition_penalty
            ),
            subtalker_dosample=values.get(
                "subtalker_dosample", DEFAULT_SAMPLING.subtalker_dosample
            ),
            subtalker_temperature=values.get(
                "subtalker_temperature", DEFAULT_SAMPLING.subtalker_temperature
            ),
            subtalker_top_k=values.get(
                "subtalker_top_k", DEFAULT_SAMPLING.subtalker_top_k
            ),
            subtalker_top_p=values.get(
                "subtalker_top_p", DEFAULT_SAMPLING.subtalker_top_p
            ),
        )
        request = CustomVoiceRequest(
            text=prompt.get("text"),  # type: ignore[arg-type]
            voice=prompt.get("voice", "ryan"),  # type: ignore[arg-type]
            language=prompt.get("language", "auto"),  # type: ignore[arg-type]
            instructions=prompt.get("instructions"),  # type: ignore[arg-type]
            stream=prompt.get("stream", False),  # type: ignore[arg-type]
            sampling=model_sampling,
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=sampling.max_tokens,
            temperature=model_sampling.talker_temperature,
            top_p=sampling.top_p,
            encoder_input=request,
        )

    def prompt_text(self, request_context: object) -> str:
        return (
            request_context.text
            if isinstance(request_context, CustomVoiceRequest)
            else ""
        )

    def build_prompt_tokens(
        self,
        runtime: Any,
        request_context: object,
    ) -> Sequence[TextToken]:
        del runtime
        if not isinstance(request_context, CustomVoiceRequest):
            raise TypeError("Qwen3-TTS synthesis requires a CustomVoiceRequest")
        return (_PROMPT_MARKER,)

    def create_state(
        self,
        runtime: Any,
        request: Any,
        request_context: object,
    ) -> "Qwen3TTSSynthesisState":
        del runtime
        if not isinstance(request_context, CustomVoiceRequest):
            raise TypeError("Qwen3-TTS admission requires a CustomVoiceRequest")
        from .runtime import PreparedSynthesis

        prepared = getattr(request, "encoder_input", None)
        if not isinstance(prepared, PreparedSynthesis):
            raise TypeError("Qwen3-TTS admission must provide prepared synthesis input")
        return Qwen3TTSSynthesisState(self, request, request_context, prepared)


class Qwen3TTSSynthesisState(SkillState):
    def __init__(
        self,
        spec: SkillSpec,
        request: Any,
        synthesis: CustomVoiceRequest,
        prepared: object,
    ) -> None:
        super().__init__(spec, request)
        self.synthesis = synthesis
        self.prepared = prepared
        self._pcm: list[np.ndarray] = []
        self._streamed_chunks = 0
        self._playback_started_at: float | None = None
        self._streamed_samples = 0
        self._onset = SpeechOnsetTrimmer(SAMPLE_RATE)
        self._suppress_fixed_bootstrap_audio = (
            not synthesis.instructions and synthesis.sampling == DEFAULT_SAMPLING
        )

    @property
    def suppresses_fixed_bootstrap_audio(self) -> bool:
        return self._suppress_fixed_bootstrap_audio

    @property
    def has_audio(self) -> bool:
        return bool(self._pcm)

    def consume_step(self, runtime: Any, step: DecodeStep) -> None:
        if not isinstance(step.token, TextToken):
            raise TypeError("Qwen3-TTS decode emitted a non-text token")
        self.append_token(step.token)
        if (
            step.token.token_id in runtime.eos_token_ids
            or self.token_count >= self.request.max_new_tokens
        ):
            self._finish_onset()

    def append_pcm(self, chunk: np.ndarray) -> None:
        if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32:
            raise TypeError("Qwen3-TTS PCM chunks must be float32 numpy arrays")
        chunk = np.array(
            chunk, dtype=np.float32, order="C", copy=True
        ).reshape(-1)
        audible = self._onset.push(chunk)
        # Tried holding one frame before first publication at H100 RPS10:
        # 1 / 26.8 ms and 49.81 xRT versus 1 / 16.0 ms and 50.38 xRT.
        # Keeping eager publication.
        if audible.size:
            self._pcm.append(audible)

    def pop_stream_output(self, runtime: Any) -> Mapping[str, object] | None:
        del runtime
        if not self.synthesis.stream or self._streamed_chunks == len(self._pcm):
            return None
        pending = self._pcm[self._streamed_chunks :]
        chunk = pending[0] if len(pending) == 1 else np.concatenate(pending)
        self._streamed_chunks = len(self._pcm)
        if self._playback_started_at is None:
            self._playback_started_at = time.perf_counter()
        self._streamed_samples += int(chunk.size)
        return {"audio": chunk, "sample_rate": SAMPLE_RATE}

    def next_output_deadline(self, runtime: Any) -> float | None:
        del runtime
        if not self.synthesis.stream:
            return None
        if self._playback_started_at is None:
            return float(self.request.submitted_at)
        # Reserve a short refill's worth of playback (two codec frames).
        # One frame left 7 H100 / 2 B200 underruns at 0.6B, 6 RPS, seed 0;
        # those stalls occurred 200–320ms after onset, while refilling.
        return self._playback_started_at + (
            self._streamed_samples - 2 * SAMPLES_PER_FRAME
        ) / SAMPLE_RATE

    def _finish_onset(self) -> None:
        pending = self._onset.finish()
        if pending.size:
            self._pcm.append(pending)

    def finalize(self, runtime: Any, *, reason: str) -> SkillFinalizeResult:
        del reason
        runtime._release_synthesis_audio(self)
        self._finish_onset()
        if not self._pcm:
            waveform = np.empty(0, dtype=np.float32)
        elif len(self._pcm) == 1:
            waveform = self._pcm[0]
        else:
            waveform = np.concatenate(self._pcm)
        return SkillFinalizeResult(
            text="",
            tokens=list(self.tokens),
            output={
                "audio": waveform,
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": waveform.size / SAMPLE_RATE,
                "voice": self.synthesis.voice,
                "language": self.synthesis.language,
            },
        )


@cache
def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([Qwen3TTSSynthesizeSkill()])


__all__ = [
    "Qwen3TTSSynthesisState",
    "Qwen3TTSSynthesizeSkill",
    "build_skill_registry",
]
