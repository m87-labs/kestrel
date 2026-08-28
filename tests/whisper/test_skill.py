from __future__ import annotations

import math
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models import get_spec
from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep
from kestrel.models.whisper import CHECKPOINT_REVISION, MODEL_NAME, REPO_ID
from kestrel.models.whisper.alignment import (
    AlignedWord,
    TranscriptAnalysis,
    TranscriptScores,
)
from kestrel.models.whisper.audio import AudioSource, PreparedAudio, prepare_audio
from kestrel.models.whisper.skill import (
    WhisperDecodeContext,
    WhisperTranscribeSkill,
)
from kestrel.models.whisper.tokenizer import WhisperControlTokens, WhisperTokenizer


class _Encoding:
    ids = [10]


class _Backend:
    def get_vocab_size(self, *, with_added_tokens):
        return 51866

    def encode(self, text, *, add_special_tokens):
        return _Encoding()

    def decode(self, ids, *, skip_special_tokens):
        return "".join({10: " hello", 11: " world"}.get(i, "") for i in ids)


_HELLO_COMPRESSION_RATIO = len(b"hello") / len(zlib.compress(b"hello"))


def _segment(start, end, *, avg_logprob=-0.25, **values):
    return {
        "start": start,
        "end": end,
        "text": "hello",
        "temperature": 0.0,
        "avg_logprob": avg_logprob,
        "compression_ratio": _HELLO_COMPRESSION_RATIO,
        "no_speech_prob": 0.01,
        **values,
    }


@pytest.fixture
def runtime():
    def analyze_transcript(**_kwargs):
        return TranscriptAnalysis(
            words=(),
            scores=TranscriptScores(-0.25, 0.01),
        )

    return SimpleNamespace(
        tokenizer=WhisperTokenizer(
            _Backend(), WhisperControlTokens(suppress_tokens=())
        ),
        analyze_transcript=analyze_transcript,
    )


@pytest.fixture
def prepared_audio() -> PreparedAudio:
    return PreparedAudio(
        input_features=torch.zeros((128, 3000), dtype=torch.float32),
        duration_seconds=0.1,
        original_num_samples=1600,
        original_sample_rate=16000,
        resampled_num_samples=1600,
    )


def _consume(state, runtime, token_id, position, *, logprob=-0.25):
    state.consume_step(
        runtime,
        DecodeStep(
            token=TextToken(token_id=token_id),
            position=position,
            logprob=logprob,
        ),
    )


def _request(prepared_audio, *, max_new_tokens=444, batch_idx=0):
    return SimpleNamespace(
        encoder_input=prepared_audio,
        max_new_tokens=max_new_tokens,
        lifecycle=SimpleNamespace(state=SimpleNamespace(batch_idx=batch_idx)),
    )


def test_transcribe_state_streams_only_human_readable_text(
    runtime, prepared_audio
) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "segment",
        },
        None,
    )
    state = skill.create_state(
        runtime,
        _request(prepared_audio),
        built.request_context,
    )

    _consume(state, runtime, 50365, 0)
    assert state.pop_stream_delta(runtime) is None
    _consume(state, runtime, 10, 1)
    assert state.pop_stream_delta(runtime) == "hello"
    _consume(state, runtime, 11, 2)
    assert state.pop_stream_delta(runtime) == " world"
    assert state.pop_stream_delta(runtime) is None


def test_model_is_registered_with_its_hugging_face_identity() -> None:
    spec = get_spec(MODEL_NAME)
    assert MODEL_NAME == REPO_ID == "openai/whisper-large-v3-turbo"
    assert spec.name == MODEL_NAME
    assert spec.repo_id == REPO_ID
    assert spec.revision == CHECKPOINT_REVISION
    assert spec.filename is None
    assert spec.skills().names() == ("transcribe",)
    # Construction is covered with explicit fake execution sessions in
    # test_runtime.py. Calling the production factory here would correctly try
    # to resolve the pinned checkpoint and CUDA artifacts, which is outside this
    # registration-only CPU test.
    assert callable(spec.runtime)


def test_explicit_language_builds_complete_control_prefix(runtime) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "segment",
        },
        None,
    )
    assert isinstance(built.request_context, WhisperDecodeContext)
    assert isinstance(built.encoder_input, AudioSource)
    assert built.capture_logprobs is True
    assert not hasattr(built.request_context, "audio")
    assert built.max_new_tokens == 444
    assert [
        token.token_id
        for token in skill.build_prompt_tokens(runtime, built.request_context)
    ] == [50258, 50259, 50360]


def test_translation_uses_translate_control_for_explicit_and_detected_language(
    runtime,
    prepared_audio,
) -> None:
    skill = WhisperTranscribeSkill()
    explicit = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "fr",
            "task": "translate",
        },
        None,
    )
    assert [
        token.token_id
        for token in skill.build_prompt_tokens(runtime, explicit.request_context)
    ] == [50258, 50265, 50359]

    automatic = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "task": "translate",
        },
        None,
    )
    state = skill.create_state(
        runtime,
        _request(prepared_audio),
        automatic.request_context,
    )
    _consume(state, runtime, 50265, 0)
    assert state.phase == "task_control"
    assert state.language_probability == pytest.approx(math.exp(-0.25))
    assert state.allowed_token_ids(runtime) == (50359,)


@pytest.mark.parametrize("logprob", [None, float("nan"), float("inf"), 0.1])
def test_language_detection_rejects_missing_or_invalid_selected_logprob(
    runtime,
    prepared_audio,
    logprob,
) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {"audio": np.zeros(1600, dtype=np.float32), "sample_rate": 16000},
        None,
    )
    state = skill.create_state(runtime, _request(prepared_audio), built.request_context)

    with pytest.raises(RuntimeError, match="language detection"):
        _consume(state, runtime, 50259, 0, logprob=logprob)


def test_initial_prompt_preserves_fast_prefill_and_forces_only_the_tail(
    runtime,
    prepared_audio,
) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "initial_prompt": "M87 Labs",
        },
        None,
    )

    prepared = skill.prepare_prompt(
        runtime,
        built.request_context,
        built.max_new_tokens,
    )

    assert [token.token_id for token in prepared.tokens] == [50362, 10, 50258, 50259]
    assert prepared.request_context.forced_prefix_tail == (50360,)
    assert prepared.request_context.max_transcript_tokens == 443
    assert prepared.max_new_tokens == 444

    state = skill.create_state(
        runtime,
        _request(prepared_audio, max_new_tokens=2),
        prepared.request_context,
    )
    assert state.phase == "prompt_prefix"
    assert state.allowed_token_ids(runtime) == (50360,)
    _consume(state, runtime, 50360, 0)
    assert state.phase == "transcript"
    _consume(state, runtime, 10, 1)
    result = state.finalize(runtime, reason="max_tokens")
    assert [token.token_id for token in result.tokens] == [10]
    assert result.output["text"] == "hello"


def test_encoder_input_prepares_to_runtime_only_audio_type(runtime) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
        },
        None,
    )

    # Admission sends this exact source through
    # runtime.preprocess_encoder_input_async. The runtime must return the
    # PreparedAudio object to prepare_sequence, rather than the request-facing
    # AudioSource carrier.
    prepared = prepare_audio(built.encoder_input)
    assert isinstance(prepared, PreparedAudio)
    assert prepared.duration_seconds == pytest.approx(0.1)

    state = skill.create_state(
        runtime,
        _request(prepared),
        built.request_context,
    )
    for position, token_id in enumerate([50365, 10, 50257]):
        _consume(state, runtime, token_id, position)
    assert state.finalize(runtime, reason="eos").output["segments"] == [
        _segment(0.0, 0.1)
    ]

    with pytest.raises(TypeError, match="replace AudioSource with PreparedAudio"):
        skill.create_state(
            runtime,
            _request(built.encoder_input),
            built.request_context,
        )


def test_segment_state_exposes_plan_and_finalizes(runtime, prepared_audio) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "segment",
        },
        None,
    )
    state = skill.create_state(
        runtime,
        _request(prepared_audio),
        built.request_context,
    )
    assert state.timestamp_plan() is not None
    for position, token_id in enumerate([50365, 10, 50370, 50257]):
        _consume(state, runtime, token_id, position)

    result = state.finalize(runtime, reason="eos")
    assert result.output["text"] == "hello"
    assert result.output["language_probability"] is None
    assert result.output["segments"] == [_segment(0.0, 0.1)]


def test_word_timestamps_run_terminal_alignment_and_attach_typed_fields(
    runtime,
    prepared_audio,
) -> None:
    calls = []

    def analyze_transcript(**kwargs):
        calls.append(kwargs)
        return TranscriptAnalysis(
            words=(AlignedWord(" hello", (10,), 0.02, 0.08, 0.875),),
            scores=TranscriptScores(-0.2, 0.01),
        )

    runtime.analyze_transcript = analyze_transcript
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "word",
        },
        None,
    )
    request = SimpleNamespace(
        encoder_input=prepared_audio,
        max_new_tokens=built.max_new_tokens,
        lifecycle=SimpleNamespace(state=SimpleNamespace(batch_idx=7)),
    )
    state = skill.create_state(runtime, request, built.request_context)
    assert state.timestamp_plan() is not None
    for position, token_id in enumerate([50365, 10, 50370, 50257]):
        _consume(state, runtime, token_id, position)

    assert calls == [
        {
            "batch_idx": 7,
            "language": "en",
            "task": "transcribe",
            "prefix_token_ids": (50258, 50259, 50360),
            "text_token_ids": (10,),
            "avg_logprob": -0.25,
            "duration_seconds": 0.1,
            "include_words": True,
        }
    ]
    assert state.finalize(runtime, reason="eos").output["segments"] == [
        _segment(
            0.0,
            0.1,
            avg_logprob=-0.2,
            words=[
                {
                    "word": " hello",
                    "start": 0.02,
                    "end": 0.08,
                    "probability": 0.875,
                }
            ],
        )
    ]


def test_transcript_confidence_uses_actual_selected_token_logprobs(
    runtime,
    prepared_audio,
) -> None:
    calls = []

    def analyze_transcript(**kwargs):
        calls.append(kwargs)
        return TranscriptAnalysis(
            words=(),
            scores=TranscriptScores(kwargs["avg_logprob"], 0.01),
        )

    runtime.analyze_transcript = analyze_transcript
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "segment",
        },
        None,
    )
    state = skill.create_state(runtime, _request(prepared_audio), built.request_context)
    for position, (token_id, logprob) in enumerate(
        [(50365, -0.1), (10, -0.2), (50370, -0.3), (50257, -0.4)]
    ):
        _consume(state, runtime, token_id, position, logprob=logprob)

    assert calls[0]["avg_logprob"] == pytest.approx(-0.25)
    assert calls[0]["prefix_token_ids"] == (50258, 50259, 50360)


@pytest.mark.parametrize("logprob", [None, float("nan"), float("inf"), 0.1])
def test_transcript_rejects_missing_or_invalid_selected_logprob(
    runtime,
    prepared_audio,
    logprob,
) -> None:
    skill = WhisperTranscribeSkill()
    built = skill.build_request(
        None,
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "sample_rate": 16000,
            "language": "en",
            "timestamps": "segment",
        },
        None,
    )
    state = skill.create_state(runtime, _request(prepared_audio), built.request_context)

    with pytest.raises(RuntimeError, match="selected-token logprob"):
        _consume(state, runtime, 50365, 0, logprob=logprob)
    assert state.token_count == 0
    assert state.transcript_token_ids == ()


def test_skill_rejects_unsupported_options_and_accepts_temperature() -> None:
    skill = WhisperTranscribeSkill()
    prompt = {
        "audio": np.zeros(100, dtype=np.float32),
        "sample_rate": 16000,
    }
    with pytest.raises(ValueError, match="beam_size"):
        skill.build_request(None, {**prompt, "beam_size": 2}, None)
    sampled = skill.build_request(None, prompt, {"temperature": 0.2})
    assert sampled.temperature == 0.2
    assert sampled.request_context.temperature == 0.2
    best_of = skill.build_request(
        None,
        prompt,
        {"temperature": 0.2, "best_of": 3},
    )
    assert best_of.temperature == 0.2
    with pytest.raises(TypeError, match="positive integer"):
        skill.build_request(None, prompt, {"max_tokens": 1.5})
    with pytest.raises(ValueError, match="target-position budget"):
        skill.build_request(None, prompt, {"max_tokens": 446})
