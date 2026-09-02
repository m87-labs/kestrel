from types import SimpleNamespace

import pytest

from kestrel.models.qwen3_asr import skill as skill_module
from kestrel.models.qwen3_asr.skill import (
    Qwen3AsrTranscribeSkill,
    Qwen3AsrTranscribeState,
    QwenTranscribeContext,
)
from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep


class _Tokenizer:
    backend = object()

    def decode_result(self, token_ids, *, forced_language):
        return " ".join(map(str, token_ids)), forced_language


def _state(*, stream_callback):
    audio = SimpleNamespace(
        duration_seconds=1.0,
        source_duration_seconds=1.0,
        clip_start_seconds=0.0,
    )
    return Qwen3AsrTranscribeState(
        Qwen3AsrTranscribeSkill(),
        SimpleNamespace(stream_callback=stream_callback),
        QwenTranscribeContext(language=None, timestamps="none"),
        _Tokenizer(),
        SimpleNamespace(audio=audio),
    )


def _consume(state, token_id):
    state.consume_step(
        object(),
        DecodeStep(token=TextToken(token_id=token_id), position=0),
    )


def test_nonstream_state_skips_incremental_tokenizer_decode(monkeypatch) -> None:
    monkeypatch.setattr(
        skill_module,
        "DecodeStream",
        lambda **_kwargs: pytest.fail("constructed streaming decoder"),
    )
    state = _state(stream_callback=None)

    _consume(state, 17)

    assert state._token_ids == [17]
    assert state.pop_stream_delta(object()) is None
    assert state.stop_token_ids(SimpleNamespace(eos_token_ids=(1, 2))) is None


def test_stream_state_retains_incremental_text(monkeypatch) -> None:
    steps = []

    class _Stream:
        def __init__(self, *, skip_special_tokens):
            assert skip_special_tokens is False

        def step(self, backend, token_id):
            steps.append((backend, token_id))
            return "<asr_text> hello"

    monkeypatch.setattr(skill_module, "DecodeStream", _Stream)
    state = _state(stream_callback=lambda _update: None)

    _consume(state, 23)

    assert steps == [(_Tokenizer.backend, 23)]
    assert state.pop_stream_delta(object()) == "hello"
