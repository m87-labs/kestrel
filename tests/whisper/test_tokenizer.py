from __future__ import annotations

import pytest

from kestrel.models.whisper.tokenizer import (
    SUPPORTED_LANGUAGE_CODES,
    WhisperControlTokens,
    WhisperTokenizer,
    normalize_language_code,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Backend:
    def get_vocab_size(self, *, with_added_tokens):
        assert with_added_tokens is True
        return 51866

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return _Encoding([10, 11] if text else [])

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join({10: " hello", 11: " world"}.get(i, "") for i in ids)


def test_generation_control_ids_are_strict(generation_config_dict) -> None:
    controls = WhisperControlTokens.from_dict(generation_config_dict)
    assert len(controls.language_token_ids) == 100
    assert controls.language_id("en") == 50259
    assert controls.language_id("<|yue|>") == 50358
    assert controls.language_code(50261) == "de"
    assert controls.prompt_ids("en", timestamps="segment") == (
        50258,
        50259,
        50360,
    )
    assert controls.prompt_ids("en", timestamps="word") == (
        50258,
        50259,
        50360,
    )
    assert controls.alignment_heads == (
        (2, 4),
        (2, 11),
        (3, 3),
        (3, 6),
        (3, 11),
        (3, 14),
    )
    assert controls.prompt_ids("en", timestamps="none")[-1] == 50364
    assert controls.prompt_ids("fr", timestamps="segment", task="translate") == (
        50258,
        50265,
        50359,
    )
    assert controls.prompt_ids(None, timestamps="segment") == (50258,)

    generation_config_dict["lang_to_id"].pop("<|en|>")
    with pytest.raises(ValueError, match="lang_to_id"):
        WhisperControlTokens.from_dict(generation_config_dict)


def test_generation_config_rejects_alignment_head_drift(
    generation_config_dict,
) -> None:
    generation_config_dict["alignment_heads"] = [[2, 4]]
    with pytest.raises(ValueError, match="alignment_heads"):
        WhisperControlTokens.from_dict(generation_config_dict)


def test_language_normalization_is_codes_only() -> None:
    assert normalize_language_code(" EN ") == "en"
    assert normalize_language_code("<|fr|>") == "fr"
    assert len(SUPPORTED_LANGUAGE_CODES) == 100
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_language_code("english")


def test_tokenizer_filters_control_and_timestamp_ids() -> None:
    tokenizer = WhisperTokenizer(_Backend(), WhisperControlTokens(suppress_tokens=()))
    assert tokenizer.decode_text([50258, 10, 50365, 11, 50257]) == " hello world"
    assert tokenizer.encode_text("hello") == (10, 11)
    with pytest.raises(ValueError, match="at least one"):
        tokenizer.encode_text("")


def test_prompt_encoding_refuses_control_tokens() -> None:
    class _ControlBackend(_Backend):
        def encode(self, text, *, add_special_tokens):
            return _Encoding([10, 50362])

    tokenizer = WhisperTokenizer(
        _ControlBackend(),
        WhisperControlTokens(suppress_tokens=()),
    )
    with pytest.raises(ValueError, match="control tokens"):
        tokenizer.encode_text("unsafe")


def test_word_splitting_groups_space_delimited_bpe_and_terminal_control() -> None:
    tokenizer = WhisperTokenizer(_Backend(), WhisperControlTokens(suppress_tokens=()))
    words, token_ids = tokenizer.split_to_word_tokens(
        [10, 11, tokenizer.controls.eos_id],
        language="en",
    )
    assert words == [" hello", " world", ""]
    assert token_ids == [[10], [11], [tokenizer.controls.eos_id]]
