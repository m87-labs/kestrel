"""Tokenizer and immutable control-token facts for Whisper Turbo."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .assets import WhisperAssets


# The pinned generation config maps these codes, in this order, to the
# contiguous token range 50259..50358.
SUPPORTED_LANGUAGE_CODES = (
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
    "yue",
)
_LANGUAGE_INDEX = {code: index for index, code in enumerate(SUPPORTED_LANGUAGE_CODES)}
_EXPECTED_LANG_TO_ID = {
    f"<|{code}|>": 50259 + index for index, code in enumerate(SUPPORTED_LANGUAGE_CODES)
}
_EXPECTED_ALIGNMENT_HEADS = (
    (2, 4),
    (2, 11),
    (3, 3),
    (3, 6),
    (3, 11),
    (3, 14),
)
_LANGUAGES_WITHOUT_SPACES = frozenset({"zh", "ja", "th", "lo", "my", "yue"})


def normalize_language_code(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string code or None")
    code = language.strip().lower()
    if code.startswith("<|") and code.endswith("|>"):
        code = code[2:-2]
    if code not in _LANGUAGE_INDEX:
        raise ValueError(f"Unsupported Whisper language code: {language!r}")
    return code


@dataclass(frozen=True, slots=True)
class WhisperControlTokens:
    """Control IDs and suppression tables consumed by skill/runtime code."""

    # Required so a runtime cannot silently omit the checkpoint's model-specific
    # suppression table. Tests may pass an explicit empty tuple.
    suppress_tokens: tuple[int, ...]
    alignment_heads: tuple[tuple[int, int], ...] = _EXPECTED_ALIGNMENT_HEADS
    begin_suppress_tokens: tuple[int, ...] = (220, 50257)
    eos_id: int = 50257
    decoder_start_id: int = 50258
    transcribe_id: int = 50360
    translate_id: int = 50359
    no_speech_id: int = 50361
    no_timestamps_id: int = 50364
    prev_sot_id: int = 50362
    timestamp_begin_id: int = 50365
    max_initial_timestamp_index: int = 50
    vocab_size: int = 51866
    max_target_positions: int = 448

    @property
    def language_token_ids(self) -> tuple[int, ...]:
        return tuple(range(50259, 50359))

    def language_id(self, language: str) -> int:
        code = normalize_language_code(language)
        return 50259 + _LANGUAGE_INDEX[code]

    def language_code(self, token_id: int) -> str:
        index = int(token_id) - 50259
        if not 0 <= index < len(SUPPORTED_LANGUAGE_CODES):
            raise ValueError(f"Token {token_id} is not a Whisper language token")
        return SUPPORTED_LANGUAGE_CODES[index]

    def task_id(self, task: str) -> int:
        if task == "transcribe":
            return self.transcribe_id
        if task == "translate":
            return self.translate_id
        raise ValueError("task must be 'transcribe' or 'translate'")

    def prompt_ids(
        self,
        language: str | None,
        *,
        timestamps: str,
        task: str = "transcribe",
    ) -> tuple[int, ...]:
        if timestamps not in {"none", "segment", "word"}:
            raise ValueError("timestamps must be 'none', 'segment', or 'word'")
        task_id = self.task_id(task)
        if language is None:
            return (self.decoder_start_id,)
        ids = [self.decoder_start_id, self.language_id(language), task_id]
        if timestamps == "none":
            ids.append(self.no_timestamps_id)
        return tuple(ids)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WhisperControlTokens":
        if not isinstance(data, Mapping):
            raise TypeError("Whisper generation config must be a mapping")
        expected_scalars = {
            "bos_token_id": 50257,
            "decoder_start_token_id": 50258,
            "eos_token_id": 50257,
            "pad_token_id": 50257,
            "prev_sot_token_id": 50362,
            "no_timestamps_token_id": 50364,
            "max_initial_timestamp_index": 50,
            "max_length": 448,
            "is_multilingual": True,
            "return_timestamps": False,
        }
        failures = [
            f"{name}={data.get(name)!r} (expected {wanted!r})"
            for name, wanted in expected_scalars.items()
            if data.get(name) != wanted
        ]
        if data.get("task_to_id") != {"translate": 50359, "transcribe": 50360}:
            failures.append("task_to_id does not match Whisper Turbo control IDs")
        if data.get("forced_decoder_ids") != [[1, None], [2, 50360]]:
            failures.append("forced_decoder_ids does not match Whisper Turbo controls")
        if data.get("lang_to_id") != _EXPECTED_LANG_TO_ID:
            failures.append("lang_to_id does not match the pinned 100-language table")
        alignment = data.get("alignment_heads")
        if alignment != [list(index) for index in _EXPECTED_ALIGNMENT_HEADS]:
            failures.append(
                "alignment_heads does not match the pinned Whisper Turbo heads"
            )
        begin = data.get("begin_suppress_tokens")
        if begin != [220, 50257]:
            failures.append(f"begin_suppress_tokens={begin!r} (expected [220, 50257])")
        suppress = data.get("suppress_tokens")
        if not isinstance(suppress, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in suppress
        ):
            failures.append("suppress_tokens must be a list of integer token IDs")
            suppress = []
        elif any(not 0 <= token_id < 51866 for token_id in suppress):
            failures.append("suppress_tokens contains an out-of-vocabulary token ID")
        if failures:
            raise ValueError(
                "Unsupported Whisper generation config: " + "; ".join(failures)
            )
        return cls(suppress_tokens=tuple(suppress))

    @classmethod
    def from_assets(cls, assets: WhisperAssets) -> "WhisperControlTokens":
        return cls.from_dict(assets.load_json("generation_config.json"))


class WhisperTokenizer:
    """Small ``tokenizers`` wrapper with Whisper-aware text filtering."""

    def __init__(self, backend: Any, controls: WhisperControlTokens) -> None:
        vocab_size = int(backend.get_vocab_size(with_added_tokens=True))
        if vocab_size != controls.vocab_size:
            raise ValueError(
                f"Whisper tokenizer vocabulary has {vocab_size} tokens, "
                f"expected {controls.vocab_size}"
            )
        self._backend = backend
        self.controls = controls

    @classmethod
    def from_assets(cls, assets: WhisperAssets) -> "WhisperTokenizer":
        from tokenizers import Tokenizer

        controls = WhisperControlTokens.from_assets(assets)
        backend = Tokenizer.from_file(str(assets.path("tokenizer.json")))
        return cls(backend, controls)

    def decode_text(self, token_ids: Sequence[int]) -> str:
        # All tokenizer text/BPE IDs precede EOT. Timestamp tokens are not marked
        # special in tokenizer.json, so skip_special_tokens alone is insufficient.
        text_ids = [
            int(token_id)
            for token_id in token_ids
            if 0 <= int(token_id) < self.controls.eos_id
        ]
        if not text_ids:
            return ""
        return str(self._backend.decode(text_ids, skip_special_tokens=False))

    def encode_text(self, text: str) -> tuple[int, ...]:
        """Encode plain prompt text while refusing embedded control tokens."""

        if not isinstance(text, str):
            raise TypeError("Whisper prompt text must be a string")
        encoded = self._backend.encode(text, add_special_tokens=False)
        raw_ids = getattr(encoded, "ids", None)
        if not isinstance(raw_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in raw_ids
        ):
            raise ValueError("Whisper tokenizer returned invalid prompt token IDs")
        token_ids = tuple(int(token_id) for token_id in raw_ids)
        if not token_ids:
            raise ValueError("initial_prompt must encode to at least one text token")
        if any(not 0 <= token_id < self.controls.eos_id for token_id in token_ids):
            raise ValueError("initial_prompt must not contain Whisper control tokens")
        return token_ids

    def _split_tokens_on_unicode(
        self,
        token_ids: Sequence[int],
    ) -> tuple[list[str], list[list[int]]]:
        full = self.decode_text(token_ids)
        replacement = "\ufffd"
        words: list[str] = []
        groups: list[list[int]] = []
        current: list[int] = []
        unicode_offset = 0
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id >= self.controls.eos_id:
                if current:
                    raise ValueError("text token sequence ended inside invalid UTF-8")
                words.append("")
                groups.append([token_id])
                continue
            current.append(token_id)
            decoded = self.decode_text(current)
            replacement_index = decoded.find(replacement)
            if replacement_index < 0 or (
                unicode_offset + replacement_index < len(full)
                and full[unicode_offset + replacement_index] == replacement
            ):
                words.append(decoded)
                groups.append(current)
                current = []
                unicode_offset += len(decoded)
        if current:
            raise ValueError("text token sequence ended inside invalid UTF-8")
        return words, groups

    def split_to_word_tokens(
        self,
        token_ids: Sequence[int],
        *,
        language: str,
    ) -> tuple[list[str], list[list[int]]]:
        """Group BPE tokens into the checkpoint's language-aware word units."""

        code = normalize_language_code(language)
        subwords, subword_tokens = self._split_tokens_on_unicode(token_ids)
        if code in _LANGUAGES_WITHOUT_SPACES:
            return subwords, subword_tokens

        words: list[str] = []
        word_tokens: list[list[int]] = []
        for subword, tokens in zip(subwords, subword_tokens, strict=True):
            special = tokens[0] >= self.controls.eos_id
            starts_with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or starts_with_space or punctuation or not words:
                words.append(subword)
                word_tokens.append(tokens)
            else:
                words[-1] += subword
                word_tokens[-1].extend(tokens)
        return words, word_tokens


__all__ = [
    "SUPPORTED_LANGUAGE_CODES",
    "WhisperControlTokens",
    "WhisperTokenizer",
    "normalize_language_code",
]
