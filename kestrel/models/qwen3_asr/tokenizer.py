"""Qwen3-ASR prompt construction and output parsing."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE


LANGUAGES = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def resolve_language(language: str | None) -> str | None:
    if language is None:
        return None
    lowered = language.lower()
    if lowered in LANGUAGES:
        return LANGUAGES[lowered]
    for name in LANGUAGES.values():
        if lowered == name.lower():
            return name
    raise ValueError(f"Qwen3-ASR does not support language {language!r}")


def language_code(language: str) -> str:
    canonical = resolve_language(language)
    assert canonical is not None
    return next(code for code, name in LANGUAGES.items() if name == canonical)


class Qwen3AsrTokenizer:
    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        with (root / "tokenizer_config.json").open(encoding="utf-8") as file:
            config = json.load(file)
        backend = Tokenizer(
            BPE.from_file(str(root / "vocab.json"), str(root / "merges.txt"))
        )
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
                        r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
                        r"\s+(?!\S)|\s+"
                    ),
                    behavior="isolated",
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )
        backend.decoder = decoders.ByteLevel()
        for raw_id, raw_token in config["added_tokens_decoder"].items():
            token = AddedToken(**raw_token)
            backend.add_tokens([token])
            if backend.token_to_id(token.content) != int(raw_id):
                raise ValueError("Qwen3-ASR tokenizer token IDs are not contiguous")
        self.backend = backend
        self.audio_token_id = self._token_id("<|audio_pad|>")

    def _token_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Qwen3-ASR tokenizer is missing {token!r}")
        return token_id

    def prompt_ids(
        self,
        audio_tokens: int,
        *,
        language: str | None,
        initial_prompt: str | None,
    ) -> list[int]:
        if audio_tokens <= 0:
            raise ValueError("audio_tokens must be positive")
        language = resolve_language(language)
        system = "" if initial_prompt is None else initial_prompt
        prefix = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>"
        )
        suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        if language is not None:
            suffix += f"language {language}<asr_text>"
        prefix_ids = self.backend.encode(prefix, add_special_tokens=False).ids
        suffix_ids = self.backend.encode(suffix, add_special_tokens=False).ids
        return prefix_ids + [self.audio_token_id] * audio_tokens + suffix_ids

    def decode_result(
        self,
        token_ids: list[int],
        *,
        forced_language: str | None,
    ) -> tuple[str, str | None]:
        raw = self.backend.decode(token_ids, skip_special_tokens=False)
        raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if forced_language is not None:
            return raw, resolve_language(forced_language)
        marker = "<asr_text>"
        if marker not in raw:
            return raw, None
        prefix, text = raw.split(marker, 1)
        language = prefix.removeprefix("language ").strip() or None
        if language is not None and language.lower() == "none":
            language = None
        return text.strip(), language


__all__ = ["LANGUAGES", "Qwen3AsrTokenizer", "language_code", "resolve_language"]
