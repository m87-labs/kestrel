"""Exact CustomVoice prompt tokenization."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE

from .contract import CustomVoiceRequest, EncodedCustomVoiceRequest


_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
_ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_USER_PREFIX = "<|im_start|>user\n"
_USER_SUFFIX = "<|im_end|>\n"
_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class Qwen3TTSTextProcessor:
    """Tokenizer wrapper for the pinned checkpoint's official chat layout."""

    def __init__(self, checkpoint_directory: str | Path) -> None:
        root = Path(checkpoint_directory)
        config = json.loads((root / "tokenizer_config.json").read_text())
        backend = Tokenizer(
            BPE.from_file(str(root / "vocab.json"), str(root / "merges.txt"))
        )
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(_SPLIT_PATTERN),
                    behavior="isolated",
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )
        for raw_id, raw_token in config["added_tokens_decoder"].items():
            token = AddedToken(**raw_token)
            backend.add_tokens([token])
            if backend.token_to_id(token.content) != int(raw_id):
                raise ValueError("Qwen3-TTS tokenizer token IDs are not contiguous")
        self.backend = backend
        if len(self._tokenize(_ASSISTANT_PREFIX)) != 3:
            raise RuntimeError("Qwen3-TTS assistant prefix no longer tokenizes to three IDs")
        if len(self._tokenize(_ASSISTANT_SUFFIX)) != 5:
            raise RuntimeError("Qwen3-TTS assistant suffix no longer tokenizes to five IDs")

    def _tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self.backend.encode(text, add_special_tokens=False).ids)

    def encode(self, request: CustomVoiceRequest) -> EncodedCustomVoiceRequest:
        text_ids = self._tokenize(
            f"{_ASSISTANT_PREFIX}{request.text}{_ASSISTANT_SUFFIX}"
        )
        instruction_ids = (
            self._tokenize(f"{_USER_PREFIX}{request.instructions}{_USER_SUFFIX}")
            if request.instructions
            else ()
        )
        return EncodedCustomVoiceRequest(
            request=request,
            text_token_ids=text_ids,
            instruction_token_ids=instruction_ids,
        )


__all__ = ["Qwen3TTSTextProcessor"]
