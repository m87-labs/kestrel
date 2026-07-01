"""Tokenizer loading helpers for Moondream runtimes."""

from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(
    tokenizer_id: str | None,
    tokenizer_path: str | Path | None,
) -> Tokenizer:
    if tokenizer_path is not None:
        path = Path(tokenizer_path).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        return Tokenizer.from_file(str(path))
    if tokenizer_id is None:
        raise ValueError("Moondream model spec must declare tokenizer_id")
    return Tokenizer.from_pretrained(tokenizer_id)
