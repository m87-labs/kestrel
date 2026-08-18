"""Pinned Hugging Face asset resolution for Whisper large-v3-turbo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ID = "openai/whisper-large-v3-turbo"
MODEL_NAME = REPO_ID

# Hugging Face model_info(repo_id).sha on 2026-08-09. The repository was last
# modified on 2024-10-04. Never replace this with ``main``: config, tokenizer,
# weights, correctness fixtures, and performance results must identify one
# immutable checkpoint together.
CHECKPOINT_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

CHECKPOINT_ASSETS = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "normalizer.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)


@dataclass(frozen=True, slots=True)
class WhisperAssets:
    """Resolve only declared checkpoint assets, from a local bundle or the pin."""

    local_dir: Path | None = None
    repo_id: str = REPO_ID
    revision: str = CHECKPOINT_REVISION

    def __post_init__(self) -> None:
        if self.local_dir is not None:
            root = Path(self.local_dir)
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Whisper checkpoint directory not found: {root}"
                )
            object.__setattr__(self, "local_dir", root)
        if not self.repo_id:
            raise ValueError("Whisper repo_id must not be empty")
        if not self.revision:
            raise ValueError("Whisper revision must not be empty")

    def path(self, filename: str) -> Path:
        """Return one declared file without ever falling back from a local bundle."""

        if filename not in CHECKPOINT_ASSETS:
            raise ValueError(f"Undeclared Whisper checkpoint asset: {filename!r}")
        if self.local_dir is not None:
            path = self.local_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Whisper checkpoint is missing required asset {filename!r}: {path}"
                )
            return path

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo_id,
                filename=filename,
                revision=self.revision,
            )
        )

    def load_json(self, filename: str) -> dict[str, Any]:
        if not filename.endswith(".json"):
            raise ValueError(f"Whisper JSON asset must end in .json: {filename!r}")
        with self.path(filename).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Whisper asset {filename!r} must contain a JSON object")
        return value


__all__ = [
    "CHECKPOINT_ASSETS",
    "CHECKPOINT_REVISION",
    "MODEL_NAME",
    "REPO_ID",
    "WhisperAssets",
]
