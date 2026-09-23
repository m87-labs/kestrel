"""Pinned checkpoint resolution and direct safetensors loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .config import (
    DEFAULT_REPO_ID,
    Qwen3TTSConfig,
    SUPPORTED_CHECKPOINTS,
)


MODEL_CONFIG_FILENAME = "config.json"
MODEL_WEIGHTS_FILENAME = "model.safetensors"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
TOKENIZER_MERGES_FILENAME = "merges.txt"
TOKENIZER_VOCAB_FILENAME = "vocab.json"
CODEC_CONFIG_FILENAME = "speech_tokenizer/config.json"
CODEC_WEIGHTS_FILENAME = "speech_tokenizer/model.safetensors"


@dataclass(frozen=True, slots=True)
class Qwen3TTSCheckpointFiles:
    root: Path
    model_config: Path
    model_weights: Path
    tokenizer_config: Path
    tokenizer_merges: Path
    tokenizer_vocab: Path
    codec_config: Path
    codec_weights: Path

def _checkpoint_files(root: Path) -> Qwen3TTSCheckpointFiles:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen3-TTS checkpoint directory does not exist: {root}")
    files = Qwen3TTSCheckpointFiles(
        root=root,
        model_config=root / MODEL_CONFIG_FILENAME,
        model_weights=root / MODEL_WEIGHTS_FILENAME,
        tokenizer_config=root / TOKENIZER_CONFIG_FILENAME,
        tokenizer_merges=root / TOKENIZER_MERGES_FILENAME,
        tokenizer_vocab=root / TOKENIZER_VOCAB_FILENAME,
        codec_config=root / CODEC_CONFIG_FILENAME,
        codec_weights=root / CODEC_WEIGHTS_FILENAME,
    )
    missing = [
        path.relative_to(root).as_posix()
        for path in (
            files.model_config,
            files.model_weights,
            files.tokenizer_config,
            files.tokenizer_merges,
            files.tokenizer_vocab,
            files.codec_config,
            files.codec_weights,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Qwen3-TTS checkpoint is missing files: {missing}")
    Qwen3TTSConfig.from_directory(files.root)
    return files


def resolve_checkpoint_files(
    checkpoint: str | Path = DEFAULT_REPO_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Qwen3TTSCheckpointFiles:
    """Resolve a local snapshot or a supported immutable Hub revision."""

    local = Path(checkpoint).expanduser()
    if local.exists() or isinstance(checkpoint, Path):
        return _checkpoint_files(local)
    expected_revision = SUPPORTED_CHECKPOINTS.get(str(checkpoint))
    if expected_revision is None:
        raise ValueError(
            f"unsupported Qwen3-TTS checkpoint {checkpoint!r}; expected one of "
            f"{sorted(SUPPORTED_CHECKPOINTS)}"
        )
    selected_revision = expected_revision if revision is None else revision
    if selected_revision != expected_revision:
        raise ValueError(
            f"unsupported revision for {checkpoint!r}: {selected_revision!r}; "
            f"expected {expected_revision!r}"
        )

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=str(checkpoint),
            revision=selected_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=[
                MODEL_CONFIG_FILENAME,
                MODEL_WEIGHTS_FILENAME,
                TOKENIZER_CONFIG_FILENAME,
                TOKENIZER_MERGES_FILENAME,
                TOKENIZER_VOCAB_FILENAME,
                CODEC_CONFIG_FILENAME,
                CODEC_WEIGHTS_FILENAME,
            ],
        )
    )
    return _checkpoint_files(root)


def _matches(
    name: str,
    prefix: str | None,
    excluded_prefixes: Sequence[str],
) -> bool:
    return (prefix is None or name.startswith(prefix)) and not name.startswith(
        tuple(excluded_prefixes)
    )


def safetensor_names(
    path: str | Path,
    *,
    prefix: str | None = None,
    excluded_prefixes: Sequence[str] = (),
) -> tuple[str, ...]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return tuple(
            name
            for name in handle.keys()
            if _matches(name, prefix, excluded_prefixes)
        )


@torch.no_grad()
def load_module_from_safetensors(
    module: nn.Module,
    path: str | Path,
    *,
    prefix: str = "",
    excluded_prefixes: Sequence[str] = (),
) -> None:
    """Copy an exactly matching checkpoint prefix into an allocated module."""

    destination: Mapping[str, torch.Tensor] = module.state_dict()
    source_names = safetensor_names(
        path,
        prefix=prefix,
        excluded_prefixes=excluded_prefixes,
    )
    mapped_names = {name: name.removeprefix(prefix) for name in source_names}
    expected = set(destination)
    provided = set(mapped_names.values())
    missing = expected - provided
    unexpected = provided - expected
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if unexpected:
            parts.append(f"unexpected={sorted(unexpected)}")
        raise RuntimeError("invalid Qwen3-TTS checkpoint keys: " + "; ".join(parts))

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for source_name, target_name in mapped_names.items():
            source = handle.get_tensor(source_name)
            target = destination[target_name]
            if source.shape != target.shape:
                raise RuntimeError(
                    f"Qwen3-TTS tensor {source_name} has shape {tuple(source.shape)}; "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))


__all__ = [
    "CODEC_CONFIG_FILENAME",
    "CODEC_WEIGHTS_FILENAME",
    "MODEL_CONFIG_FILENAME",
    "MODEL_WEIGHTS_FILENAME",
    "Qwen3TTSCheckpointFiles",
    "load_module_from_safetensors",
    "resolve_checkpoint_files",
    "safetensor_names",
]
