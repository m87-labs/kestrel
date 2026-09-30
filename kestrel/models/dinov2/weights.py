"""Direct config, processor, and safetensors loading for DINOv2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Dinov2Config, Dinov2ProcessorConfig
from .metadata import DEFAULT_DINOV2_REPO_ID, DEFAULT_DINOV2_REVISION


CONFIG_FILENAME = "config.json"
PROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"
WEIGHTS_FILENAME = "model.safetensors"

# The checkpoint's mask token is used only by the removed pretraining path.
_IGNORED_CHECKPOINT_KEYS = frozenset({"embeddings.mask_token"})


@dataclass(frozen=True)
class Dinov2CheckpointFiles:
    root: Path
    config: Path
    processor_config: Path
    weights: Path


@dataclass(frozen=True)
class LoadedDinov2:
    state_dict: dict[str, torch.Tensor]
    model_config: Dinov2Config
    processor_config: Dinov2ProcessorConfig
    files: Dinov2CheckpointFiles


def _files_in_directory(root: Path) -> Dinov2CheckpointFiles:
    if not root.is_dir():
        raise FileNotFoundError(f"DINOv2 checkpoint directory does not exist: {root}")
    files = Dinov2CheckpointFiles(
        root=root,
        config=root / CONFIG_FILENAME,
        processor_config=root / PROCESSOR_CONFIG_FILENAME,
        weights=root / WEIGHTS_FILENAME,
    )
    missing = [
        path.name
        for path in (files.config, files.processor_config, files.weights)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"DINOv2 checkpoint is missing files: {missing}")
    return files


def resolve_checkpoint_files(
    checkpoint: str | Path = DEFAULT_DINOV2_REPO_ID,
    *,
    revision: str = DEFAULT_DINOV2_REVISION,
    local_files_only: bool = False,
) -> Dinov2CheckpointFiles:
    """Resolve either a local checkpoint directory or a pinned Hub snapshot."""

    if isinstance(checkpoint, Path):
        local_path = checkpoint.expanduser()
        if local_path.is_file():
            if local_path.name != WEIGHTS_FILENAME:
                raise ValueError(
                    f"DINOv2 checkpoint file must be named {WEIGHTS_FILENAME!r}, got "
                    f"{local_path.name!r}"
                )
            return _files_in_directory(local_path.parent.resolve())
        return _files_in_directory(local_path.resolve())
    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        if local_path.is_file():
            if local_path.name != WEIGHTS_FILENAME:
                raise ValueError(
                    f"DINOv2 checkpoint file must be named {WEIGHTS_FILENAME!r}, got "
                    f"{local_path.name!r}"
                )
            return _files_in_directory(local_path.parent.resolve())
        return _files_in_directory(local_path.resolve())

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            checkpoint,
            revision=revision,
            allow_patterns=[CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, WEIGHTS_FILENAME],
            local_files_only=local_files_only,
        )
    )
    return _files_in_directory(root)


def load_dinov2(
    checkpoint: str | Path = DEFAULT_DINOV2_REPO_ID,
    *,
    revision: str = DEFAULT_DINOV2_REVISION,
    local_files_only: bool = False,
) -> LoadedDinov2:
    """Load the pinned V1 checkpoint without importing ``transformers``."""

    files = resolve_checkpoint_files(
        checkpoint,
        revision=revision,
        local_files_only=local_files_only,
    )
    model_config = Dinov2Config.from_json_file(files.config)
    model_config.validate_v1()
    processor_config = Dinov2ProcessorConfig.from_json_file(files.processor_config)
    processor_config.validate_v1()

    from safetensors.torch import load_file

    checkpoint_state = load_file(str(files.weights), device="cpu")
    state_dict = {}
    for name, value in checkpoint_state.items():
        if name in _IGNORED_CHECKPOINT_KEYS:
            continue
        if not value.is_floating_point():
            raise ValueError(f"DINOv2 checkpoint tensor {name!r} must be floating point")
        # Preserve the source dtype used by the original checkpoint binding path.
        # FP32 checkpoint tensors retain their storage without a model-sized copy.
        state_dict[name] = value.float()
    if not state_dict:
        raise ValueError("DINOv2 checkpoint has no inference tensors")
    return LoadedDinov2(
        state_dict=state_dict,
        model_config=model_config,
        processor_config=processor_config,
        files=files,
    )


__all__ = [
    "CONFIG_FILENAME",
    "Dinov2CheckpointFiles",
    "LoadedDinov2",
    "PROCESSOR_CONFIG_FILENAME",
    "WEIGHTS_FILENAME",
    "load_dinov2",
    "resolve_checkpoint_files",
]
