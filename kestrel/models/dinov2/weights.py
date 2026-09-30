"""Direct config, processor, and safetensors loading for DINOv2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .config import Dinov2Config, Dinov2ProcessorConfig
from .model import Dinov2Model
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
    model: Dinov2Model
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


def _copy_checkpoint_into_model(
    model: Dinov2Model,
    checkpoint_state: Mapping[str, torch.Tensor],
) -> None:
    destination = model.state_dict()
    expected = set(destination)
    provided = set(checkpoint_state)
    missing = expected - provided
    unexpected = provided - expected - _IGNORED_CHECKPOINT_KEYS
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if unexpected:
            parts.append(f"unexpected={sorted(unexpected)}")
        raise RuntimeError("invalid DINOv2 checkpoint keys: " + "; ".join(parts))

    shape_errors = [
        f"{name}: checkpoint {tuple(checkpoint_state[name].shape)} != model {tuple(tensor.shape)}"
        for name, tensor in destination.items()
        if tuple(checkpoint_state[name].shape) != tuple(tensor.shape)
    ]
    dtype_errors = [
        f"{name}: checkpoint dtype {checkpoint_state[name].dtype} is not floating point"
        for name in destination
        if not checkpoint_state[name].is_floating_point()
    ]
    if shape_errors or dtype_errors:
        raise RuntimeError(
            "invalid DINOv2 checkpoint tensors: " + "; ".join(shape_errors + dtype_errors)
        )

    with torch.no_grad():
        for name, target in destination.items():
            source = checkpoint_state[name]
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_dinov2(
    checkpoint: str | Path = DEFAULT_DINOV2_REPO_ID,
    *,
    revision: str = DEFAULT_DINOV2_REVISION,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
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

    # Checkpoint loading overwrites every inference parameter; avoid allocating
    # and randomly initializing a second full model before that copy.
    with torch.device("meta"):
        model = Dinov2Model(model_config).to(dtype=dtype)
    model.to_empty(device=device)
    from safetensors.torch import load_file

    checkpoint_state = load_file(str(files.weights), device="cpu")
    _copy_checkpoint_into_model(model, checkpoint_state)
    model.eval()
    return LoadedDinov2(
        model=model,
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
