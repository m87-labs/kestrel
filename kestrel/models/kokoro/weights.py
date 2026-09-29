"""Pinned Kokoro checkpoint and voice-pack loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .config import KokoroConfig
from .model import KokoroModel


DEFAULT_KOKORO_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "kokoro-v1_0.pth"

_COMPONENTS = frozenset(
    {"bert", "bert_encoder", "predictor", "decoder", "text_encoder"}
)


@dataclass(frozen=True, slots=True)
class KokoroCheckpointFiles:
    root: Path
    config: Path
    weights: Path


@dataclass(frozen=True, slots=True)
class LoadedKokoro:
    model: KokoroModel
    config: KokoroConfig
    voices: "VoiceStore"
    files: KokoroCheckpointFiles


def _files_in_directory(root: Path) -> KokoroCheckpointFiles:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kokoro checkpoint directory does not exist: {root}")
    files = KokoroCheckpointFiles(
        root=root,
        config=root / CONFIG_FILENAME,
        weights=root / WEIGHTS_FILENAME,
    )
    missing = [
        path.name for path in (files.config, files.weights) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Kokoro checkpoint is missing files: {missing}")
    return files


def resolve_checkpoint_files(
    checkpoint: str | Path = DEFAULT_KOKORO_REPO_ID,
    *,
    revision: str = DEFAULT_KOKORO_REVISION,
    local_files_only: bool = False,
) -> KokoroCheckpointFiles:
    path = Path(checkpoint).expanduser()
    if path.exists():
        if path.is_file():
            if path.name != WEIGHTS_FILENAME:
                raise ValueError(
                    f"Kokoro checkpoint file must be named {WEIGHTS_FILENAME!r}"
                )
            return _files_in_directory(path.parent)
        return _files_in_directory(path)

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            str(checkpoint),
            revision=revision,
            allow_patterns=(CONFIG_FILENAME, WEIGHTS_FILENAME, "voices/*.pt"),
            local_files_only=local_files_only,
        )
    )
    return _files_in_directory(root)


def _flatten_checkpoint(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Kokoro checkpoint must contain a component mapping")
    provided_components = set(checkpoint)
    if provided_components != _COMPONENTS:
        raise RuntimeError(
            "invalid Kokoro checkpoint components: "
            f"expected={sorted(_COMPONENTS)}, got={sorted(map(str, provided_components))}"
        )

    flattened: dict[str, torch.Tensor] = {}
    for component in sorted(_COMPONENTS):
        state = checkpoint[component]
        if not isinstance(state, Mapping):
            raise TypeError(f"Kokoro component {component!r} must be a state dict")
        for raw_name, tensor in state.items():
            if not isinstance(raw_name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError(f"invalid tensor entry in Kokoro component {component!r}")
            name = raw_name.removeprefix("module.")
            if name == raw_name:
                raise RuntimeError(
                    f"Kokoro checkpoint key lacks module. prefix: {raw_name!r}"
                )
            if component == "bert" and name in {"pooler.weight", "pooler.bias"}:
                continue
            flattened[f"{component}.{name}"] = tensor
    return flattened


def _materialize_weight_norm(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Collapse the checkpoint's training-time weight-norm representation."""

    materialized = {}
    for name, value in state.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            weight_name = name.removesuffix("_v")
            axes = tuple(range(1, value.ndim))
            value = value * (
                state[f"{weight_name}_g"]
                / torch.linalg.vector_norm(value, dim=axes, keepdim=True)
            )
            name = weight_name
        materialized[name] = value
    return materialized


def load_kokoro(
    checkpoint: str | Path = DEFAULT_KOKORO_REPO_ID,
    *,
    revision: str = DEFAULT_KOKORO_REVISION,
    device: torch.device | str = "cpu",
    local_files_only: bool = False,
) -> LoadedKokoro:
    """Load the pinned V1 model without importing ``kokoro`` or transformers."""

    files = resolve_checkpoint_files(
        checkpoint, revision=revision, local_files_only=local_files_only
    )
    config = KokoroConfig.from_json_file(files.config)
    # Constructing on meta avoids holding initialized parameters alongside the
    # 327 MB checkpoint. ``assign=True`` installs the validated CPU tensors.
    with torch.device("meta"):
        model = KokoroModel(config)
    state = torch.load(
        files.weights,
        map_location="cpu",
        weights_only=True,
    )
    flattened = _materialize_weight_norm(_flatten_checkpoint(state))
    model.load_state_dict(flattened, strict=True, assign=True)
    model.to(device=torch.device(device), dtype=torch.float32).eval()
    del state, flattened

    voices = VoiceStore(root=files.root)
    return LoadedKokoro(model=model, config=config, voices=voices, files=files)


class VoiceStore:
    """Load checkpoint-local voice packs into CPU memory on first use."""

    def __init__(
        self,
        *,
        root: Path,
    ) -> None:
        self.root = root
        self._cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _names(voice: str) -> tuple[str, ...]:
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice must be a non-empty string")
        names = tuple(part.strip().casefold() for part in voice.split(","))
        if any(
            not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            for name in names
        ):
            raise ValueError("voice must contain comma-separated voice names")
        return names

    def _voice_path(self, name: str) -> Path:
        relative = f"voices/{name}.pt"
        local = self.root / relative
        if not local.is_file():
            raise FileNotFoundError(f"Kokoro voice pack does not exist: {local}")
        return local

    def _load_one(self, name: str) -> torch.Tensor:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        pack = torch.load(
            self._voice_path(name), map_location="cpu", weights_only=True
        )
        if not isinstance(pack, torch.Tensor):
            raise TypeError(f"Kokoro voice {name!r} must contain one tensor")
        if pack.ndim != 3 or tuple(pack.shape[1:]) != (1, 256):
            raise RuntimeError(
                f"Kokoro voice {name!r} has shape {tuple(pack.shape)}, "
                "expected [length, 1, 256]"
            )
        if not pack.is_floating_point():
            raise TypeError(f"Kokoro voice {name!r} must be floating point")
        pack = pack.to(dtype=torch.float32).contiguous()
        self._cache[name] = pack
        return pack

    def load(self, voice: str) -> torch.Tensor:
        names = self._names(voice)
        key = ",".join(names)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        packs = [self._load_one(name) for name in names]
        if len({tuple(pack.shape) for pack in packs}) != 1:
            raise ValueError("blended Kokoro voices must have matching pack shapes")
        pack = packs[0] if len(packs) == 1 else torch.stack(packs).mean(dim=0)
        self._cache[key] = pack
        return pack

    def style(self, voice: str, phoneme_length: int) -> torch.Tensor:
        pack = self.load(voice)
        if not 1 <= phoneme_length <= pack.shape[0]:
            raise ValueError(
                f"phoneme length {phoneme_length} is outside voice pack range "
                f"1..{pack.shape[0]}"
            )
        # This index is part of the published Kokoro pipeline contract.
        return pack[phoneme_length - 1]


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_KOKORO_MODEL",
    "DEFAULT_KOKORO_REPO_ID",
    "DEFAULT_KOKORO_REVISION",
    "KokoroCheckpointFiles",
    "LoadedKokoro",
    "VoiceStore",
    "WEIGHTS_FILENAME",
    "load_kokoro",
    "resolve_checkpoint_files",
]
