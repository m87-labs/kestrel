"""Direct Gemma 4 config and weight loading without Transformers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from kestrel.runtime.bounded_projection import (
    bind_declared_packed_projections,
    declared_packed_projection_source_keys,
)

from .config import Gemma4Config
from .model import Gemma4InferenceModel


_UNSUPPORTED_WEIGHT_PREFIXES = (
    "model.audio_tower.",
    "model.embed_audio.",
)


def _snapshot(source: str | Path, *, revision: str | None = None) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Gemma model_path must be a directory, got {path}")
        return path
    kwargs: dict[str, object] = {
        "allow_patterns": [
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ]
    }
    if revision is not None:
        kwargs["revision"] = revision
    return Path(snapshot_download(str(source), **kwargs))


def load_weights(source: str | Path, model: torch.nn.Module) -> None:
    snapshot = _snapshot(source)

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]

    expected_state = model.state_dict()
    expected = set(expected_state)
    packed_source_keys = declared_packed_projection_source_keys(model)
    loaded: set[str] = set()
    seen_checkpoint_names: set[str] = set()
    pending_packed: dict[str, torch.Tensor] = {}

    def copy_loaded_targets() -> None:
        for name in tuple(pending_packed):
            if name not in expected:
                continue
            source_tensor = pending_packed.pop(name)
            target = expected_state[name]
            if source_tensor.shape != target.shape:
                raise ValueError(
                    f"Gemma checkpoint tensor {name!r} has shape "
                    f"{tuple(source_tensor.shape)}, expected {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(source_tensor)
            loaded.add(name)

    def bind_ready_packed(*, require_complete: bool) -> None:
        bind_declared_packed_projections(
            model,
            pending_packed,
            already_bound_targets=loaded,
            require_complete=require_complete,
        )
        copy_loaded_targets()

    def is_omitted_shared_kv(name: str) -> bool:
        shared_kv_members = ("k_proj", "v_proj", "k_norm", "v_norm")
        for member in shared_kv_members:
            marker = f".self_attn.{member}."
            if marker not in name:
                continue
            attention_path = name.partition(marker)[0] + ".self_attn"
            try:
                attention = model.get_submodule(attention_path)
            except AttributeError:
                return False
            return not hasattr(attention, member)
        return False

    for shard_name in shard_names:
        shard_path = snapshot / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            shard_keys = list(handle.keys())
        # A shard-wide mmap retained every faulted page through the load. Opening
        # per tensor cut Gemma 31B host high-water from 74.65 GB to 24.88 GB on
        # B200 while leaving the 65.45 GB CUDA peak unchanged; warm load time was
        # 9.81 s versus 9.27 s with the shard-wide mapping.
        for name in shard_keys:
            if name in seen_checkpoint_names:
                raise ValueError(f"duplicate Gemma checkpoint tensor {name!r}")
            seen_checkpoint_names.add(name)
            if name.startswith(_UNSUPPORTED_WEIGHT_PREFIXES):
                continue
            if (
                name not in expected
                and name not in packed_source_keys
                and is_omitted_shared_kv(name)
            ):
                continue
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
            try:
                if name in expected:
                    target = expected_state[name]
                    if tensor.shape != target.shape:
                        raise ValueError(
                            f"Gemma checkpoint tensor {name!r} has shape "
                            f"{tuple(tensor.shape)}, expected {tuple(target.shape)}"
                        )
                    with torch.no_grad():
                        target.copy_(tensor)
                    loaded.add(name)
                else:
                    pending_packed[name] = tensor
            finally:
                del tensor
        bind_ready_packed(require_complete=False)

    bind_ready_packed(require_complete=True)

    embedding_key = "model.language_model.embed_tokens.weight"
    if "lm_head.weight" in expected and "lm_head.weight" not in loaded:
        if embedding_key not in loaded:
            raise KeyError(
                "tied Gemma checkpoint is missing its text embedding"
            )
        with torch.no_grad():
            expected_state["lm_head.weight"].copy_(expected_state[embedding_key])
        loaded.add("lm_head.weight")

    if pending_packed:
        raise RuntimeError(
            "Gemma checkpoint contains unexpected tensors: "
            f"{sorted(pending_packed)[:10]}"
        )
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Gemma checkpoint is missing tensors: {missing[:10]}")


def load_model(
    source: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
) -> Gemma4InferenceModel:
    snapshot = _snapshot(source, revision=revision)
    config_path = snapshot / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = Gemma4Config.from_dict(json.load(handle))

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)

    load_weights(snapshot, model)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    return model.to(device=device).eval()


__all__ = [
    "load_model",
    "load_weights",
]
