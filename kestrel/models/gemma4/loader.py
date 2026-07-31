"""Direct Gemma 4 config and weight loading without Transformers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from kestrel.runtime.bounded_projection import (
    bind_declared_packed_projections,
)

from .config import Gemma4Config
from .model import Gemma4InferenceModel


_UNSUPPORTED_WEIGHT_PREFIXES = (
    "model.audio_tower.",
    "model.embed_audio.",
)


def load_weights(repo_id: str, model: torch.nn.Module) -> None:
    snapshot = Path(
        snapshot_download(
            repo_id,
            allow_patterns=[
                "*.safetensors",
                "model.safetensors.index.json",
            ],
        )
    )

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = snapshot / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith(_UNSUPPORTED_WEIGHT_PREFIXES):
                    continue
                state_dict[name] = handle.get_tensor(name)

    bind_declared_packed_projections(model, state_dict)

    expected = set(model.state_dict())
    embedding_key = "model.language_model.embed_tokens.weight"
    if "lm_head.weight" in expected and "lm_head.weight" not in state_dict:
        try:
            state_dict["lm_head.weight"] = state_dict[embedding_key]
        except KeyError as exc:
            raise KeyError(
                "tied Gemma checkpoint is missing its text embedding"
            ) from exc

    # Published checkpoints retain K/V parameters for layers whose config
    # explicitly reuses an earlier producer. The inference module omits those
    # dead parameters; discard only that exact declared family.
    shared_kv_members = ("k_proj", "v_proj", "k_norm", "v_norm")
    for name in tuple(state_dict):
        if name in expected:
            continue
        for member in shared_kv_members:
            marker = f".self_attn.{member}."
            if marker not in name:
                continue
            attention_path = name.partition(marker)[0] + ".self_attn"
            try:
                attention = model.get_submodule(attention_path)
            except AttributeError:
                break
            if not hasattr(attention, member):
                state_dict.pop(name)
            break

    model.load_state_dict(state_dict, strict=True)


def load_model(
    repo_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Gemma4InferenceModel:
    config_path = hf_hub_download(repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = Gemma4Config.from_dict(json.load(handle))

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)

    load_weights(repo_id, model)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    return model.to(device=device).eval()


__all__ = [
    "load_model",
    "load_weights",
]
