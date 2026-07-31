"""Direct Gemma 4 config and weight loading without Transformers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from kestrel.runtime.bounded_projection import (
    bind_declared_packed_bounded_projections,
)

from .config import Gemma4Config, parse_gemma4_config
from .model import Gemma4InferenceModel


_UNSUPPORTED_WEIGHT_PREFIXES = (
    "model.audio_tower.",
    "model.embed_audio.",
)


def load_config(repo_id: str) -> Gemma4Config:
    config_path = hf_hub_download(repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        return parse_gemma4_config(json.load(handle))


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

    bind_declared_packed_bounded_projections(model, state_dict)
    expected_keys = set(model.state_dict())
    projection_suffixes = (
        ("gate_proj.weight", "up_proj.weight", "gate_up_proj.weight"),
        (
            "gate_proj.linear.weight",
            "up_proj.linear.weight",
            "gate_up_proj.linear.weight",
        ),
    )
    for gate_key in list(state_dict):
        for gate_suffix, up_suffix, fused_suffix in projection_suffixes:
            if not gate_key.endswith(f".mlp.{gate_suffix}"):
                continue
            prefix = gate_key[: -len(gate_suffix)]
            up_key = prefix + up_suffix
            fused_key = prefix + fused_suffix
            if up_key not in state_dict or fused_key not in expected_keys:
                break
            state_dict[fused_key] = torch.cat(
                (state_dict.pop(gate_key), state_dict.pop(up_key)),
                dim=0,
            )

            if ".linear." in gate_suffix:
                for bound in ("input_min", "input_max", "output_min", "output_max"):
                    gate_bound = prefix + "gate_proj." + bound
                    up_bound = prefix + "up_proj." + bound
                    fused_bound = prefix + "gate_up_proj." + bound
                    if fused_bound not in expected_keys:
                        continue
                    if gate_bound not in state_dict or up_bound not in state_dict:
                        raise KeyError(
                            f"fused clipped projection requires {gate_bound!r} and "
                            f"{up_bound!r}"
                        )
                    if not torch.equal(state_dict[gate_bound], state_dict[up_bound]):
                        raise ValueError(
                            f"cannot fuse projections with different {bound} bounds: "
                            f"{gate_bound!r} vs {up_bound!r}"
                        )
                    state_dict[fused_bound] = state_dict.pop(gate_bound)
                    state_dict.pop(up_bound)
            break

    # Published checkpoints retain K/V tensors for layers whose config reuses
    # an earlier layer's cache, while tied checkpoints omit ``lm_head.weight``.
    model.load_state_dict(state_dict, strict=False)


def load_model(
    repo_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Gemma4InferenceModel:
    config = load_config(repo_id)

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
    "load_config",
    "load_model",
    "load_weights",
]
