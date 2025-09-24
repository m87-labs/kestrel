"""Utilities to load Moondream text weights into the self-contained model.

Adapted from the Moondream project (Apache-2.0).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List

import safetensors
import torch
import torch.nn as nn


@contextmanager
def safetensors_open(path: str):
    with safetensors.safe_open(path, framework="pt") as f:
        def get_tensor(name: str) -> torch.Tensor:
            return f.get_tensor(name)

        def get_keys() -> List[str]:
            return f.keys()

        get_tensor.keys = get_keys  # type: ignore[attr-defined]
        yield get_tensor


def _assign_text_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
    text = model.text

    weight_map: Dict[str, torch.Tensor] = {
        "text_model.transformer.embd.wte.weight": text.wte,
        "text_model.lm_head.ln.weight": text["post_ln"].weight,
        "text_model.lm_head.ln.bias": text["post_ln"].bias,
        "text_model.lm_head.linear.weight": text["lm_head"].weight,
        "text_model.lm_head.linear.bias": text["lm_head"].bias,
    }

    for i, block in enumerate(text["blocks"]):
        prefix = f"text_model.transformer.h.{i}"
        is_moe = hasattr(block.mlp, "router")
        weight_map.update(
            {
                f"{prefix}.ln.weight": block["ln"].weight,
                f"{prefix}.ln.bias": block["ln"].bias,
                f"{prefix}.mixer.Wqkv.weight": block["attn"]["qkv"].weight,
                f"{prefix}.mixer.Wqkv.bias": block["attn"]["qkv"].bias,
                f"{prefix}.mixer.out_proj.weight": block["attn"]["proj"].weight,
                f"{prefix}.mixer.out_proj.bias": block["attn"]["proj"].bias,
                f"{prefix}.tau_wq": block["attn"]["tau"]["wq"],
                f"{prefix}.tau_wv": block["attn"]["tau"]["wv"],
                f"{prefix}.tau_alpha": block["attn"]["tau"]["alpha"],
            }
        )
        if is_moe:
            weight_map.update(
                {
                    f"{prefix}.gate.weight": block["mlp"]["router"].weight,
                    f"{prefix}.gate.bias": block["mlp"]["router"].bias,
                    f"{prefix}.mlp.experts.weight": block["mlp"]["fc1"].weight,
                    f"{prefix}.mlp.output_experts.weight": block["mlp"]["fc2"].weight,
                }
            )
        else:
            weight_map.update(
                {
                    f"{prefix}.mlp.fc1.weight": block["mlp"]["fc1"].weight,
                    f"{prefix}.mlp.fc1.bias": block["mlp"]["fc1"].bias,
                    f"{prefix}.mlp.fc2.weight": block["mlp"]["fc2"].weight,
                    f"{prefix}.mlp.fc2.bias": block["mlp"]["fc2"].bias,
                }
            )

    for key, tensor in weight_map.items():
        tensor.data.copy_(get_tensor(key))

    for param in text.parameters():
        param.data = param.data.contiguous()


def load_text_weights(path: str, model: nn.Module) -> None:
    target_dtype = next(model.text.parameters()).dtype
    if path.endswith(".safetensors"):
        with safetensors_open(path) as get_tensor:
            name_map = {k.replace("._orig_mod", ""): k for k in get_tensor.keys()}
            _assign_text_weights(lambda x: get_tensor(name_map[x]).to(target_dtype), model)
    else:
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        tensors = {k.replace("._orig_mod", ""): v.to(target_dtype) for k, v in tensors.items()}
        _assign_text_weights(lambda x: tensors[x], model)


__all__ = ["load_text_weights"]
