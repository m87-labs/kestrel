"""Direct Qwen 3.5 config/weight loading without Transformers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

from .qwen_config import Qwen3_5Config
from .qwen_model import Qwen3_5ForConditionalGeneration


_GDN_IN_PROJ_PARTS = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_b.weight",
    "in_proj_a.weight",
)
_ATTN_QKV_WEIGHT_PARTS = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
_ATTN_QKV_BIAS_PARTS = ("q_proj.bias", "k_proj.bias", "v_proj.bias")
_MLP_GATE_UP_WEIGHT_PARTS = ("gate_proj.weight", "up_proj.weight")
_FUSED_PROJECTION_SPECS = (
    ("in_proj.weight", _GDN_IN_PROJ_PARTS),
    ("qkv_proj.weight", _ATTN_QKV_WEIGHT_PARTS),
    ("qkv_proj.bias", _ATTN_QKV_BIAS_PARTS),
    ("gate_up_proj.weight", _MLP_GATE_UP_WEIGHT_PARTS),
)
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_FP8_TEXT_DENSE_WEIGHT_RE = re.compile(
    r"^(?:model\.language_model\.)?layers\.\d+\."
    r"(?:linear_attn|self_attn|mlp)\."
    r"(?!experts\.)"
    r".+\.weight$"
)


def _torch_device_arg(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    return device.type


def _ignored_missing_key(key: str) -> bool:
    return (
        key == "lm_head.weight"
        or key.endswith("rotary_emb.inv_freq")
        or key.endswith("rotary_emb.original_inv_freq")
        or key.endswith("visual.rotary_pos_emb.inv_freq")
    )


def _fused_projection_keys(key: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    matches = []
    for fused_suffix, parts in _FUSED_PROJECTION_SPECS:
        for suffix in parts:
            if key.endswith(f".{suffix}"):
                prefix = key[: -len(suffix)]
                matches.append((f"{prefix}{fused_suffix}", suffix, parts))
    return tuple(matches)


def _expert_projection_key(key: str) -> tuple[str, int, str] | None:
    match = _EXPERT_WEIGHT_RE.match(key)
    if match is None:
        return None
    prefix = match.group("prefix")
    expert_idx = int(match.group("expert"))
    proj = match.group("proj")
    target = f"{prefix}.down_proj" if proj == "down_proj" else f"{prefix}.gate_up_proj"
    return target, expert_idx, f"{proj}.weight"


def _qwen_rms_norm_weight_keys(model: torch.nn.Module) -> set[str]:
    return {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleDict)
        and hasattr(module, "eps")
        and hasattr(module, "weight")
    }


def _loadable_tensor(
    key: str,
    value: torch.Tensor,
    qwen_rms_norm_weight_keys: set[str],
    expected_shape: torch.Size,
    scale_inv: torch.Tensor | None = None,
    *,
    allow_fp8_dequant: bool = False,
) -> torch.Tensor:
    if _is_float8_tensor(value):
        if not allow_fp8_dequant:
            raise ValueError(
                f"Qwen FP8 weight {key!r} reached a non-FP8 target; add a native FP8 "
                "module with a matching weight_scale_inv buffer or an explicit "
                "BF16 fallback entry"
            )
        value = _dequantize_fp8_weight(value, scale_inv, expected_shape, key=key)
    if key.endswith("visual.patch_embed.proj.weight") and value.ndim == 5:
        value = value.reshape(value.shape[0], -1)
        if value.shape != expected_shape:
            raise ValueError(
                "Qwen vision patch projection weight has unexpected shape after "
                f"flattening: got {tuple(value.shape)}, expected {tuple(expected_shape)}"
            )
    if key in qwen_rms_norm_weight_keys:
        return value.to(torch.float32) + 1.0
    return value


def _allow_fp8_dequant_fallback(key: str) -> bool:
    return _FP8_TEXT_DENSE_WEIGHT_RE.match(key) is not None


def _is_float8_tensor(value: torch.Tensor) -> bool:
    return value.dtype in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }


def _dequantize_fp8_weight(
    value: torch.Tensor,
    scale_inv: torch.Tensor | None,
    expected_shape: torch.Size,
    *,
    key: str,
) -> torch.Tensor:
    if scale_inv is None:
        raise ValueError(f"Qwen FP8 weight {key!r} is missing weight_scale_inv")
    if value.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            "Qwen FP8 block weight dequantization expects 2D weight/scale tensors, "
            f"got weight={tuple(value.shape)} scale={tuple(scale_inv.shape)}"
        )
    if value.shape != expected_shape:
        raise ValueError(
            f"Qwen FP8 weight {key!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    block_m = 128
    block_n = 128
    expected_scale_shape = (
        (value.shape[0] + block_m - 1) // block_m,
        (value.shape[1] + block_n - 1) // block_n,
    )
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Qwen FP8 weight scale {_scale_inv_key(key)!r} has shape "
            f"{tuple(scale_inv.shape)}, expected {expected_scale_shape}"
        )
    expanded_scale = scale_inv.repeat_interleave(block_m, dim=0).repeat_interleave(
        block_n,
        dim=1,
    )
    expanded_scale = expanded_scale[: value.shape[0], : value.shape[1]]
    return (value.to(torch.float32) * expanded_scale.to(torch.float32)).to(
        torch.bfloat16
    )


def _load_fp8_expert_weight_and_scale(
    value: torch.Tensor,
    scale_inv: torch.Tensor | None,
    expected_shape: torch.Size,
    *,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_float8_tensor(value):
        raise ValueError(f"Qwen FP8 expert weight {key!r} has dtype {value.dtype}")
    if scale_inv is None:
        raise ValueError(f"Qwen FP8 expert weight {key!r} is missing weight_scale_inv")
    if value.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            "Qwen FP8 expert loading expects 2D weight/scale tensors, "
            f"got weight={tuple(value.shape)} scale={tuple(scale_inv.shape)}"
        )
    if value.shape != expected_shape:
        raise ValueError(
            f"Qwen FP8 expert weight {key!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    return value.view(torch.uint8).contiguous(), scale_inv.to(torch.float32).contiguous()


def _scale_inv_key(weight_key: str) -> str:
    return f"{weight_key}_scale_inv"


def _load_scale_inv_tensors(
    shard_paths: list[tuple[str, str]],
    *,
    device_arg: str,
) -> dict[str, torch.Tensor]:
    scales: dict[str, torch.Tensor] = {}
    for _shard_name, shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device=device_arg) as handle:
            for key in handle.keys():
                if key.startswith("mtp.") or not key.endswith(".weight_scale_inv"):
                    continue
                scales[key] = handle.get_tensor(key)
    return scales


def _interleave_gate_up_weight(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(
            f"gate/up weights must have matching shapes, got {gate.shape} and {up.shape}"
        )
    if gate.shape[0] % 8 != 0:
        raise ValueError(
            f"gate/up output dimension must be divisible by 8, got {gate.shape[0]}"
        )
    gate_blocks = gate.reshape(gate.shape[0] // 8, 8, *gate.shape[1:])
    up_blocks = up.reshape(up.shape[0] // 8, 8, *up.shape[1:])
    return torch.stack((gate_blocks, up_blocks), dim=1).reshape(
        2 * gate.shape[0],
        *gate.shape[1:],
    )


def _load_ready_fused_projections(
    model: torch.nn.Module,
    pending: dict[str, dict[str, torch.Tensor]],
    pending_parts: dict[str, tuple[str, ...]],
    loaded_keys: set[str],
) -> None:
    compatible_state = {}
    for fused_key, part_tensors in list(pending.items()):
        parts = pending_parts[fused_key]
        if not all(part in part_tensors for part in parts):
            continue
        if parts == _MLP_GATE_UP_WEIGHT_PARTS:
            compatible_state[fused_key] = _interleave_gate_up_weight(
                part_tensors["gate_proj.weight"],
                part_tensors["up_proj.weight"],
            )
        else:
            compatible_state[fused_key] = torch.cat(
                [part_tensors[part] for part in parts],
                dim=0,
            )
        loaded_keys.add(fused_key)
        del pending[fused_key]
        del pending_parts[fused_key]

    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)


def _load_ready_packed_experts(
    model: torch.nn.Module,
    pending: dict[str, dict[int, dict[str, torch.Tensor]]],
    loaded_keys: set[str],
) -> None:
    compatible_state = {}
    expected_state = model.state_dict()
    for target_key, expert_tensors in list(pending.items()):
        if target_key not in expected_state:
            continue
        expected_shape = expected_state[target_key].shape
        is_fp8 = expected_state[target_key].dtype == torch.uint8
        scale_key = f"{target_key}_scale"
        num_experts = int(expected_shape[0])
        if len(expert_tensors) != num_experts:
            continue
        if target_key.endswith(".gate_up_proj"):
            if not all(
                "gate_proj.weight" in parts and "up_proj.weight" in parts
                for parts in expert_tensors.values()
            ):
                continue
            packed = [
                _interleave_gate_up_weight(
                    expert_tensors[idx]["gate_proj.weight"],
                    expert_tensors[idx]["up_proj.weight"],
                )
                for idx in range(num_experts)
            ]
            if is_fp8:
                if scale_key not in expected_state:
                    raise ValueError(f"missing FP8 expert scale buffer {scale_key}")
                if not all(
                    "gate_proj.weight_scale_inv" in parts
                    and "up_proj.weight_scale_inv" in parts
                    for parts in expert_tensors.values()
                ):
                    continue
                packed_scale = [
                    torch.stack(
                        (
                            expert_tensors[idx]["gate_proj.weight_scale_inv"],
                            expert_tensors[idx]["up_proj.weight_scale_inv"],
                        ),
                        dim=0,
                    )
                    for idx in range(num_experts)
                ]
        else:
            if not all("down_proj.weight" in parts for parts in expert_tensors.values()):
                continue
            packed = [expert_tensors[idx]["down_proj.weight"] for idx in range(num_experts)]
            if is_fp8:
                if scale_key not in expected_state:
                    raise ValueError(f"missing FP8 expert scale buffer {scale_key}")
                if not all(
                    "down_proj.weight_scale_inv" in parts
                    for parts in expert_tensors.values()
                ):
                    continue
                packed_scale = [
                    expert_tensors[idx]["down_proj.weight_scale_inv"]
                    for idx in range(num_experts)
                ]
        tensor = torch.stack(packed, dim=0)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"packed expert tensor {target_key} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        compatible_state[target_key] = tensor
        loaded_keys.add(target_key)
        if is_fp8:
            scale_tensor = torch.stack(packed_scale, dim=0)
            expected_scale_shape = expected_state[scale_key].shape
            if scale_tensor.shape != expected_scale_shape:
                raise ValueError(
                    f"packed expert scale {scale_key} has shape "
                    f"{tuple(scale_tensor.shape)}, expected {tuple(expected_scale_shape)}"
                )
            compatible_state[scale_key] = scale_tensor
            loaded_keys.add(scale_key)
        del pending[target_key]

    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)


def _load_sharded_safetensors(
    model: torch.nn.Module,
    source: str | Path,
    shard_names: list[str],
    *,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    loaded_keys: set[str] = set()
    pending_fused: dict[str, dict[str, torch.Tensor]] = {}
    pending_fused_parts: dict[str, tuple[str, ...]] = {}
    pending_experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    qwen_rms_norm_weight_keys = _qwen_rms_norm_weight_keys(model)
    unexpected: list[str] = []
    device_arg = _torch_device_arg(device)
    shard_paths = [
        (shard_name, _resolve_checkpoint_file(source, shard_name))
        for shard_name in shard_names
    ]
    scale_inv_by_key = _load_scale_inv_tensors(shard_paths, device_arg=device_arg)

    for _shard_name, shard_path in shard_paths:
        shard_state = load_file(shard_path, device=device_arg)
        compatible_state = {}
        for key, value in shard_state.items():
            if key.startswith("mtp."):
                continue
            if key.endswith(".weight_scale_inv"):
                continue
            if key in expected_keys:
                scale_key = _scale_inv_key(key)
                compatible_state[key] = _loadable_tensor(
                    key,
                    value,
                    qwen_rms_norm_weight_keys,
                    expected_state[key].shape,
                    scale_inv_by_key.get(scale_key),
                    allow_fp8_dequant=_allow_fp8_dequant_fallback(key),
                )
                loaded_keys.add(key)
                continue
            fused_parts = _fused_projection_keys(key)
            fused_handled = False
            for fused_key, part, parts in fused_parts:
                if fused_key in expected_keys:
                    pending_fused.setdefault(fused_key, {})[part] = (
                        _loadable_tensor(
                            key,
                            value,
                            set(),
                            value.shape,
                            scale_inv_by_key.get(_scale_inv_key(key)),
                            allow_fp8_dequant=_allow_fp8_dequant_fallback(key),
                        )
                    )
                    pending_fused_parts[fused_key] = parts
                    fused_handled = True
                    break
            if fused_handled:
                continue
            expert_part = _expert_projection_key(key)
            if expert_part is not None:
                target_key, expert_idx, part = expert_part
                if target_key in expected_keys:
                    if expected_state[target_key].dtype == torch.uint8:
                        target_shape = expected_state[target_key].shape
                        if target_key.endswith("gate_up_proj"):
                            expected_part_shape = torch.Size(
                                (target_shape[1] // 2, target_shape[2])
                            )
                        else:
                            expected_part_shape = torch.Size(
                                (target_shape[1], target_shape[2])
                            )
                        weight, scale = _load_fp8_expert_weight_and_scale(
                            value,
                            scale_inv_by_key.get(_scale_inv_key(key)),
                            expected_part_shape,
                            key=key,
                        )
                        expert_entry = pending_experts.setdefault(
                            target_key, {}
                        ).setdefault(expert_idx, {})
                        expert_entry[part] = weight
                        expert_entry[f"{part}_scale_inv"] = scale
                        continue
                    pending_experts.setdefault(target_key, {}).setdefault(
                        expert_idx, {}
                    )[part] = _loadable_tensor(
                        key,
                        value,
                        set(),
                        value.shape,
                        scale_inv_by_key.get(_scale_inv_key(key)),
                        allow_fp8_dequant=_allow_fp8_dequant_fallback(key),
                    )
                    continue
            unexpected.append(key)
        model.load_state_dict(compatible_state, strict=False)
        _load_ready_fused_projections(
            model,
            pending_fused,
            pending_fused_parts,
            loaded_keys,
        )
        _load_ready_packed_experts(model, pending_experts, loaded_keys)
        del shard_state, compatible_state
    _load_ready_fused_projections(
        model,
        pending_fused,
        pending_fused_parts,
        loaded_keys,
    )
    _load_ready_packed_experts(model, pending_experts, loaded_keys)

    missing = [
        key
        for key in sorted(expected_keys - loaded_keys)
        if not _ignored_missing_key(key)
    ]
    return missing, sorted(unexpected)


def _resolve_checkpoint_file(source: str | Path, filename: str) -> str:
    if isinstance(source, Path):
        root = source if source.is_dir() else source.parent
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Qwen checkpoint is missing required file {path}"
            )
        return str(path)
    return hf_hub_download(source, filename)


def load_qwen35_model(
    source: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3_5ForConditionalGeneration:
    config_path = _resolve_checkpoint_file(source, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data: dict[str, Any] = json.load(handle)
    config = Qwen3_5Config.from_dict(config_data)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Qwen3_5ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(old_dtype)

    index_path = _resolve_checkpoint_file(
        source, "model.safetensors.index.json"
    )
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    shard_names = sorted(set(index["weight_map"].values()))

    missing, unexpected = _load_sharded_safetensors(
        model,
        source,
        shard_names,
        device=device,
    )
    if unexpected or missing:
        raise RuntimeError(
            "Failed to load Qwen 3.5 weights: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if config.text_config.tie_word_embeddings:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    return model.to(device=device).eval()


__all__ = ["load_qwen35_model"]
