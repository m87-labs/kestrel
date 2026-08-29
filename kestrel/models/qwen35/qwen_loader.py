"""Direct Qwen 3.5 config/weight loading without Transformers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from kestrel.ops.rotary import default_inv_freq

from .qwen_config import Qwen3_5Config
from .qwen_model import Qwen3_5ForConditionalGeneration, Qwen3_5RMSNormGated


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
_DEQUANT_CHUNK_ROWS = 1024


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


def _qwen_gdn_norm_weight_keys(model: torch.nn.Module) -> set[str]:
    return {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5RMSNormGated)
    }


def _loadable_tensor(
    key: str,
    value: torch.Tensor,
    qwen_rms_norm_weight_keys: set[str],
    expected_shape: torch.Size,
    scale_inv: torch.Tensor | None = None,
    *,
    convert_fp8_to_bf16: bool = False,
    exact_fp32_weight_keys: set[str] | None = None,
) -> torch.Tensor:
    if exact_fp32_weight_keys is not None and key in exact_fp32_weight_keys:
        if value.dtype != torch.float32:
            raise ValueError(
                f"Qwen GDN norm weight {key!r} requires an FP32 checkpoint tensor, "
                f"got {value.dtype}"
            )
        if value.shape != expected_shape:
            raise ValueError(
                f"Qwen GDN norm weight {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        return value
    if _is_float8_tensor(value):
        if not convert_fp8_to_bf16:
            raise ValueError(
                f"Qwen FP8 weight {key!r} reached a non-FP8 target; add a native FP8 "
                "module with a matching weight_scale_inv buffer or declare BF16 "
                "storage conversion"
            )
        value = _dequantize_fp8_weight(value, scale_inv, expected_shape, key=key)
    if key.endswith("visual.patch_embed.proj.weight") and value.ndim == 5:
        value = value.reshape(value.shape[0], -1)
    if value.shape != expected_shape:
        raise ValueError(
            f"Qwen weight {key!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if key in qwen_rms_norm_weight_keys:
        return value.to(torch.float32) + 1.0
    return value


def _stores_checkpoint_fp8_as_bf16(key: str) -> bool:
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
    output = torch.empty(value.shape, dtype=torch.bfloat16, device=value.device)
    scale_fp32 = scale_inv.to(torch.float32)
    for row_start in range(0, value.shape[0], _DEQUANT_CHUNK_ROWS):
        row_end = min(row_start + _DEQUANT_CHUNK_ROWS, value.shape[0])
        scale_row_start = row_start // block_m
        scale_row_end = (row_end + block_m - 1) // block_m
        expanded_scale = scale_fp32[
            scale_row_start:scale_row_end
        ].repeat_interleave(block_m, dim=0)
        expanded_scale = expanded_scale[
            : row_end - row_start
        ].repeat_interleave(block_n, dim=1)
        expanded_scale = expanded_scale[:, : value.shape[1]]
        chunk = value[row_start:row_end].to(torch.float32)
        chunk.mul_(expanded_scale)
        output[row_start:row_end].copy_(chunk)
    return output


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


def _copy_fused_projection_part(
    expected_state: dict[str, torch.Tensor],
    tensor_shapes: dict[str, torch.Size],
    *,
    checkpoint_key: str,
    fused_key: str,
    part: str,
    parts: tuple[str, ...],
    value: torch.Tensor,
    loaded_parts: dict[str, set[str]],
    loaded_keys: set[str],
) -> None:
    target = expected_state[fused_key]
    if value.shape != tensor_shapes[checkpoint_key]:
        raise ValueError(
            f"checkpoint tensor {checkpoint_key!r} changed shape while loading"
        )
    seen = loaded_parts.setdefault(fused_key, set())
    if part in seen:
        raise ValueError(f"duplicate fused projection part {checkpoint_key!r}")
    if parts == _MLP_GATE_UP_WEIGHT_PARTS:
        expected_target_shape = (2 * value.shape[0], *value.shape[1:])
        if target.shape != expected_target_shape:
            raise ValueError(
                f"fused target {fused_key!r} has shape {tuple(target.shape)}, "
                f"expected {expected_target_shape}"
            )
        slot = parts.index(part)
        target_blocks = target.reshape(
            target.shape[0] // 16,
            2,
            8,
            *target.shape[1:],
        )
        source_blocks = value.reshape(
            value.shape[0] // 8,
            8,
            *value.shape[1:],
        )
        with torch.no_grad():
            target_blocks[:, slot].copy_(source_blocks)
    else:
        prefix = checkpoint_key[: -len(part)]
        expected_target_shape = (
            sum(tensor_shapes[f"{prefix}{item}"][0] for item in parts),
            *value.shape[1:],
        )
        if target.shape != expected_target_shape:
            raise ValueError(
                f"fused target {fused_key!r} has shape {tuple(target.shape)}, "
                f"expected {expected_target_shape}"
            )
        offset = sum(
            tensor_shapes[f"{prefix}{previous}"][0]
            for previous in parts[: parts.index(part)]
        )
        with torch.no_grad():
            target.narrow(0, offset, value.shape[0]).copy_(value)

    seen.add(part)
    if seen == set(parts):
        loaded_keys.add(fused_key)


def _copy_bf16_expert_part(
    expected_state: dict[str, torch.Tensor],
    *,
    checkpoint_key: str,
    target_key: str,
    expert_idx: int,
    part: str,
    value: torch.Tensor,
    loaded_parts: dict[str, dict[int, set[str]]],
    loaded_keys: set[str],
) -> None:
    target = expected_state[target_key]
    if target.dtype != torch.bfloat16:
        raise ValueError(f"expert target {target_key!r} is not BF16")
    if not 0 <= expert_idx < target.shape[0]:
        raise ValueError(f"expert index out of range in {checkpoint_key!r}")
    seen = loaded_parts.setdefault(target_key, {}).setdefault(expert_idx, set())
    if part in seen:
        raise ValueError(f"duplicate expert projection part {checkpoint_key!r}")

    expert_target = target[expert_idx]
    if target_key.endswith(".gate_up_proj"):
        parts = _MLP_GATE_UP_WEIGHT_PARTS
        expected_shape = (
            expert_target.shape[0] // 2,
            *expert_target.shape[1:],
        )
        if value.shape != expected_shape:
            raise ValueError(
                f"expert projection {checkpoint_key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        slot = parts.index(part)
        target_blocks = expert_target.reshape(
            expert_target.shape[0] // 16,
            2,
            8,
            *expert_target.shape[1:],
        )
        source_blocks = value.reshape(
            value.shape[0] // 8,
            8,
            *value.shape[1:],
        )
        with torch.no_grad():
            target_blocks[:, slot].copy_(source_blocks)
    else:
        parts = ("down_proj.weight",)
        if value.shape != expert_target.shape:
            raise ValueError(
                f"expert projection {checkpoint_key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expert_target.shape)}"
            )
        with torch.no_grad():
            expert_target.copy_(value)

    seen.add(part)
    if len(loaded_parts[target_key]) == target.shape[0] and all(
        item == set(parts) for item in loaded_parts[target_key].values()
    ):
        loaded_keys.add(target_key)


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
    revision: str | None = None,
) -> tuple[list[str], list[str]]:
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    loaded_keys: set[str] = set()
    loaded_fused_parts: dict[str, set[str]] = {}
    loaded_expert_parts: dict[str, dict[int, set[str]]] = {}
    pending_experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    qwen_rms_norm_weight_keys = _qwen_rms_norm_weight_keys(model)
    qwen_gdn_norm_weight_keys = _qwen_gdn_norm_weight_keys(model)
    unexpected: list[str] = []
    device_arg = _torch_device_arg(device)
    shard_paths = [
        (
            shard_name,
            _resolve_checkpoint_file(source, shard_name, revision=revision),
        )
        for shard_name in shard_names
    ]
    tensor_shapes: dict[str, torch.Size] = {}
    for _shard_name, shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensor_shapes:
                    raise ValueError(f"duplicate Qwen checkpoint tensor {key!r}")
                tensor_shapes[key] = torch.Size(handle.get_slice(key).get_shape())
    scale_inv_by_key = _load_scale_inv_tensors(shard_paths, device_arg=device_arg)

    for _shard_name, shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device=device_arg) as handle:
            for key in handle.keys():
                if key.startswith("mtp."):
                    continue
                if key.endswith(".weight_scale_inv"):
                    continue
                value = handle.get_tensor(key)
                if key in expected_keys:
                    scale_key = _scale_inv_key(key)
                    loaded = _loadable_tensor(
                        key,
                        value,
                        qwen_rms_norm_weight_keys,
                        expected_state[key].shape,
                        scale_inv_by_key.get(scale_key),
                        convert_fp8_to_bf16=_stores_checkpoint_fp8_as_bf16(key),
                        exact_fp32_weight_keys=qwen_gdn_norm_weight_keys,
                    )
                    with torch.no_grad():
                        expected_state[key].copy_(loaded)
                    loaded_keys.add(key)
                    continue
                fused_parts = _fused_projection_keys(key)
                fused_handled = False
                for fused_key, part, parts in fused_parts:
                    if fused_key in expected_keys:
                        loaded = _loadable_tensor(
                            key,
                            value,
                            set(),
                            value.shape,
                            scale_inv_by_key.get(_scale_inv_key(key)),
                            convert_fp8_to_bf16=_stores_checkpoint_fp8_as_bf16(key),
                        )
                        _copy_fused_projection_part(
                            expected_state,
                            tensor_shapes,
                            checkpoint_key=key,
                            fused_key=fused_key,
                            part=part,
                            parts=parts,
                            value=loaded,
                            loaded_parts=loaded_fused_parts,
                            loaded_keys=loaded_keys,
                        )
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
                        loaded = _loadable_tensor(
                            key,
                            value,
                            set(),
                            value.shape,
                            scale_inv_by_key.get(_scale_inv_key(key)),
                            convert_fp8_to_bf16=_stores_checkpoint_fp8_as_bf16(key),
                        )
                        _copy_bf16_expert_part(
                            expected_state,
                            checkpoint_key=key,
                            target_key=target_key,
                            expert_idx=expert_idx,
                            part=part,
                            value=loaded,
                            loaded_parts=loaded_expert_parts,
                            loaded_keys=loaded_keys,
                        )
                        continue
                unexpected.append(key)
        _load_ready_packed_experts(model, pending_experts, loaded_keys)
    _load_ready_packed_experts(model, pending_experts, loaded_keys)

    missing = [
        key
        for key in sorted(expected_keys - loaded_keys)
        if not _ignored_missing_key(key)
    ]
    return missing, sorted(unexpected)


def _resolve_checkpoint_file(
    source: str | Path,
    filename: str,
    *,
    revision: str | None = None,
) -> str:
    if isinstance(source, Path):
        root = source if source.is_dir() else source.parent
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Qwen checkpoint is missing required file {path}"
            )
        return str(path)
    kwargs = {} if revision is None else {"revision": revision}
    return hf_hub_download(source, filename, **kwargs)


def _materialize_remaining_meta_tensors(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> None:
    parameter_replacements: dict[int, torch.nn.Parameter] = {}
    buffer_replacements: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is None or parameter.device.type != "meta":
                continue
            replacement = parameter_replacements.get(id(parameter))
            if replacement is None:
                replacement = torch.nn.Parameter(
                    torch.empty_like(parameter, device=device),
                    requires_grad=parameter.requires_grad,
                )
                parameter_replacements[id(parameter)] = replacement
            module._parameters[name] = replacement
        for name, buffer in tuple(module._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            if name in module._non_persistent_buffers_set:
                raise RuntimeError(
                    "Qwen load-time preparation left derived buffer "
                    f"{type(module).__name__}.{name} uninitialized"
                )
            replacement = buffer_replacements.get(id(buffer))
            if replacement is None:
                replacement = torch.empty_like(buffer, device=device)
                buffer_replacements[id(buffer)] = replacement
            module._buffers[name] = replacement


def _restore_rotary_buffers(
    model: Qwen3_5ForConditionalGeneration,
    config: Qwen3_5Config,
    *,
    device: torch.device,
) -> None:
    """Materialize checkpoint-independent RoPE tables after meta construction."""

    vision_rotary = model.model.visual.rotary_pos_emb
    vision_rotary.inv_freq = default_inv_freq(
        int(vision_rotary.inv_freq.numel()) * 2,
        10_000.0,
        device=device,
    )
    text_config = config.text_config
    model.model.language_model.rotary_emb.inv_freq = default_inv_freq(
        text_config.head_dim,
        text_config.rope_theta,
        partial_rotary_factor=text_config.partial_rotary_factor,
        device=device,
    )


def load_qwen35_model(
    source: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
    prepare_model: Callable[[torch.nn.Module], None] | None = None,
) -> Qwen3_5ForConditionalGeneration:
    config_path = _resolve_checkpoint_file(
        source, "config.json", revision=revision
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data: dict[str, Any] = json.load(handle)
    config = Qwen3_5Config.from_dict(config_data)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        construction_device = torch.device("meta") if prepare_model else device
        with torch.device(construction_device):
            model = Qwen3_5ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(old_dtype)
    if prepare_model is not None:
        prepare_model(model)
        _restore_rotary_buffers(model, config, device=device)
        _materialize_remaining_meta_tensors(model, device=device)

    index_path = _resolve_checkpoint_file(
        source, "model.safetensors.index.json", revision=revision
    )
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    shard_names = sorted(set(index["weight_map"].values()))

    missing, unexpected = _load_sharded_safetensors(
        model,
        source,
        shard_names,
        device=device,
        revision=revision,
    )
    if unexpected or missing:
        raise RuntimeError(
            "Failed to load Qwen 3.5 weights: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if config.text_config.tie_word_embeddings:
        embedding_weight = model.model.language_model.embed_tokens.weight
        with torch.no_grad():
            model.lm_head.weight.copy_(embedding_weight)
        model.lm_head.weight = embedding_weight
    model = model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return model


__all__ = ["load_qwen35_model"]
