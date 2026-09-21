"""Direct loading for the pinned Parakeet TDT checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from kestrel.models.asr.checkpoint import resolve_checkpoint

from .config import ParakeetTdtConfig
from .model import ParakeetTdt
from .tokenizer import ParakeetTokenizer
from .vad import VAD_HEAD_PREFIX


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
_FILES = ("config.json", "tokenizer.json", "model.safetensors")
# The ternary student distilled from MODEL_ID (thrush), published at TERNARY_MODEL_ID: config.json and
# tokenizer.json as usual, a ``model.safetensors`` with the encoder projections as base-3 packed codes (1.6 bits
# per weight, decoded to the kernels' two-bit layout at load) and the ``ternary.json`` manifest written by thrush's
# ``scripts/export_ternary.py --names hf``. That manifest is the only thing that marks a checkpoint as ternary.
TERNARY_MODEL_ID = "moondream/parakeet-redux"
EXPORT_FORMAT = "thrush-ternary-v2"  # base-3 packed codes on disk; see ``unpack_export``
TERNARY_REVISION = "af60db939ebab3ca8b95b5983174e669599a2352"  # p0g-p1a-warp5-2way in thrush-ternary-v2 (base-3), private until Photon ships
_MANIFEST = "ternary.json"
_QKV = ("q_proj", "k_proj", "v_proj")
_REL = "relative_k_proj"


@dataclass(frozen=True, slots=True)
class LoadedParakeetTdt:
    model: ParakeetTdt
    tokenizer: ParakeetTokenizer


def _quantized_modules(manifest: Path) -> tuple[dict, ...]:
    """The manifest's quantized-module entries: name, shape, group size and bias per ternary layer."""
    export = json.loads(manifest.read_text())
    if export.get("format") != EXPORT_FORMAT or export.get("names") != "hf":
        raise ValueError(
            f"expected a {EXPORT_FORMAT} export with HF tensor names (export_ternary.py --names hf), got "
            f"format={export.get('format')!r} names={export.get('names')!r}"
        )
    modules = tuple(export["quantized_modules"])
    names = [entry["name"] for entry in modules]
    if len(names) != len(set(names)):
        raise ValueError("the ternary manifest contains duplicate module names")
    return modules


_POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)


def _base3_digits() -> torch.Tensor:
    """uint8 ``[256, 5]``: the five base-3 digits of every byte value, least significant first."""
    return ((torch.arange(256, dtype=torch.int16).unsqueeze(1) // _POW3) % 3).to(torch.uint8)


def unpack_export(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    """The export's base-3 codes -> the kernels' two-bit layout.

    On disk (thrush ``export_ternary.py``, ``thrush-ternary-v2``) a quantized row is ``ceil(in/5)`` bytes of five
    base-3 digits each, element ``i`` being digit ``i % 5`` of byte ``i // 5`` (least significant first), code
    {0,1,2} = {-1,0,+1} — 1.6 bits per weight. In memory the kernels take ``uint8 [out, in/4]`` with four
    two-bit codes per byte (``kestrel_kernels.ternary``); the conversion is a table lookup per byte and runs once,
    at load.
    """
    from kestrel_kernels.ternary import pack_codes

    if qweight.dtype != torch.uint8 or qweight.shape[1] * 5 < in_features:
        raise ValueError(f"qweight must be uint8 [out, ceil(in/5)], got {tuple(qweight.shape)} for in={in_features}")
    codes = _base3_digits()[qweight.long()].reshape(qweight.shape[0], -1)[:, :in_features]
    return pack_codes(codes.to(torch.int8) - 1)


def ternarize(model: ParakeetTdt, quantized: tuple[dict, ...]) -> None:
    """Replace the manifest's quantized modules by ternary layers, in place (on the meta device, before loading).

    Every quantized module is a linear: the encoder applies its 1x1 convolutions in the linear layout too. The
    export keeps the attention projections separate, so a block's q/k/v triple becomes the model's one fused
    ``qkv_proj`` — ``RelativeAttention._load_from_state_dict`` concatenates their rows as it does for fp weights.
    The blocks' relative-position projections fuse the same way, but across blocks rather than within one:
    they do not depend on the activations, so the encoder holds a single layer for all of them.
    """
    from kestrel_kernels.ternary import TernaryLinear

    def layer(entry: dict, out_features: int | None = None) -> TernaryLinear:
        return TernaryLinear(
            entry["in_features"],
            entry["out_features"] if out_features is None else out_features,
            group_size=entry["group_size"],
            bias=entry["has_bias"],
        )

    def validate_fused(entries: tuple[dict, ...], label: str) -> None:
        for field in ("in_features", "out_features", "group_size", "has_bias"):
            values = {entry[field] for entry in entries}
            if len(values) != 1:
                raise ValueError(f"{label}: fused projections disagree on {field}")
        if entries[0]["has_bias"]:
            raise ValueError(f"{label}: fused projection biases are not supported")

    fused: dict[str, dict[str, dict]] = {}
    relative: list[dict] = []
    for entry in quantized:
        parent, _, attr = entry["name"].rpartition(".")
        if attr == _REL and hasattr(model.encoder, _REL):
            relative.append(entry)
        elif attr in _QKV and hasattr(model.get_submodule(parent), "qkv_proj"):
            fused.setdefault(parent, {})[attr] = entry
        else:
            model.set_submodule(entry["name"], layer(entry))
    for parent, parts in fused.items():
        if set(parts) != set(_QKV):
            raise ValueError(f"{parent}: the export quantizes {sorted(parts)}, not all of q/k/v")
        ordered = tuple(parts[attr] for attr in _QKV)
        validate_fused(ordered, parent)
        out_features = sum(entry["out_features"] for entry in ordered)
        model.set_submodule(f"{parent}.qkv_proj", layer(ordered[0], out_features))
    if relative:
        expected = {
            f"encoder.layers.{index}.self_attn.{_REL}"
            for index in range(len(model.encoder.layers))
        }
        names = {entry["name"] for entry in relative}
        if len(relative) != len(expected) or names != expected:
            raise ValueError(
                "the export's relative-position projections do not match the encoder blocks"
            )
        validate_fused(tuple(relative), f"encoder.{_REL}")
        model.set_submodule(
            f"encoder.{_REL}",
            layer(relative[0], sum(entry["out_features"] for entry in relative)),
        )


def load_parakeet_tdt(
    checkpoint: str | Path = MODEL_ID,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
) -> LoadedParakeetTdt:
    """Load the pinned fp checkpoint, or the ternary student when the checkpoint carries a ``ternary.json``.

    The ternary variant loads the packed export with ``strict=True`` and then materializes the resident
    weight form for the device it is on: the packed codes on the CPU and on Metal, the dense weight in
    ``dtype`` on CUDA (the small download at the fp model's speed). There is nothing to select; see
    ``kestrel_kernels.ternary``. Activations stay in ``dtype``; nothing else is quantized at run time.
    """
    from safetensors.torch import load_file

    ternary_repo = str(checkpoint) == TERNARY_MODEL_ID
    root = resolve_checkpoint(
        checkpoint,
        revision=revision or (TERNARY_REVISION if ternary_repo else REVISION),
        filenames=_FILES + ((_MANIFEST,) if ternary_repo else ()),
        local_files_only=local_files_only,
    )
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    tokenizer = ParakeetTokenizer(root / "tokenizer.json")
    if (
        tokenizer.blank_token_id != config.blank_token_id
        or tokenizer.pad_token_id != config.pad_token_id
    ):
        raise ValueError("Parakeet tokenizer and model special tokens disagree")
    manifest = root / _MANIFEST
    state = load_file(str(root / "model.safetensors"), device="cpu")
    with torch.device("meta"):
        model = ParakeetTdt(config)
        if manifest.exists():
            quantized = _quantized_modules(manifest)
            ternarize(model, quantized)
            model.is_ternary = True
        # A capability of the weights, not a configured option: a checkpoint
        # that ships a speech head segments its own long audio with it, and one
        # that does not falls back to the energy detector. Checked on the fp
        # and the ternary export alike, before the strict load.
        if any(key.startswith(VAD_HEAD_PREFIX) for key in state):
            model.attach_vad_head()
    if manifest.exists():  # the export's base-3 rows become the kernels' two-bit rows, once, here (not on meta)
        for entry in quantized:
            key = f"{entry['name']}.qweight"
            state[key] = unpack_export(state[key], entry["in_features"])
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_nonpersistent_buffers()
    if manifest.exists():
        from kestrel_kernels.ternary import materialize_ternary

        # Between the two casts: the cache dequantizes the packed codes on the target device and keeps its
        # scales in fp32, so it must exist before the module cast reaches the dequantized weights.
        model.to(device=device)
        materialize_ternary(model, dtype)
        model.to(dtype=dtype)
    else:
        model.to(device=device, dtype=dtype)
    return LoadedParakeetTdt(model.eval(), tokenizer)


__all__ = [
    "LoadedParakeetTdt",
    "MODEL_ID",
    "REVISION",
    "TERNARY_MODEL_ID",
    "TERNARY_REVISION",
    "load_parakeet_tdt",
    "ternarize",
    "unpack_export",
]
