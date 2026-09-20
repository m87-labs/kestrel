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


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
_FILES = ("config.json", "tokenizer.json", "model.safetensors")
# The ternary (2-bit) student distilled from MODEL_ID (thrush), published at TERNARY_MODEL_ID: config.json and
# tokenizer.json as usual, a packed ``model.safetensors`` and the ``ternary.json`` manifest written by thrush's
# ``scripts/export_ternary.py --names hf``. That manifest is the only thing that marks a checkpoint as ternary.
TERNARY_MODEL_ID = "moondream/parakeet-redux"
TERNARY_REVISION = "a0639ca117380949d782a505569d637d180f5378"  # release weights, private until Photon ships
_MANIFEST = "ternary.json"
_QKV = ("q_proj", "k_proj", "v_proj")


@dataclass(frozen=True, slots=True)
class LoadedParakeetTdt:
    model: ParakeetTdt
    tokenizer: ParakeetTokenizer


def _quantized_modules(manifest: Path) -> tuple[dict, ...]:
    """The manifest's quantized-module entries: name, shape, group size and bias per ternary layer."""
    export = json.loads(manifest.read_text())
    if export.get("format") != "thrush-ternary-v1" or export.get("names") != "hf":
        raise ValueError(
            "expected a thrush-ternary-v1 export with HF tensor names (export_ternary.py --names hf), got "
            f"format={export.get('format')!r} names={export.get('names')!r}"
        )
    return tuple(export["quantized_modules"])


def ternarize(model: ParakeetTdt, quantized: tuple[dict, ...]) -> None:
    """Replace the manifest's quantized modules by ternary layers, in place (on the meta device, before loading).

    Every quantized module is a linear: the encoder applies its 1x1 convolutions in the linear layout too. The
    export keeps the attention projections separate, so a block's q/k/v triple becomes the model's one fused
    ``qkv_proj`` — ``RelativeAttention._load_from_state_dict`` concatenates their rows as it does for fp weights.
    """
    from kestrel_kernels.ternary import TernaryLinear

    def layer(entry: dict, out_features: int | None = None) -> TernaryLinear:
        return TernaryLinear(
            entry["in_features"],
            entry["out_features"] if out_features is None else out_features,
            group_size=entry["group_size"],
            bias=entry["has_bias"],
        )

    fused: dict[str, dict[str, dict]] = {}
    for entry in quantized:
        parent, _, attr = entry["name"].rpartition(".")
        if attr in _QKV and hasattr(model.get_submodule(parent), "qkv_proj"):
            fused.setdefault(parent, {})[attr] = entry
        else:
            model.set_submodule(entry["name"], layer(entry))
    for parent, parts in fused.items():
        if set(parts) != set(_QKV):
            raise ValueError(f"{parent}: the export quantizes {sorted(parts)}, not all of q/k/v")
        out_features = sum(parts[attr]["out_features"] for attr in _QKV)
        model.set_submodule(f"{parent}.qkv_proj", layer(parts["q_proj"], out_features))


def load_parakeet_tdt(
    checkpoint: str | Path = MODEL_ID,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
    ternary_mode: str = "auto",
) -> LoadedParakeetTdt:
    """Load the pinned fp checkpoint, or the ternary student when the checkpoint carries a ``ternary.json``.

    The ternary variant loads the packed export with ``strict=True`` and then materializes the weight form
    ``ternary_mode`` selects: ``dense`` dequantizes the codes once (2 bytes/weight), ``gemm8`` keeps them
    packed as GEMM panels and quantizes the activations to int8, and ``auto`` leaves the choice to
    kestrel-kernels. Activations stay in ``dtype`` otherwise; nothing else is quantized at run time.
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
    with torch.device("meta"):
        model = ParakeetTdt(config)
        if manifest.exists():
            ternarize(model, _quantized_modules(manifest))
    state = load_file(str(root / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_nonpersistent_buffers()
    if manifest.exists():
        from kestrel_kernels.ternary import materialize_ternary

        # Between the two casts: the cache dequantizes the packed codes on the target device and keeps its
        # scales in fp32, so it must exist before the module cast reaches the dequantized weights.
        model.to(device=device)
        materialize_ternary(model, ternary_mode, dtype)
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
]
