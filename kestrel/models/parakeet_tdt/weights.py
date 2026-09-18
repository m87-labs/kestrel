"""Direct loading for the pinned Parakeet TDT checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from kestrel.models.asr.checkpoint import resolve_checkpoint

from .config import ParakeetTdtConfig
from .model import ParakeetTdt
from .tokenizer import ParakeetTokenizer


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
_FILES = ("config.json", "tokenizer.json", "model.safetensors")
# The ternary (2-bit) student distilled from MODEL_ID (thrush), published at TERNARY_MODEL_ID. A checkpoint holds the packed
# ``model.safetensors`` + ``ternary.json`` (thrush ``scripts/export_ternary.py --names hf``), plus config.json and
# tokenizer.json; ``load_parakeet_tdt`` recognises the manifest and builds the ternary variant.
TERNARY_MODEL_ID = "moondream/parakeet-redux"
TERNARY_REVISION = "70828b0628e071b8f58c36911f418d6c8096bc6f"  # rl6 export, private, for runtime testing
TERNARY_MANIFEST = "ternary.json"
_TERNARY_FILES = _FILES + (TERNARY_MANIFEST,)
# The ternary student ships for CPU and Apple silicon only for now: a CUDA request runs it on MPS when available,
# else on the CPU (its GEMMs are dense bf16/fp16/fp32 through torch there; the 2-bit kernels are CPU/ARM code).
TERNARY_DEVICE_TYPES = frozenset({"cpu", "mps"})


@dataclass(frozen=True, slots=True)
class LoadedParakeetTdt:
    model: ParakeetTdt
    tokenizer: ParakeetTokenizer


def load_parakeet_tdt(
    checkpoint: str | Path = MODEL_ID,
    *,
    revision: str = REVISION,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
    ternary_mode: str = "auto",
) -> LoadedParakeetTdt:
    ternary_repo = str(checkpoint) == TERNARY_MODEL_ID
    root = resolve_checkpoint(
        checkpoint,
        revision=TERNARY_REVISION if ternary_repo and revision == REVISION else revision,
        filenames=_TERNARY_FILES if ternary_repo else _FILES,
        local_files_only=local_files_only,
    )
    manifest_path = root / TERNARY_MANIFEST
    if manifest_path.exists():
        return load_parakeet_tdt_ternary(root, device=device, dtype=dtype, mode=ternary_mode)
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    tokenizer = ParakeetTokenizer(root / "tokenizer.json")
    if (
        tokenizer.blank_token_id != config.blank_token_id
        or tokenizer.pad_token_id != config.pad_token_id
    ):
        raise ValueError("Parakeet tokenizer and model special tokens disagree")
    with torch.device("meta"):
        model = ParakeetTdt(config)
    from safetensors.torch import load_file

    state = load_file(str(root / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_nonpersistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return LoadedParakeetTdt(model, tokenizer)


# ---- ternary (2-bit) variant ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TernaryManifest:
    quantized_modules: tuple[dict, ...]
    group_size: int
    size_mb: float
    source: dict

    @classmethod
    def load(cls, path: Path) -> "TernaryManifest":
        m = json.loads(path.read_text())
        if m.get("format") != "thrush-ternary-v1":
            raise ValueError(f"unknown ternary export format {m.get('format')!r}")
        if m.get("names") != "hf":
            raise ValueError("the ternary export must use HF tensor names (export_ternary.py --names hf)")
        return cls(
            tuple(m["quantized_modules"]),
            int(m["quant"]["group_size"]),
            float(m["size_mb"]),
            m["source"],
        )


def is_ternary_checkpoint(checkpoint: str | Path | None, model_name: str | None = None) -> bool:
    """Whether ``checkpoint`` (a local directory) or ``model_name`` designates the ternary student."""
    if model_name == TERNARY_MODEL_ID:
        return True
    if checkpoint is None:
        return False
    path = Path(checkpoint)
    return path.is_dir() and (path / TERNARY_MANIFEST).exists()


def ternary_runtime_device(device: torch.device) -> torch.device:
    """The device the ternary student actually runs on: ``device`` when it is CPU or MPS, otherwise MPS if this
    torch has it, else CPU (see ``TERNARY_DEVICE_TYPES``)."""
    if device.type in TERNARY_DEVICE_TYPES:
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, module)


_QKV = ("q_proj", "k_proj", "v_proj")


def _fused_qkv_groups(model: nn.Module, manifest: TernaryManifest) -> dict[str, list[dict]]:
    """``{parent: [q, k, v manifest entries]}`` for attention blocks whose module fuses q/k/v into ``qkv_proj``
    (kestrel's ``ParakeetAttention``); the export keeps the three projections separate."""
    by_parent: dict[str, dict[str, dict]] = {}
    for q in manifest.quantized_modules:
        parent, _, attr = q["name"].rpartition(".")
        if attr in _QKV:
            by_parent.setdefault(parent, {})[attr] = q
    groups = {}
    for parent, parts in by_parent.items():
        if len(parts) == 3 and hasattr(model.get_submodule(parent), "qkv_proj"):
            groups[parent] = [parts[a] for a in _QKV]
    return groups


def ternarize(model: ParakeetTdt, manifest: TernaryManifest) -> ParakeetTdt:
    """Replace the manifest's linears / 1x1 convs by ternary layers (on the meta device, before loading). The
    export's separate q/k/v projections become one ``qkv_proj`` ternary layer where the model fuses them."""
    from kestrel_kernels.ternary import TernaryLinear

    fused = _fused_qkv_groups(model, manifest)
    fused_names = {q["name"] for parts in fused.values() for q in parts}
    for q in manifest.quantized_modules:
        if q["name"] in fused_names:
            continue
        # the encoder applies its 1x1 convs in the linear layout (see ``Convolution.forward``), so every quantized
        # module is a TernaryLinear here; TernaryConv1x1 is for models that call them on [B, C, T]
        mod = TernaryLinear(q["in_features"], q["out_features"], group_size=q["group_size"], bias=q["has_bias"])
        _set_submodule(model, q["name"], mod)
    for parent, parts in fused.items():
        q = parts[0]
        mod = TernaryLinear(
            q["in_features"], sum(p["out_features"] for p in parts), group_size=q["group_size"], bias=q["has_bias"]
        )
        _set_submodule(model, f"{parent}.qkv_proj", mod)
    return model


def _fuse_qkv_state(state: dict[str, torch.Tensor], fused: dict[str, list[dict]]) -> None:
    """Row-concatenate the export's q/k/v ``qweight`` / ``scales`` (/ ``bias``) into the fused layer's tensors;
    ternary rows are independent, so packing and scales concatenate directly."""
    for parent, parts in fused.items():
        for tensor in ("qweight", "scales", "bias"):
            keys = [f"{p['name']}.{tensor}" for p in parts]
            if all(k in state for k in keys):
                state[f"{parent}.qkv_proj.{tensor}"] = torch.cat([state.pop(k) for k in keys], dim=0).contiguous()


def load_parakeet_tdt_ternary(
    root: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    mode: str = "auto",
) -> LoadedParakeetTdt:
    """Build the fp architecture, swap in the ternary layers, load the packed export with ``strict=True``, then
    materialize the weight cache: ``dense`` (dequantized once, 2 bytes/weight in bf16/fp16; the default until the
    packed-code GEMM kernels ship), ``jit`` (2-bit weights resident, each layer expanded into a shared scratch
    buffer by the native op right before its GEMM; opt-in, 0.6-0.8x of dense), ``int8`` (1 byte/weight, one scaling
    pass per call) or ``packed`` (0.25 byte/weight, bit-unpack in torch per call). ``auto`` = the runtime's default
    (dense). Activations
    stay in ``dtype``; nothing is quantized at run time. ``root`` needs config.json and tokenizer.json next to the
    export (the pinned fp checkpoint's files are used when they are absent)."""
    from kestrel_kernels.ternary import materialize_ternary
    from safetensors.torch import load_file

    root = Path(root)
    manifest = TernaryManifest.load(root / TERNARY_MANIFEST)
    if (root / "config.json").exists() and (root / "tokenizer.json").exists():
        config_dir = root
    else:
        config_dir = resolve_checkpoint(MODEL_ID, revision=REVISION, filenames=("config.json", "tokenizer.json"))
    config = ParakeetTdtConfig.from_json_file(config_dir / "config.json")
    tokenizer = ParakeetTokenizer(config_dir / "tokenizer.json")
    with torch.device("meta"):
        model = ternarize(ParakeetTdt(config), manifest)
    state = load_file(str(root / "model.safetensors"), device="cpu")
    _fuse_qkv_state(state, _fused_qkv_groups(model, manifest))
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_nonpersistent_buffers()
    model.to(device=device)
    materialize_ternary(model, mode, dtype)  # before the dtype cast: the cache keeps fp32 scales / dtype weights
    model.to(dtype=dtype).eval()
    return LoadedParakeetTdt(model, tokenizer)


__all__ = [
    "LoadedParakeetTdt",
    "MODEL_ID",
    "REVISION",
    "TERNARY_DEVICE_TYPES",
    "TERNARY_MODEL_ID",
    "TERNARY_REVISION",
    "TernaryManifest",
    "is_ternary_checkpoint",
    "load_parakeet_tdt",
    "load_parakeet_tdt_ternary",
    "ternarize",
    "ternary_runtime_device",
]
