"""Direct Gemma 4 config and weight loading without Transformers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from kestrel.runtime.bounded_projection import (
    bind_declared_packed_projections,
    declared_packed_projection_source_keys,
)
from kestrel.runtime.generated_decode import materialize_remaining_meta_tensors

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


def _fill_derived_buffer(
    module: torch.nn.Module,
    name: str,
    value: float,
    *,
    device: torch.device,
) -> None:
    buffer = module._buffers[name]
    if buffer.device.type == "meta":
        module._buffers[name] = torch.full(
            tuple(buffer.shape), value, dtype=buffer.dtype, device=device
        )
    else:
        with torch.no_grad():
            buffer.fill_(value)


def _restore_checkpoint_independent_state(
    model: Gemma4InferenceModel,
    config: Gemma4Config,
    *,
    device: torch.device,
) -> None:
    """Restore constants intentionally omitted from Gemma checkpoints."""

    for module in model.modules():
        if (
            hasattr(module, "eps")
            and "weight" in module._non_persistent_buffers_set
            and module._buffers.get("weight") is not None
        ):
            _fill_derived_buffer(module, "weight", 1.0, device=device)

    text = model.model.language_model
    _fill_derived_buffer(
        text.embed_tokens,
        "embed_scale",
        config.text_config.hidden_size**0.5,
        device=device,
    )
    if config.text_config.hidden_size_per_layer_input:
        _fill_derived_buffer(
            text.embed_tokens_per_layer,
            "embed_scale",
            config.text_config.hidden_size_per_layer_input**0.5,
            device=device,
        )
        _fill_derived_buffer(
            text,
            "per_layer_input_scale",
            2.0**-0.5,
            device=device,
        )
    text.rotary_emb = type(text.rotary_emb)(config.text_config, device=device)
    vision = model.model.vision_tower
    vision.encoder.rotary_emb = type(vision.encoder.rotary_emb)(
        config.vision_config.head_dim,
        config.vision_config.rope.theta,
        dimensions=2,
        device=device,
    )


def load_model(
    source: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
    prepare_model: Callable[[torch.nn.Module], None] | None = None,
    finalize_model: Callable[[torch.nn.Module], None] | None = None,
) -> Gemma4InferenceModel:
    snapshot = _snapshot(source, revision=revision)
    config_path = snapshot / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = Gemma4Config.from_dict(json.load(handle))

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        construction_device = torch.device("meta") if prepare_model else device
        with torch.device(construction_device):
            model = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)
    if prepare_model is not None:
        prepare_model(model)
        _restore_checkpoint_independent_state(model, config, device=device)
        materialize_remaining_meta_tensors(model, device=device)

    load_weights(snapshot, model)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    if finalize_model is not None:
        finalize_model(model)
    return model.to(device=device).eval()


__all__ = [
    "load_model",
    "load_weights",
]
