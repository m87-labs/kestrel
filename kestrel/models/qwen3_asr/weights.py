"""Direct loading for the two pinned Qwen3-ASR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from kestrel.models.asr.checkpoint import resolve_checkpoint

from .config import Qwen3AsrConfig
from .model import Qwen3AsrForConditionalGeneration
from .tokenizer import Qwen3AsrTokenizer


QWEN3_ASR_MODELS = {
    "Qwen/Qwen3-ASR-0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "Qwen/Qwen3-ASR-1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
}
_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model*.safetensors",
)


@dataclass(frozen=True, slots=True)
class LoadedQwen3Asr:
    model: Qwen3AsrForConditionalGeneration
    tokenizer: Qwen3AsrTokenizer


def load_qwen3_asr(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> LoadedQwen3Asr:
    name = str(checkpoint)
    resolved_revision = revision or QWEN3_ASR_MODELS.get(name)
    root = resolve_checkpoint(
        checkpoint,
        revision=resolved_revision,
        filenames=_FILES,
        local_files_only=local_files_only,
    )
    config = Qwen3AsrConfig.from_checkpoint(root)
    tokenizer = Qwen3AsrTokenizer(root)
    if tokenizer.audio_token_id != config.audio_token_id:
        raise ValueError("Qwen3-ASR tokenizer and model audio tokens disagree")
    with torch.device("meta"):
        model = Qwen3AsrForConditionalGeneration(config)
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    for shard_path in sorted(root.glob("model*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for source_name in shard.keys():
                target_name = _weight_name(source_name)
                if target_name is not None:
                    state[target_name] = shard.get_tensor(source_name)
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.reset_nonpersistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return LoadedQwen3Asr(model, tokenizer)


def _weight_name(name: str) -> str | None:
    if name == "thinker.lm_head.weight":
        return None
    prefixes = (
        ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
        ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
        ("thinker.audio_tower.", "model.audio_tower."),
        ("thinker.model.", "model.language_model."),
    )
    for source, target in prefixes:
        if name.startswith(source):
            return target + name.removeprefix(source)
    raise KeyError(f"unsupported Qwen3-ASR checkpoint tensor {name!r}")


__all__ = ["LoadedQwen3Asr", "QWEN3_ASR_MODELS", "load_qwen3_asr"]
