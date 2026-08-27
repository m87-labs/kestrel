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
    "Qwen/Qwen3-ASR-0.6B-hf": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
    "Qwen/Qwen3-ASR-1.7B-hf": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
}
_FILES = ("config.json", "tokenizer.json", "model.safetensors")


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
    config = Qwen3AsrConfig.from_json_file(root / "config.json")
    tokenizer = Qwen3AsrTokenizer(root / "tokenizer.json")
    if tokenizer.audio_token_id != config.audio_token_id:
        raise ValueError("Qwen3-ASR tokenizer and model audio tokens disagree")
    with torch.device("meta"):
        model = Qwen3AsrForConditionalGeneration(config)
    from safetensors.torch import load_file

    state = load_file(str(root / "model.safetensors"), device="cpu")
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.reset_nonpersistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return LoadedQwen3Asr(model, tokenizer)


__all__ = ["LoadedQwen3Asr", "QWEN3_ASR_MODELS", "load_qwen3_asr"]
