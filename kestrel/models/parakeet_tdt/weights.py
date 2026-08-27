"""Direct loading for the pinned Parakeet TDT checkpoint."""

from __future__ import annotations

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
) -> LoadedParakeetTdt:
    root = resolve_checkpoint(
        checkpoint,
        revision=revision,
        filenames=_FILES,
        local_files_only=local_files_only,
    )
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


__all__ = ["LoadedParakeetTdt", "MODEL_ID", "REVISION", "load_parakeet_tdt"]
