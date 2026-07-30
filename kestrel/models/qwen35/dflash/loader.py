"""Load a published z-lab DFlash draft checkpoint into ``DFlashDraftModel``.

The draft module's parameter names mirror the checkpoint keys exactly, so loading
is a direct ``load_state_dict`` — no key remapping, no fused-projection assembly,
and (unlike the target loader) **no RMSNorm offset folding** (these weights are
standard).
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .model import DFlashConfig, DFlashDraftModel


def load_dflash_drafter(
    repo_id: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DFlashDraftModel, DFlashConfig]:
    """Download and load a DFlash drafter checkpoint (single-file safetensors)."""
    config = DFlashConfig.from_dict(
        json.load(open(hf_hub_download(repo_id, "config.json")))
    )
    model = DFlashDraftModel(config).to(dtype)
    state = load_file(hf_hub_download(repo_id, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"DFlash checkpoint mismatch for {repo_id}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval(), config
