"""Smoke-test the Moondream DFlash drafter checkpoint.

This intentionally avoids constructing the full target transformer. The drafter
forward only needs raw target text tensors and spatial decode tables.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from kestrel.models.moondream.config import RegionConfig
from kestrel.models.moondream.dflash import (
    DFlashBatch,
    inspect_dflash_checkpoint,
    MoondreamDFlashDrafter,
)
from kestrel.models.moondream.region import (
    build_region_module,
    build_spatial_decode_tables,
)


class FakeTargetText(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.wte = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype, device=device)
        )
        self.post_ln = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, dtype=dtype, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16
    metadata = inspect_dflash_checkpoint(args.checkpoint)
    cfg = metadata.config

    target_text = FakeTargetText(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.target_hidden_size,
        dtype=dtype,
        device=device,
    ).eval()
    region = build_region_module(RegionConfig(), dtype=dtype, device=device).eval()
    spatial_tables = build_spatial_decode_tables(region)
    drafter = MoondreamDFlashDrafter.from_checkpoint(
        args.checkpoint,
        target_text=target_text,
        spatial_tables=spatial_tables,
        device=device,
        dtype=dtype,
    )

    batch_size = int(args.batch_size)
    target_hidden = torch.randn(
        (
            batch_size,
            cfg.max_context_tokens,
            cfg.target_layer_count,
            cfg.target_hidden_size,
        ),
        device=device,
        dtype=dtype,
    )
    batch = DFlashBatch(
        current_token_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
        target_hidden_states=target_hidden,
        target_hidden_mask=torch.ones(
            batch_size, cfg.max_context_tokens, device=device, dtype=torch.bool
        ),
    )
    out = drafter(batch)

    expected = (batch_size, cfg.target_width)
    assert out.token_logits.shape == (
        *expected,
        int(metadata.draft_vocab_size),
    ), out.token_logits.shape
    assert out.coord_logits.shape == (*expected, cfg.spatial_bin_count)
    assert out.size_width_logits.shape == (*expected, cfg.spatial_bin_count)
    assert out.size_height_logits.shape == (*expected, cfg.spatial_bin_count)
    assert out.target_hidden.shape == (*expected, cfg.target_hidden_size)
    if cfg.confidence_head:
        assert out.confidence_logits is not None
        assert out.confidence_logits.shape == expected
    if cfg.refine_steps > 1:
        assert out.first_pass_token_logits is not None
        assert out.first_pass_token_logits.shape == out.token_logits.shape

    print(
        "ok",
        {
            "step": metadata.step,
            "target_width": cfg.target_width,
            "draft_vocab_size": metadata.draft_vocab_size,
            "token_logits": tuple(out.token_logits.shape),
            "confidence_logits": (
                None
                if out.confidence_logits is None
                else tuple(out.confidence_logits.shape)
            ),
            "dtype": str(out.token_logits.dtype),
        },
    )


if __name__ == "__main__":
    main()
