"""Spatial token (coord/size) value decoding."""

from typing import List, Optional

import torch
from torch import Tensor

from kestrel.moondream.region import (
    SpatialDecodeTables,
    spatial_bins_to_values,
    spatial_decode_logits,
)
from kestrel.scheduler.types import GenerationRequest
from .sampling import sample_tokens


def compute_spatial_values(
    token_ids: Tensor,
    hidden_last: Tensor,
    requests: List[GenerationRequest],
    spatial_tables: SpatialDecodeTables,
    *,
    temperatures: Tensor | None = None,
    top_ps: Tensor | None = None,
    out_coord: Tensor,
    out_size: Tensor,
    rng: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Decode coord/size token values from hidden states on GPU.

    Args:
        token_ids: Sampled token IDs [batch].
        hidden_last: Last hidden states [batch, hidden_dim].
        requests: Generation requests for sampling parameters.
        spatial_tables: Precomputed spatial decode tables.
        temperatures: Per-request temperatures [batch] (optional).
        top_ps: Per-request top_p values [batch] (optional).
        out_coord: Output buffer for coord values [batch, 1].
        out_size: Output buffer for size values [batch, 2].
        rng: Random generator for sampling (if None, uses greedy decoding).

    Returns:
        Tuple of (coord_values, size_values) sliced to actual batch size.
    """
    if token_ids.ndim != 1:
        token_ids = token_ids.view(-1)
    batch = int(token_ids.shape[0])
    if batch == 0:
        device = hidden_last.device
        coord_decode = torch.empty((0, 1), device=device, dtype=out_coord.dtype)
        size_decode = torch.empty((0, 2), device=device, dtype=out_size.dtype)
        return coord_decode, size_decode

    hidden = hidden_last.unsqueeze(0) if hidden_last.ndim == 1 else hidden_last

    do_sample = not all(req.temperature <= 0.0 for req in requests)
    if do_sample and (temperatures is None or top_ps is None):
        temps_cpu = torch.tensor(
            [req.temperature for req in requests],
            dtype=torch.float32,
        )
        top_ps_cpu = torch.tensor(
            [req.top_p for req in requests],
            dtype=torch.float32,
        )
        temperatures = temps_cpu.to(device=hidden.device)
        top_ps = top_ps_cpu.to(device=hidden.device)

    coord_logits, width_logits, height_logits = spatial_decode_logits(
        hidden, spatial_tables
    )

    if not do_sample:
        coord_bins = torch.argmax(coord_logits, dim=-1)
        width_bins = torch.argmax(width_logits, dim=-1)
        height_bins = torch.argmax(height_logits, dim=-1)
    else:
        if temperatures is None or top_ps is None:  # pragma: no cover - defensive
            raise RuntimeError("Missing sampling parameters for spatial decode")
        coord_bins_raw = sample_tokens(
            coord_logits, temperatures, top_ps, generator=rng
        )
        # Ensure long dtype for indexing
        if coord_bins_raw.dtype == torch.long:
            coord_bins = coord_bins_raw
        else:
            coord_bins = coord_bins_raw.to(torch.long)

        logits_2 = torch.cat((width_logits, height_logits), dim=0)
        bins_2_raw = sample_tokens(
            logits_2,
            temperatures.repeat(2),
            top_ps.repeat(2),
            generator=rng,
        )
        if bins_2_raw.dtype == torch.long:
            bins_2 = bins_2_raw
        else:
            bins_2 = bins_2_raw.to(torch.long)
        width_bins = bins_2[:batch]
        height_bins = bins_2[batch:]

    coord_out = out_coord[:batch]
    size_out = out_size[:batch]
    spatial_bins_to_values(
        coord_bins,
        width_bins,
        height_bins,
        spatial_tables,
        out_coord=coord_out,
        out_size=size_out,
    )
    return coord_out, size_out
