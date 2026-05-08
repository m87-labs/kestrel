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
from kestrel_kernels.sampling import sample_step_from_logits


def _as_long_bins(raw: Tensor) -> Tensor:
    return raw if raw.dtype == torch.long else raw.to(torch.long)


def compute_spatial_values(
    token_ids: Tensor,
    hidden_last: Tensor,
    requests: List[GenerationRequest],
    spatial_tables: SpatialDecodeTables,
    *,
    temperatures: Tensor | None = None,
    top_ps: Tensor | None = None,
    token_logprobs: Tensor | None = None,
    coord_id: int | None = None,
    size_id: int | None = None,
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
        token_logprobs: Optional per-token logprob buffer to update in-place.
        coord_id: Vocabulary id for coord tokens when token_logprobs is provided.
        size_id: Vocabulary id for size tokens when token_logprobs is provided.
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
    if token_logprobs is not None and (coord_id is None or size_id is None):
        raise AssertionError(
            "coord_id and size_id are required when token_logprobs is provided"
        )

    hidden = hidden_last.unsqueeze(0) if hidden_last.ndim == 1 else hidden_last

    do_sample = not all(req.temperature <= 0.0 for req in requests)
    if do_sample and (temperatures is None or top_ps is None):
        raise RuntimeError("Missing sampling parameters for spatial decode")

    coord_logits, width_logits, height_logits = spatial_decode_logits(
        hidden, spatial_tables
    )

    if not do_sample and token_logprobs is None:
        coord_bins = torch.argmax(coord_logits, dim=-1)
        width_bins = torch.argmax(width_logits, dim=-1)
        height_bins = torch.argmax(height_logits, dim=-1)
        coord_logprobs = size_logprobs = None
    else:
        coord_logprobs = (
            torch.empty((batch,), dtype=torch.float32, device=hidden.device)
            if token_logprobs is not None
            else None
        )
        size_logprobs = (
            torch.empty((batch * 2,), dtype=torch.float32, device=hidden.device)
            if token_logprobs is not None
            else None
        )
        if do_sample:
            assert temperatures is not None and top_ps is not None
        elif temperatures is None or top_ps is None:
            temperatures = torch.zeros(
                (batch,), dtype=torch.float32, device=hidden.device
            )
            top_ps = torch.ones((batch,), dtype=torch.float32, device=hidden.device)

        coord_kwargs = {"generator": rng}
        if coord_logprobs is not None:
            coord_kwargs["logprobs_out"] = coord_logprobs
        coord_bins_raw = sample_step_from_logits(
            coord_logits,
            temperatures,
            top_ps,
            **coord_kwargs,
        )
        coord_bins = _as_long_bins(coord_bins_raw)

        logits_2 = torch.cat((width_logits, height_logits), dim=0)
        size_kwargs = {"generator": rng}
        if size_logprobs is not None:
            size_kwargs["logprobs_out"] = size_logprobs
        bins_2_raw = sample_step_from_logits(
            logits_2,
            temperatures.repeat(2),
            top_ps.repeat(2),
            **size_kwargs,
        )
        bins_2 = _as_long_bins(bins_2_raw)
        width_bins = bins_2[:batch]
        height_bins = bins_2[batch:]

    if token_logprobs is not None:
        assert coord_id is not None and size_id is not None
        assert coord_logprobs is not None and size_logprobs is not None
        row_token_ids = token_ids.view(-1)[:batch]
        coord_mask = row_token_ids == int(coord_id)
        size_mask = row_token_ids == int(size_id)
        token_logprobs[:batch].add_(coord_logprobs * coord_mask)
        width_logprobs = size_logprobs[:batch]
        height_logprobs = size_logprobs[batch:]
        token_logprobs[:batch].add_((width_logprobs + height_logprobs) * size_mask)

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
