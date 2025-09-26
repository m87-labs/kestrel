"""Sampling utilities for batched decode without host synchronization."""

from __future__ import annotations

import torch
from torch import Tensor


_EPS = 1e-6


def sample_tokens(
    logits: Tensor,
    temperatures: Tensor,
    top_ps: Tensor,
) -> Tensor:
    """Return sampled token ids for a batch of logits.

    Parameters
    ----------
    logits:
        Tensor of shape ``(batch, vocab)`` located on the same device as the model.
    temperatures, top_ps:
        Per-request sampling parameters. ``temperatures`` are clamped to be non-negative
        and ``top_ps`` are clipped into ``(0, 1]``.

    Returns
    -------
    Tensor
        Token ids with shape ``(batch,)`` on the same device as ``logits``.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, vocab); received shape {logits.shape}")

    device = logits.device
    dtype = logits.dtype
    batch = logits.shape[0]

    if temperatures.shape[0] != batch or top_ps.shape[0] != batch:
        raise ValueError("Sampling parameters must match logits batch dimension")

    temps = torch.clamp(temperatures, min=0.0)
    top_ps_tensor = torch.clamp(top_ps, min=_EPS, max=1.0)

    result = torch.empty(batch, device=device, dtype=torch.long)

    greedy_mask = temps <= _EPS
    if greedy_mask.any():
        greedy_logits = logits[greedy_mask]
        if greedy_logits.numel():
            result[greedy_mask] = torch.argmax(greedy_logits, dim=-1)

    sample_mask = ~greedy_mask
    if sample_mask.any():
        idx = sample_mask.nonzero(as_tuple=False).squeeze(-1)
        selected_logits = logits.index_select(0, idx)
        temp = temps.index_select(0, idx).unsqueeze(-1)

        scaled = selected_logits / torch.clamp(temp, min=_EPS)
        probs = torch.softmax(scaled.to(dtype=torch.float32), dim=-1)

        # Guard against NaN/Inf from softmax in low-precision regimes.
        invalid = torch.isnan(probs).any(dim=-1) | torch.isinf(probs).any(dim=-1)
        if invalid.any():
            fallback_idx = idx[invalid]
            if fallback_idx.numel():
                result[fallback_idx] = torch.argmax(logits.index_select(0, fallback_idx), dim=-1)
            valid_idx = idx[~invalid]
            probs = probs[~invalid]
            idx = valid_idx
        top_ps = top_ps_tensor.index_select(0, idx)
        if idx.numel():
            sampled = _sample_with_top_p(probs, top_ps)
            result[idx] = sampled

    return result


def _sample_with_top_p(probs: Tensor, top_ps: Tensor) -> Tensor:
    """Apply top-p filtering and sample tokens for each row."""

    if probs.ndim != 2:
        raise ValueError("probs must be 2D")

    device = probs.device
    batch = probs.shape[0]
    tokens = torch.empty(batch, device=device, dtype=torch.long)

    full_mask = top_ps >= (1.0 - _EPS)
    if full_mask.any():
        full_probs = probs[full_mask]
        if full_probs.numel():
            draws = torch.multinomial(full_probs, num_samples=1)
            tokens[full_mask] = draws.squeeze(-1)

    filtered_mask = ~full_mask
    if filtered_mask.any():
        idx = filtered_mask.nonzero(as_tuple=False).squeeze(-1)
        sub_probs = probs.index_select(0, idx)
        top_p = top_ps.index_select(0, idx).unsqueeze(-1)

        sorted_probs, sorted_idx = torch.sort(sub_probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        filtered = torch.where(cutoff, torch.zeros_like(sorted_probs), sorted_probs)

        mass = filtered.sum(dim=-1, keepdim=True)
        zero_mass = mass.squeeze(-1) <= _EPS
        if zero_mass.any():
            fallback = sorted_idx[zero_mass, 0]
            tokens[idx[zero_mass]] = fallback

        if (~zero_mass).any():
            keep_idx = idx[~zero_mass]
            filtered = filtered[~zero_mass] / mass[~zero_mass]
            draws = torch.multinomial(filtered, num_samples=1)
            picked = sorted_idx[~zero_mass].gather(1, draws).squeeze(-1)
            tokens[keep_idx] = picked

    return tokens


__all__ = ["sample_tokens"]
