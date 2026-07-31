"""Shared physical row operations for batched prefill."""

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn


def gather_padded_last_rows(
    hidden_states: torch.Tensor,
    lengths: Sequence[int],
) -> torch.Tensor:
    """Gather the final active row from each tail-padded batch segment."""

    if hidden_states.ndim != 3:
        raise ValueError(
            "padded hidden states must have shape [batch, rows, channels], "
            f"got {tuple(hidden_states.shape)}"
        )
    values = tuple(lengths)
    batch, rows, _ = hidden_states.shape
    if len(values) != batch:
        raise ValueError(
            f"padded row lengths must match batch {batch}, got {len(values)}"
        )
    if any(
        isinstance(length, bool) or not isinstance(length, Integral)
        for length in values
    ):
        raise TypeError("padded row lengths must be integers")
    values = tuple(int(length) for length in values)
    if any(length <= 0 or length > rows for length in values):
        raise ValueError(
            f"padded row lengths must lie in [1, {rows}], got {values}"
        )
    row_indices = torch.tensor(
        [length - 1 for length in values],
        dtype=torch.long,
        device=hidden_states.device,
    )
    batch_indices = torch.arange(
        batch,
        dtype=torch.long,
        device=hidden_states.device,
    )
    return hidden_states[batch_indices, row_indices]


def project_padded_last_rows(
    hidden_states: torch.Tensor,
    lengths: Sequence[int],
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a padded batch boundary and project all rows in one contraction."""

    rows = gather_padded_last_rows(hidden_states, lengths)
    return rows, projection(rows)


def project_packed_last_rows(
    hidden_states: torch.Tensor,
    last_token_offsets: torch.Tensor,
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed sequence boundaries and project them together."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(
            "packed hidden states must have shape [1, tokens, channels]"
        )
    if (
        last_token_offsets.ndim != 1
        or last_token_offsets.dtype != torch.long
    ):
        raise ValueError("packed last-token offsets must be a long vector")
    rows = hidden_states[0].index_select(0, last_token_offsets)
    return rows, projection(rows)
