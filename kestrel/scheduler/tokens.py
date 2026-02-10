"""Token materialization and prompt processing helpers."""

from __future__ import annotations

from torch import Tensor

from kestrel.moondream.runtime import (
    CoordToken,
    SizeToken,
    TextToken,
    Token,
)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def prompt_with_spatial_tokens(
    prompt: Tensor,
    coord_id: int,
    size_id: int,
    spatial_refs: list[list[float]],
) -> list[Token]:
    """Render a prompt id tensor into typed tokens with spatial placeholders filled.

    `spatial_refs` entries are either:
      - point: [x, y]
      - bbox: [x_min, y_min, x_max, y_max] (converted to center + size)
    """
    ids = prompt.view(-1).tolist()

    expected_tokens: list[Token] = []
    for ref in spatial_refs:
        if len(ref) == 2:
            x, y = ref
            expected_tokens.append(CoordToken(pos=_clamp01(float(x))))
            expected_tokens.append(CoordToken(pos=_clamp01(float(y))))
            continue
        if len(ref) == 4:
            x_min, y_min, x_max, y_max = [float(v) for v in ref]
            if not (
                0.0 <= x_min <= x_max <= 1.0 and 0.0 <= y_min <= y_max <= 1.0
            ):
                raise ValueError("bbox must satisfy 0<=x_min<=x_max<=1 and 0<=y_min<=y_max<=1")
            expected_tokens.append(CoordToken(pos=(x_min + x_max) * 0.5))
            expected_tokens.append(CoordToken(pos=(y_min + y_max) * 0.5))
            expected_tokens.append(SizeToken(width=x_max - x_min, height=y_max - y_min))
            continue
        raise ValueError("Each spatial ref must have 2 (point) or 4 (bbox) values")

    out: list[Token] = []
    expected_idx = 0
    for token_id in ids:
        if token_id == coord_id:
            if expected_idx >= len(expected_tokens):
                raise ValueError("Mismatch between spatial placeholders and spatial refs")
            token = expected_tokens[expected_idx]
            if not isinstance(token, CoordToken):
                raise ValueError("Mismatch between spatial placeholders and spatial refs")
            out.append(token)
            expected_idx += 1
        elif token_id == size_id:
            if expected_idx >= len(expected_tokens):
                raise ValueError("Mismatch between spatial placeholders and spatial refs")
            token = expected_tokens[expected_idx]
            if not isinstance(token, SizeToken):
                raise ValueError("Mismatch between spatial placeholders and spatial refs")
            out.append(token)
            expected_idx += 1
        else:
            out.append(TextToken(token_id=token_id))

    if expected_idx != len(expected_tokens):
        raise ValueError("Mismatch between spatial placeholders and spatial refs")

    return out


def render_tokens_from_packed(
    token_ids: Tensor,
    coord_values: Tensor,
    size_values: Tensor,
    *,
    coord_id: int,
    size_id: int,
) -> list[Token]:
    """Materialize sampled ids + value tensors into typed tokens on host."""

    ids = token_ids.view(-1).tolist()
    batch = len(ids)
    if batch == 0:
        return []

    out: list[Token] = []
    for i, token_id in enumerate(ids):
        if token_id == coord_id:
            out.append(CoordToken(pos=float(coord_values[i, 0].item())))
        elif token_id == size_id:
            out.append(
                SizeToken(
                    width=float(size_values[i, 0].item()),
                    height=float(size_values[i, 1].item()),
                )
            )
        else:
            out.append(TextToken(token_id=token_id))
    return out


__all__ = ["prompt_with_spatial_tokens", "render_tokens_from_packed"]
