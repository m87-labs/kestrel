"""Token materialization and prompt processing."""

from typing import Sequence

from torch import Tensor

from kestrel.moondream.runtime import (
    TextToken,
    CoordToken,
    SizeToken,
    Token,
)


def prompt_with_spatial_tokens(
    prompt_tokens: Tensor,
    coord_id: int,
    size_id: int,
    spatial_refs: Sequence[Sequence[float]],
) -> list[Token]:
    """Replace coord/size placeholder ids in ``prompt_tokens`` with typed tokens.

    - 2-value refs are treated as points: ``[x, y]``.
    - 4-value refs are treated strictly as bounding boxes in
      ``[x_min, y_min, x_max, y_max]`` format.
    """
    if prompt_tokens.ndim != 1:
        tokens_1d = prompt_tokens.view(-1)
    else:
        tokens_1d = prompt_tokens
    ids = tokens_1d.cpu().tolist()

    # Precompute expected placeholder counts
    coord_placeholders = sum(1 for t in ids if t == coord_id)
    size_placeholders = sum(1 for t in ids if t == size_id)

    # Build coord and size lists from spatial refs
    coord_vals: list[float] = []
    size_vals: list[tuple[float, float]] = []
    for ref in spatial_refs:
        n = len(ref)
        if n == 2:
            x, y = float(ref[0]), float(ref[1])
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            coord_vals.extend([x, y])
        elif n == 4:
            x_min, y_min, x_max, y_max = map(float, ref)
            if not (0.0 <= x_min <= x_max <= 1.0 and 0.0 <= y_min <= y_max <= 1.0):
                raise ValueError(
                    "bbox spatial_ref must satisfy 0<=x_min<=x_max<=1 and 0<=y_min<=y_max<=1"
                )
            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            coord_vals.extend([x_c, y_c])
            size_vals.append((width, height))
        else:
            raise ValueError(
                "Each spatial_ref must contain 2 (point) or 4 (bbox) values"
            )

    expected_coords = 2 * len(spatial_refs)
    expected_sizes = sum(1 for r in spatial_refs if len(r) == 4)
    if coord_placeholders != expected_coords or size_placeholders != expected_sizes:
        raise ValueError(
            "Mismatch between spatial_refs and placeholder tokens: "
            f"prompt has {coord_placeholders} coord and {size_placeholders} size placeholders, "
            f"but refs require {expected_coords} coord and {expected_sizes} size placeholders."
        )

    # Replace placeholders in order of appearance
    coord_iter = iter(coord_vals)
    size_iter = iter(size_vals)
    out: list[Token] = []
    for tid in ids:
        if tid == coord_id:
            try:
                pos = next(coord_iter)
            except StopIteration as exc:
                raise ValueError("Insufficient coord placeholders for spatial_refs") from exc
            out.append(CoordToken(pos=float(pos)))
        elif tid == size_id:
            try:
                w, h = next(size_iter)
            except StopIteration as exc:
                raise ValueError("Insufficient size placeholders for bbox spatial_refs") from exc
            # Clamp sizes to [0, 1]
            w = min(max(float(w), 0.0), 1.0)
            h = min(max(float(h), 0.0), 1.0)
            out.append(SizeToken(width=w, height=h))
        else:
            out.append(TextToken(token_id=int(tid)))

    # Ensure all refs were consumed
    try:
        next(coord_iter)
        raise ValueError("Unconsumed coord values after placeholder replacement")
    except StopIteration:
        pass
    try:
        next(size_iter)
        raise ValueError("Unconsumed size values after placeholder replacement")
    except StopIteration:
        pass

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
