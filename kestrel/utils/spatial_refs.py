"""Spatial reference validation and token helpers."""

import math
from typing import List, Optional, Sequence, Tuple

from kestrel.moondream.runtime import CoordToken, SizeToken, Token


def normalize_spatial_refs(
    spatial_refs: Optional[Sequence[Sequence[float]]],
) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if spatial_refs is None:
        return None
    normalized_refs: List[Tuple[float, ...]] = []
    for idx, ref in enumerate(spatial_refs):
        if len(ref) not in (2, 4):
            raise ValueError(
                f"spatial_refs[{idx}] must contain 2 (point) or 4 (bbox) values"
            )
        converted = [float(value) for value in ref]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"spatial_refs[{idx}] contains non-finite values")
        if not all(0.0 <= value <= 1.0 for value in converted):
            raise ValueError(
                f"spatial_refs[{idx}] values must be normalised to [0, 1]"
            )
        if len(converted) == 4:
            x_min, y_min, x_max, y_max = converted
            if x_min > x_max or y_min > y_max:
                raise ValueError(
                    f"spatial_refs[{idx}] bbox must satisfy x_min<=x_max and y_min<=y_max"
                )
        normalized_refs.append(tuple(converted))
    return tuple(normalized_refs) if normalized_refs else None


def build_spatial_tokens(
    spatial_refs: Optional[Sequence[Sequence[float]]],
) -> List[Token]:
    refs = normalize_spatial_refs(spatial_refs)
    if not refs:
        return []
    tokens: List[Token] = []
    for ref in refs:
        if len(ref) == 2:
            x, y = ref
            tokens.extend([CoordToken(pos=float(x)), CoordToken(pos=float(y))])
        else:
            x_min, y_min, x_max, y_max = ref
            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            tokens.extend(
                [
                    CoordToken(pos=x_c),
                    CoordToken(pos=y_c),
                    SizeToken(width=width, height=height),
                ]
            )
    return tokens
