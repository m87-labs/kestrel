"""Token materialization helpers."""

from torch import Tensor

from kestrel.moondream.runtime import (
    TextToken,
    CoordToken,
    SizeToken,
    Token,
)
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
