"""Generic preparation passes for inference compilation."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def materialize_dynamic_batch_domain(
    compiled: Callable[..., Any],
    *,
    max_batch_size: int,
    inputs_for_batch: Callable[[int], tuple[Any, ...]],
    synchronize: Callable[[], None] | None = None,
) -> None:
    """Materialize every admitted batch before accepting serving work.

    ``torch.compile(dynamic=True)`` still specializes dimensions whose first
    observed value is zero or one. A symbolic graph can also retain
    shape-specific first-execution work below the graph/compiler-cache layer.
    The admitted batch domain is finite and small, so exercise it completely
    before public work is accepted instead of relying on compiler internals to
    make intermediate extents serving-ready. Keep capacity as the first
    positive extent so it remains the representative symbolic compile input.
    """

    max_batch_size = int(max_batch_size)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    batch_sizes = (
        (1,)
        if max_batch_size == 1
        else (1, max_batch_size, *range(2, max_batch_size))
    )
    with torch.inference_mode():
        for batch_size in batch_sizes:
            output = compiled(*inputs_for_batch(batch_size))
            del output
    if synchronize is not None:
        synchronize()


def immutable_scalar_key(value: torch.Tensor) -> tuple[object, ...]:
    host_bytes = bytes(
        value.detach()
        .cpu()
        .contiguous()
        .reshape(1)
        .view(torch.uint8)
        .tolist()
    )
    return (value.dtype, value.device.type, value.device.index, value.layout, host_bytes)


def canonicalize_immutable_scalar_buffers(
    module: nn.Module,
) -> None:
    """Alias bit-identical immutable scalar buffers before inference compilation.

    Sharing the Tensor object lets graph compilers represent repeated scalar
    constants once. The caller owns the immutability assertion: none of the
    candidate scalar buffers may be mutated after this pass.
    """

    training_modules = [
        name or "<root>" for name, child in module.named_modules() if child.training
    ]
    if training_modules:
        raise ValueError(
            "immutable buffer canonicalization requires eval mode; training modules: "
            + ", ".join(training_modules[:4])
        )

    canonical: dict[tuple[object, ...], torch.Tensor] = {}
    for child in module.modules():
        for name, value in child._buffers.items():
            if (
                value is None
                or value.ndim != 0
                or value.layout is not torch.strided
                or value.is_quantized
            ):
                continue
            if value.requires_grad:
                raise ValueError(
                    f"immutable scalar buffer {name!r} must not require gradients"
                )
            if value.device.type == "meta":
                raise ValueError(
                    f"immutable scalar buffer {name!r} must be materialized"
                )
            key = immutable_scalar_key(value)
            existing = canonical.setdefault(key, value)
            if existing is not value:
                child._buffers[name] = existing
