"""Paged KV helpers for Gemma 4."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from kestrel.kv_cache import KVMemoryPool, PageTable, PagedKVCache

from .gemma4_compat import SimpleDynamicCache
from .gemma4_config import attention_kv_heads


def _first_shared_layer(config: Any) -> int:
    return int(config.num_hidden_layers) - int(getattr(config, "num_kv_shared_layers", 0) or 0)


def _stores_shared_kv(config: Any, layer_idx: int) -> bool:
    first_shared = _first_shared_layer(config)
    layer_types = list(config.layer_types or [])
    layer_type = layer_types[layer_idx]
    prev_layers = layer_types[:first_shared]
    return (
        layer_type in prev_layers
        and layer_idx < first_shared
        and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
    )


def _paged_layer_config(config: Any, layer_idx: int) -> tuple[int, int] | None:
    if layer_idx >= _first_shared_layer(config):
        return None
    layer_type = config.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        return attention_kv_heads(config, is_sliding=True), int(config.head_dim)
    if layer_type == "full_attention":
        return (
            attention_kv_heads(config, is_sliding=False),
            int(config.global_head_dim),
        )
    raise ValueError(f"unsupported Gemma attention layer type {layer_type!r}")


class Gemma4PagedHybridCache(SimpleDynamicCache):
    """Gemma cache with paged KV for every non-shared KV producer."""

    @staticmethod
    def build_shared_paged_layers(
        *,
        config: Any,
        page_table: PageTable,
        pool: KVMemoryPool,
        dtype: torch.dtype,
    ) -> list[PagedKVCache | None]:
        layers: list[PagedKVCache | None] = []
        for layer_idx in range(int(config.num_hidden_layers)):
            layer_cfg = _paged_layer_config(config, layer_idx)
            if layer_cfg is None:
                layers.append(None)
                continue
            heads, head_dim = layer_cfg
            layers.append(
                PagedKVCache(
                    page_table,
                    n_heads=heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    pool=pool,
                )
            )
        return layers

    def __init__(
        self,
        *,
        config: Any,
        page_table: PageTable,
        pool: KVMemoryPool,
        dtype: torch.dtype,
        shared_paged_layers: Sequence[PagedKVCache | None] | None = None,
    ) -> None:
        super().__init__()
        layer_count = int(config.num_hidden_layers)
        if shared_paged_layers is not None and len(shared_paged_layers) != layer_count:
            raise ValueError("shared_paged_layers length must match Gemma layer count")

        self.layers: list[PagedKVCache | None] = []
        for layer_idx in range(layer_count):
            layer_cfg = _paged_layer_config(config, layer_idx)
            if layer_cfg is None:
                self.layers.append(None)
                continue
            if shared_paged_layers is not None:
                layer = shared_paged_layers[layer_idx]
                if not isinstance(layer, PagedKVCache):
                    raise ValueError(
                        "shared_paged_layers must contain PagedKVCache for "
                        f"Gemma paged layer {layer_idx}"
                    )
                self.layers.append(layer)
                continue
            heads, head_dim = layer_cfg
            self.layers.append(
                PagedKVCache(
                    page_table,
                    n_heads=heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    pool=pool,
                )
            )
        self.seq_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.get_paged_layer(layer_idx) is not None:
            raise RuntimeError(
                "Gemma paged KV layers update PagedKVCache directly with BSHD tensors"
            )
        return super().update(key_states, value_states, layer_idx)

    def get_paged_layer(self, layer_idx: int) -> PagedKVCache | None:
        if not (0 <= layer_idx < len(self.layers)):
            return None
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.seq_length) if any(self.layers) else super().get_seq_length(layer_idx)

    def uses_paged_kv(self) -> bool:
        return any(layer is not None for layer in self.layers)

    def advance_to(self, seq_length: int) -> None:
        self.seq_length = max(self.seq_length, int(seq_length))


__all__ = ["Gemma4PagedHybridCache"]
