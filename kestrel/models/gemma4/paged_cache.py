"""Derive Gemma's logical-to-physical paged-KV layout."""

from __future__ import annotations

from kestrel.kv_cache import PagedKVLayerSpec

from .config import Gemma4TextConfig, attention_kv_heads


def kv_source_layers(config: Gemma4TextConfig) -> tuple[int, ...]:
    """Map every layer to the non-shared layer that owns its K/V storage."""

    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    latest: dict[str, int] = {}
    sources: list[int] = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_idx < first_shared:
            latest[layer_type] = layer_idx
            sources.append(layer_idx)
            continue
        try:
            sources.append(latest[layer_type])
        except KeyError as exc:
            raise ValueError(
                f"shared {layer_type!r} layer {layer_idx} has no K/V producer"
            ) from exc
    return tuple(sources)


def paged_kv_specs(
    config: Gemma4TextConfig,
) -> tuple[PagedKVLayerSpec | None, ...]:
    sources = kv_source_layers(config)
    specs: list[PagedKVLayerSpec | None] = []
    for layer_idx, source_idx in enumerate(sources):
        if source_idx != layer_idx:
            specs.append(None)
            continue
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        specs.append(
            PagedKVLayerSpec(
                n_heads=attention_kv_heads(config, is_sliding=is_sliding),
                head_dim=config.head_dim if is_sliding else config.global_head_dim,
            )
        )
    return tuple(specs)


__all__ = ["kv_source_layers", "paged_kv_specs"]
