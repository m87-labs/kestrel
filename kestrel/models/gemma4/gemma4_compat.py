"""Small local compatibility layer for Gemma 4."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


ACT2FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
    "silu": F.silu,
    "relu": F.relu,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in ACT2FN:
        raise ValueError(f"Unsupported activation {name!r}; known: {sorted(ACT2FN)}")
    return ACT2FN[name]


class Cache:
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_seq_length(self, layer_idx: int = 0) -> int:
        raise NotImplementedError


class SimpleDynamicCache(Cache):
    def __init__(self) -> None:
        self._k: list[Optional[torch.Tensor]] = []
        self._v: list[Optional[torch.Tensor]] = []

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._k) <= layer_idx:
            self._k.append(None)
            self._v.append(None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)
        if self._k[layer_idx] is None:
            self._k[layer_idx] = key_states
            self._v[layer_idx] = value_states
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], key_states], dim=-2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], value_states], dim=-2)
        return self._k[layer_idx], self._v[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._k) or self._k[layer_idx] is None:
            return 0
        return int(self._k[layer_idx].shape[-2])


__all__ = ["ACT2FN", "Cache", "SimpleDynamicCache", "get_activation"]
