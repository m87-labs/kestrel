"""Minimal Moondream text-only model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MoondreamTextConfig
from .text import build_text_model


class MoondreamTextModel(nn.Module):
    """Container for the Moondream text transformer."""

    def __init__(
        self,
        config: MoondreamTextConfig,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        setup_caches: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.text = build_text_model(config.text, dtype)
        if device is not None:
            self.to(device=device)
        if setup_caches:
            self._setup_caches()

    def _setup_caches(self) -> None:
        for block in self.text.blocks:
            block.kv_cache = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


__all__ = ["MoondreamTextModel"]
