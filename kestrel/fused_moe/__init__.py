"""Fused MoE kernels adapted from vLLM for single-GPU decode.

Keep imports lazy so environments that only run Moondream 2 do not require
Triton at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FusedMoEModule",
    "FusedMoEConfig",
    "ExpertWeights",
    "ExpertWeightsFp8E4M3FN",
    "preallocate_shared_moe_workspaces",
]


def __getattr__(name: str) -> Any:
    if name in {"FusedMoEModule", "FusedMoEConfig", "preallocate_shared_moe_workspaces"}:
        from .module import (
            FusedMoEConfig,
            FusedMoEModule,
            preallocate_shared_moe_workspaces,
        )

        exports = {
            "FusedMoEModule": FusedMoEModule,
            "FusedMoEConfig": FusedMoEConfig,
            "preallocate_shared_moe_workspaces": preallocate_shared_moe_workspaces,
        }
        return exports[name]

    if name in {"ExpertWeights", "ExpertWeightsFp8E4M3FN"}:
        from .weights import ExpertWeights, ExpertWeightsFp8E4M3FN

        exports = {
            "ExpertWeights": ExpertWeights,
            "ExpertWeightsFp8E4M3FN": ExpertWeightsFp8E4M3FN,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
