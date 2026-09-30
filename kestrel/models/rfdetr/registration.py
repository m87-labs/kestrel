"""Resolve the RF-DETR implementation only when selected."""

from kestrel.models.registry import ModelSpec, register_builtin
from .config import CONFIGS
from .runtime import create_rfdetr_runtime

for variant in CONFIGS:
    register_builtin(ModelSpec(name=f"rfdetr-{variant}", runtime=create_rfdetr_runtime,
                               needs_kv_pool=False))
