"""Resolve the DINOv2 implementation only when selected."""

from kestrel.models.registry import ModelSpec, register_builtin
from .factory import create_dinov2_runtime
from .metadata import DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID

for name in (DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID):
    register_builtin(ModelSpec(name=name, runtime=create_dinov2_runtime, needs_kv_pool=False))
