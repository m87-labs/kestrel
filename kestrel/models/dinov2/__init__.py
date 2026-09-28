"""Inference-only DINOv2 ViT-S/14 image embeddings."""

from kestrel.models.registry import ModelSpec, register

from .factory import create_dinov2_runtime
from .weights import DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID


for _name in (DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID):
    register(ModelSpec(name=_name, runtime=create_dinov2_runtime, needs_kv_pool=False))


__all__ = ["create_dinov2_runtime"]
