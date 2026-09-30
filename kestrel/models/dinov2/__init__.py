"""Inference-only DINOv2 ViT-S/14 image embeddings."""

from kestrel.models.registry import register_lazy
from .metadata import DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID

register_lazy([DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID], __name__ + ".registration")
