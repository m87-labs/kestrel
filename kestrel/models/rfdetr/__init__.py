"""Single-pass RF-DETR COCO detection on Hopper."""

from kestrel.models.registry import register_lazy
from .config import CONFIGS

register_lazy([f"rfdetr-{variant}" for variant in CONFIGS], __name__ + ".registration")
