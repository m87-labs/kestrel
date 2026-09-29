"""Single-pass RF-DETR COCO detection on Hopper."""
from kestrel.models.registry import ModelSpec, register
from .config import CONFIGS
from .runtime import create_rfdetr_runtime

for _variant in CONFIGS:
    register(ModelSpec(name=f'rfdetr-{_variant}', runtime=create_rfdetr_runtime, needs_kv_pool=False))
