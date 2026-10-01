"""Resolve the standalone SigLIP runtime only when selected."""
from kestrel.models.registry import ModelSpec, register_builtin
from .runtime import create_siglip_runtime

register_builtin(ModelSpec(name="siglip-so400m-378", runtime=create_siglip_runtime,
                           needs_kv_pool=False))
