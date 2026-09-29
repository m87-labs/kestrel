"""Resolve Kokoro implementation only when selected."""

from kestrel.models.registry import ModelSpec, register_builtin

from .metadata import DEFAULT_KOKORO_MODEL, DEFAULT_KOKORO_REPO_ID, DEFAULT_KOKORO_REVISION
from .orchestrator import build_orchestrators
from .runtime import create_kokoro_runtime

register_builtin(ModelSpec(
    name=DEFAULT_KOKORO_MODEL,
    repo_id=DEFAULT_KOKORO_REPO_ID,
    revision=DEFAULT_KOKORO_REVISION,
    runtime=create_kokoro_runtime,
    orchestrators=build_orchestrators,
))
