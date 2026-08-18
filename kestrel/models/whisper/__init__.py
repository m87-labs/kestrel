"""First-class Whisper large-v3-turbo inference for Kestrel.

Kestrel owns the model, transcription behavior, and runtime orchestration.
Optimized distributions register the generated execution backend explicitly;
the eager correctness oracle lives only in the test package.
"""

from __future__ import annotations

import traceback
from typing import Any

from kestrel.models.registry import ModelSpec, register

from .assets import CHECKPOINT_REVISION, MODEL_NAME, REPO_ID
from .runtime_abi import register_backend


def _runtime_factory(cfg: Any, **kwargs: Any) -> Any:
    from .runtime import WhisperRuntime

    runtime = WhisperRuntime(cfg, **kwargs)
    try:
        runtime.warmup()
    except BaseException as warmup_error:
        shutdown_error = runtime._abort_failed_warmup()
        if shutdown_error is not None:
            try:
                warmup_error.add_note(
                    "Whisper shutdown after warmup failure also failed: "
                    f"{shutdown_error!r}"
                )
            except BaseException:
                pass
        traceback.clear_frames(warmup_error.__traceback__)
        # A retained startup exception must not keep the unpublished runtime,
        # its engine-owned inputs, or its injected test components alive
        # through this factory frame.
        del cfg, kwargs, runtime, shutdown_error
        raise
    return runtime


def _build_skill_registry():
    from .skill import build_skill_registry

    return build_skill_registry()


register(
    ModelSpec(
        name=MODEL_NAME,
        repo_id=REPO_ID,
        revision=CHECKPOINT_REVISION,
        filename=None,
        checkpoint_format="whisper_safetensors",
        tokenizer_id=REPO_ID,
        runtime=_runtime_factory,
        skills=_build_skill_registry,
    )
)


__all__ = [
    "CHECKPOINT_REVISION",
    "MODEL_NAME",
    "REPO_ID",
    "register_backend",
]
