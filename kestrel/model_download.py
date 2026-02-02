"""Automatic model weight downloading via HuggingFace Hub."""

import os
from pathlib import Path


MODELS = {
    "moondream2": ("vikhyatk/moondream2", "model.safetensors"),
    "moondream3-preview": ("moondream/moondream3-preview", "model_fp8.pt"),
}


def ensure_model_weights(
    model: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download model weights if not already cached, returning the local path.

    By default, looks up the HuggingFace repo and filename for the given model
    name. Callers can override repo_id, filename, and revision for custom
    checkpoints (e.g. pinned revisions in production).

    The cache location is controlled by the standard HF_HOME environment
    variable (defaults to ~/.cache/huggingface).
    """
    from huggingface_hub import hf_hub_download

    if repo_id is None or filename is None:
        if model not in MODELS:
            raise ValueError(
                f"Unknown model {model!r}, expected one of {list(MODELS)}"
            )
        default_repo, default_filename = MODELS[model]
        if repo_id is None:
            repo_id = default_repo
        if filename is None:
            filename = default_filename

    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision

    return Path(hf_hub_download(repo_id, filename=filename, **kwargs))
