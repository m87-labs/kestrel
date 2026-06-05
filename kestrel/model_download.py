"""Automatic model weight downloading via HuggingFace Hub."""

from pathlib import Path
from typing import Optional


def ensure_model_weights(
    model: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> Optional[Path]:
    """Download model weights if not already cached, returning the local path.

    By default, looks up the HuggingFace repo and filename for the given
    model name via the model registry. Callers can override repo_id,
    filename, and revision for custom checkpoints (e.g. pinned revisions
    in production).

    Returns ``None`` when the model declares no single weight file — a
    single-pass model whose runtime factory owns its own loading (e.g. via
    ``snapshot_download``) registers no ``filename``, so there is nothing for
    this autoregressive-style downloader to fetch.

    The cache location is controlled by the standard HF_HOME environment
    variable (defaults to ~/.cache/huggingface).
    """
    if repo_id is None or filename is None:
        from kestrel.models import get_spec

        spec = get_spec(model)
        if repo_id is None:
            repo_id = spec.repo_id
        if filename is None:
            filename = spec.filename

    if repo_id is None or filename is None:
        return None

    from huggingface_hub import hf_hub_download

    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision

    return Path(hf_hub_download(repo_id, filename=filename, **kwargs))


def probe_supported_model_configs() -> None:
    """Best-effort access check for supported HuggingFace model configs."""
    try:
        from huggingface_hub import hf_hub_download

        from kestrel.models import get_spec, known_models

        inaccessible: list[str] = []
        for model in known_models():
            try:
                spec = get_spec(model)
                if spec.repo_id is None:
                    continue
                hf_hub_download(
                    spec.repo_id,
                    filename="config.json",
                    etag_timeout=2,
                )
            except Exception:
                inaccessible.append(model)

        if inaccessible:
            # TODO: Log a warning once dynamic model loading can route requests
            # to any supported Moondream model at runtime.
            pass
    except Exception:
        return
