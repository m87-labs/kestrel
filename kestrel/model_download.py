"""Automatic model weight downloading via HuggingFace Hub."""

from pathlib import Path


def ensure_model_weights(
    model: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download model weights if not already cached, returning the local path.

    By default, looks up the HuggingFace repo and filename for the given
    model name via the model registry. Callers can override repo_id,
    filename, and revision for custom checkpoints (e.g. pinned revisions
    in production).

    The cache location is controlled by the standard HF_HOME environment
    variable (defaults to ~/.cache/huggingface).
    """
    from huggingface_hub import hf_hub_download

    if repo_id is None or filename is None:
        from kestrel.models import get_spec

        spec = get_spec(model)
        if repo_id is None:
            repo_id = spec.repo_id
        if filename is None:
            filename = spec.filename

    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision

    return Path(hf_hub_download(repo_id, filename=filename, **kwargs))
