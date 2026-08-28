"""Pinned Hugging Face checkpoint resolution for Kestrel ASR models."""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(
    checkpoint: str | Path,
    *,
    revision: str | None,
    filenames: tuple[str, ...],
    local_files_only: bool = False,
) -> Path:
    path = Path(checkpoint).expanduser()
    if path.exists():
        root = path if path.is_dir() else path.parent
    else:
        if revision is None:
            raise ValueError("revision is required for a remote checkpoint")
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                str(checkpoint),
                revision=revision,
                allow_patterns=list(filenames),
                local_files_only=local_files_only,
            )
        )
    missing = [name for name in filenames if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing files: {missing}")
    return root.resolve()


__all__ = ["resolve_checkpoint"]
