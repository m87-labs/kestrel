"""Runtime configuration objects for the Kestrel inference engine."""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ModelPaths:
    """Paths pointing to model assets required at runtime."""

    weights: Path
    config_json: Optional[Path] = None
    tokenizer: Optional[str | Path] = None
    reference_weights: Optional[Path] = None
    reference_config: Optional[Path] = None
    reference_tokenizer: Optional[str | Path] = None


@dataclass
class RuntimeConfig:
    """Knobs controlling the text-only inference prototype."""

    model_paths: ModelPaths
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    max_batch_size: int = 4
    page_size: int = 128
    max_seq_length: int = 32768
    enable_cuda_graphs: bool = True

    def resolved_dtype(self) -> torch.dtype:
        """Return the torch dtype to use for the runtime."""

        return self.dtype

    def resolved_device(self) -> torch.device:
        """Return the torch device requested for inference."""

        device = torch.device(self.device)
        # Ensure CUDA devices have an explicit index for torch.cuda.set_device()
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device


__all__ = ["ModelPaths", "RuntimeConfig"]
