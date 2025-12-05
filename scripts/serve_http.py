#!/usr/bin/env python3
"""
Convenience launcher for the Kestrel HTTP server.

Configuration is taken from environment variables:
  KESTREL_WEIGHTS   (required) path to the weights checkpoint
  KESTREL_DEVICE    (default: cuda:0)
  KESTREL_DTYPE     (default: bfloat16)
  KESTREL_MAX_BATCH (default: 2)
  KESTREL_MAX_NEW   (default: 768)
  KESTREL_TEMP      (default: 0.2)
  KESTREL_TOP_P     (default: 0.9)
  KESTREL_PORT      (default: 8000)
  KESTREL_HOST      (default: 0.0.0.0)
  KESTREL_CUDA_GRAPHS (default: false)
"""

import os
from pathlib import Path

import torch
import uvicorn

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.server.http import create_app


def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def main() -> None:
    weights_env = os.getenv("KESTREL_WEIGHTS")
    if not weights_env:
        raise SystemExit("KESTREL_WEIGHTS must be set to the checkpoint path")
    weights = Path(weights_env).expanduser()

    device = os.getenv("KESTREL_DEVICE", "cuda:0")
    dtype_name = os.getenv("KESTREL_DTYPE", "bfloat16").lower()
    dtype = torch.bfloat16 if dtype_name in ("bfloat16", "bf16") else torch.float16
    max_batch = int(os.getenv("KESTREL_MAX_BATCH", "2"))
    max_new = int(os.getenv("KESTREL_MAX_NEW", "768"))
    default_temp = float(os.getenv("KESTREL_TEMP", "0.2"))
    default_top_p = float(os.getenv("KESTREL_TOP_P", "0.9"))
    port = int(os.getenv("KESTREL_PORT", "8000"))
    host = os.getenv("KESTREL_HOST", "0.0.0.0")
    enable_cuda_graphs = env_bool("KESTREL_CUDA_GRAPHS", False)

    runtime_cfg = RuntimeConfig(
        model_paths=ModelPaths(
            weights=str(weights),
        ),
        device=device,
        dtype=dtype,
        max_batch_size=max_batch,
        enable_cuda_graphs=enable_cuda_graphs,
    )

    app = create_app(
        runtime_cfg,
        default_max_new_tokens=max_new,
        default_temperature=default_temp,
        default_top_p=default_top_p,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
