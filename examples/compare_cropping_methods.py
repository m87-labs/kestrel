#!/usr/bin/env python3
"""Compare CPU PIL-based cropping with proposed GPU grid-sample approach."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kestrel.moondream.config import VisionConfig
from kestrel.moondream.image_crops import overlap_crop_image, select_tiling


def _to_cuda_image(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy image to CUDA float tensor in NCHW, [0,1]."""
    tensor = torch.from_numpy(image).to(torch.uint8)
    tensor = tensor.permute(2, 0, 1).contiguous()
    tensor = tensor.to(torch.float32)
    tensor = tensor.unsqueeze(0)
    return tensor.to(device="cuda", non_blocking=True) / 255.0


def _linspace(start: float, stop: float, num: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(start, stop, num, device=device)


def _make_local_grid(
    crop_size: int,
    y0: float,
    x0: float,
    scale_y: float,
    scale_x: float,
    original_h: int,
    original_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a normalized sampling grid for a local tile."""
    # Centers in the resized coordinate system
    base = torch.arange(crop_size, device=device, dtype=torch.float32) + 0.5
    y_centers = (base + y0) * scale_y - 0.5
    x_centers = (base + x0) * scale_x - 0.5

    y_norm = ((y_centers + 0.5) / original_h) * 2.0 - 1.0
    x_norm = ((x_centers + 0.5) / original_w) * 2.0 - 1.0

    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def _make_grids(
    height: int,
    width: int,
    crop_size: int,
    window: int,
    margin: int,
    h_tiles: int,
    w_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    """Build normalized sampling grids for all crops (global + local)."""
    grids: list[torch.Tensor] = []

    # Global crop covers the full image
    global_y = _linspace(-1.0 + 1.0 / crop_size, 1.0 - 1.0 / crop_size, crop_size, device)
    global_x = _linspace(-1.0 + 1.0 / crop_size, 1.0 - 1.0 / crop_size, crop_size, device)
    gy, gx = torch.meshgrid(global_y, global_x, indexing="ij")
    grids.append(torch.stack((gx, gy), dim=-1))

    target_h = h_tiles * window + 2 * margin
    target_w = w_tiles * window + 2 * margin
    scale_y = height / target_h
    scale_x = width / target_w

    for i in range(h_tiles):
        for j in range(w_tiles):
            y0 = i * window
            x0 = j * window
            grid = _make_local_grid(
                crop_size=crop_size,
                y0=y0,
                x0=x0,
                scale_y=scale_y,
                scale_x=scale_x,
                original_h=height,
                original_w=width,
                device=device,
            )
            grids.append(grid)

    return torch.stack(grids, dim=0)


def gpu_crops(
    image: np.ndarray,
    *,
    overlap_margin: int,
    max_crops: int,
    crop_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    device = torch.device("cuda")
    img = _to_cuda_image(image)

    margin = overlap_margin * patch_size
    window = crop_size - 2 * margin

    h_tiles, w_tiles = select_tiling(
        image.shape[0] - 2 * margin,
        image.shape[1] - 2 * margin,
        window,
        max_crops,
    )

    grids = _make_grids(
        height=image.shape[0],
        width=image.shape[1],
        crop_size=crop_size,
        window=window,
        margin=margin,
        h_tiles=h_tiles,
        w_tiles=w_tiles,
        device=device,
    )

    crops = F.grid_sample(
        img.expand(grids.shape[0], -1, -1, -1),
        grids,
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    )
    return crops, (h_tiles, w_tiles)


def compare(image_path: Path, config: VisionConfig) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    cpu_crops, cpu_tiling = overlap_crop_image(
        image,
        overlap_margin=config.overlap_margin,
        max_crops=config.max_crops,
        base_size=(config.crop_size, config.crop_size),
        patch_size=config.enc_patch_size,
    )
    gpu_crop_tensor, gpu_tiling = gpu_crops(
        image,
        overlap_margin=config.overlap_margin,
        max_crops=config.max_crops,
        crop_size=config.crop_size,
        patch_size=config.enc_patch_size,
    )

    assert cpu_tiling == gpu_tiling, f"Tiling mismatch: CPU {cpu_tiling} vs GPU {gpu_tiling}"

    cpu_tensor = torch.from_numpy(cpu_crops).to(torch.float32) / 255.0
    cpu_tensor = cpu_tensor.permute(0, 3, 1, 2).to(device="cuda")

    diff = (gpu_crop_tensor - cpu_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    mse = (diff ** 2).mean().item()

    print(f"Image: {image_path}")
    print(f"  tiling: {cpu_tiling}, crops: {cpu_crops.shape[0]}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print(f"  max abs diff : {max_diff:.6f}")
    print(f"  mse          : {mse:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU cropping outputs")
    parser.add_argument("images", nargs="+", type=Path, help="Image paths to test")
    args = parser.parse_args()

    config = VisionConfig()
    for image_path in args.images:
        compare(image_path, config)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison script")
    main()
