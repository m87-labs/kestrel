"""Local Qwen 3.5 image preprocessing."""

import math
from dataclasses import dataclass
from typing import Any

import kestrel_native
import numpy as np
import torch
from kestrel.utils.image import decode_to_srgb
from torch.nn import functional as F


@dataclass(frozen=True)
class QwenImageProcessorConfig:
    shortest_edge: int = 65536
    longest_edge: int = 16777216
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(
    image: Any,
    config: QwenImageProcessorConfig = QwenImageProcessorConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = np.ascontiguousarray(decode_to_srgb(image))
    height, width = rgb.shape[:2]
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=config.patch_size * config.merge_size,
        min_pixels=config.shortest_edge,
        max_pixels=config.longest_edge,
    )
    resized = kestrel_native.resize_bicubic(rgb, resized_h, resized_w)
    array = resized.astype(np.float32)
    array *= 1.0 / 255.0
    array = (array - np.asarray(config.image_mean, dtype=np.float32)) / np.asarray(
        config.image_std, dtype=np.float32
    )
    patches = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    batch_size, channels = patches.shape[:2]
    grid_h = resized_h // config.patch_size
    grid_w = resized_w // config.patch_size
    patches = patches.reshape(
        batch_size,
        channels,
        grid_h // config.merge_size,
        config.merge_size,
        config.patch_size,
        grid_w // config.merge_size,
        config.merge_size,
        config.patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flattened = (
        patches.unsqueeze(6)
        .expand(
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            config.temporal_patch_size,
            -1,
            -1,
        )
        .reshape(
            batch_size,
            grid_h * grid_w,
            channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size,
        )
    )
    pixel_values = flattened.reshape(-1, flattened.shape[-1])
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return pixel_values, image_grid_thw


def vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int | torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_ids is not None:
        return position_ids
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        merge_sizes = [int(spatial_merge_size)] * int(grid_thw.shape[0])
    else:
        merge_sizes = [int(value) for value in spatial_merge_size.tolist()]
    parts = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), merge_sizes):
        t, h, w, merge_size = int(t), int(h), int(w), int(merge_size)
        h_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        h_ids = h_ids.reshape(
            h // merge_size, merge_size, w // merge_size, merge_size
        ).transpose(1, 2).flatten()
        w_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        w_ids = w_ids.reshape(
            h // merge_size, merge_size, w // merge_size, merge_size
        ).transpose(1, 2).flatten()
        parts.append(torch.stack((h_ids, w_ids), dim=-1).repeat(t, 1))
    return torch.cat(parts, dim=0)


def vision_bilinear_coordinates(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    indices: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices is not None and weights is not None:
        return indices, weights
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = (
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        )
        corner_weights = (
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        )
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for corner in range(4):
            index_parts[corner].append(corner_indices[corner][reorder])
            weight_parts[corner].append(corner_weights[corner][reorder])
    return (
        torch.stack([torch.cat(part) for part in index_parts]),
        torch.stack([torch.cat(part) for part in weight_parts]),
    )


def vision_cu_seqlens(
    grid_thw: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        return cu_seqlens
    lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(lengths, (1, 0), value=0)


__all__ = [
    "QwenImageProcessorConfig",
    "preprocess_image",
    "smart_resize",
    "vision_bilinear_coordinates",
    "vision_cu_seqlens",
    "vision_position_ids",
]
