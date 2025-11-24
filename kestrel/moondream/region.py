"""Region encoding and decoding utilities for spatial skills."""


import math
from typing import Iterable, List, Tuple, Union

import torch
import torch.nn as nn

from .config import RegionConfig

SpatialRef = Union[Tuple[float, float], Tuple[float, float, float, float]]


def fourier_features(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    f = 2 * math.pi * x @ w
    return torch.cat([f.cos(), f.sin()], dim=-1)


def encode_coordinate(coord: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["coord_encoder"](fourier_features(coord, module.coord_features))


def decode_coordinate(hidden_state: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["coord_decoder"](hidden_state)


def encode_size(size: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["size_encoder"](fourier_features(size, module.size_features))


def decode_size(hidden_state: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["size_decoder"](hidden_state).view(2, -1)


def encode_spatial_refs(
    spatial_refs: Iterable[SpatialRef], module: nn.ModuleDict
) -> dict[str, torch.Tensor | None]:
    coords: List[float] = []
    sizes: List[Tuple[float, float]] = []
    for ref in spatial_refs:
        if len(ref) == 2:
            coords.extend(ref)
        else:
            x_min, y_min, x_max, y_max = ref
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            coords.extend([x_c, y_c])
            sizes.append((width, height))

    coord_tensor = torch.tensor(coords, device=module.coord_features.device, dtype=module.coord_features.dtype).view(-1, 1)
    coord_enc = encode_coordinate(coord_tensor, module)

    if sizes:
        size_tensor = torch.tensor(sizes, device=module.size_features.device, dtype=module.size_features.dtype)
        size_enc = encode_size(size_tensor, module)
    else:
        size_enc = None

    return {"coords": coord_enc, "sizes": size_enc}


def build_region_module(config: RegionConfig, dtype: torch.dtype) -> nn.ModuleDict:
    module = nn.ModuleDict(
        {
            "coord_encoder": nn.Linear(config.coord_feat_dim, config.dim, dtype=dtype),
            "coord_decoder": nn.Linear(config.dim, config.coord_out_dim, dtype=dtype),
            "size_encoder": nn.Linear(config.size_feat_dim, config.dim, dtype=dtype),
            "size_decoder": nn.Linear(config.dim, config.size_out_dim, dtype=dtype),
        }
    )

    coord_feats = torch.empty(config.coord_feat_dim // 2, 1, dtype=dtype).T
    size_feats = torch.empty(config.size_feat_dim // 2, 2, dtype=dtype).T
    module.coord_features = nn.Parameter(coord_feats)
    module.size_features = nn.Parameter(size_feats)
    return module


__all__ = [
    "SpatialRef",
    "fourier_features",
    "encode_coordinate",
    "decode_coordinate",
    "encode_size",
    "decode_size",
    "encode_spatial_refs",
    "build_region_module",
]
