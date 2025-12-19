"""Region encoding and decoding utilities for spatial skills."""


from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import RegionConfig


@dataclass(frozen=True)
class SpatialDecodeTables:
    coord_value_lut: Tensor
    size_value_lut: Tensor
    coord_logits_dim: int
    weight: Tensor
    bias: Tensor


def fourier_features(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    f = 2 * math.pi * x @ w
    return torch.cat([f.cos(), f.sin()], dim=-1)


def encode_coordinate(coord: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["coord_encoder"](fourier_features(coord, module.coord_features))


def encode_size(size: torch.Tensor, module: nn.ModuleDict) -> torch.Tensor:
    return module["size_encoder"](fourier_features(size, module.size_features))


def build_spatial_decode_tables(
    module: nn.ModuleDict,
) -> SpatialDecodeTables:
    coord_decoder = module["coord_decoder"]
    size_decoder = module["size_decoder"]

    weight = torch.cat((coord_decoder.weight, size_decoder.weight), dim=0).contiguous()
    bias = torch.cat((coord_decoder.bias, size_decoder.bias), dim=0).contiguous()

    coord_bins = int(coord_decoder.out_features)
    device = module.coord_features.device
    coord_dtype = module.coord_features.dtype
    coord_value_lut = torch.linspace(
        0.0, 1.0, coord_bins, device=device, dtype=torch.float32
    ).to(dtype=coord_dtype)

    size_bins = int(size_decoder.out_features // 2)
    size_device = module.size_features.device
    size_dtype = module.size_features.dtype
    size_exponents = torch.linspace(
        -10.0, 0.0, size_bins, device=size_device, dtype=torch.float32
    )
    size_value_lut = torch.exp2(size_exponents).to(dtype=size_dtype)

    return SpatialDecodeTables(
        coord_value_lut=coord_value_lut,
        size_value_lut=size_value_lut,
        coord_logits_dim=coord_bins,
        weight=weight,
        bias=bias,
    )


def spatial_decode_logits(
    hidden_state: Tensor,
    tables: SpatialDecodeTables,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden = hidden_state.unsqueeze(0) if hidden_state.ndim == 1 else hidden_state
    logits = F.linear(hidden, tables.weight, tables.bias)

    coord_logits = logits[:, : tables.coord_logits_dim]
    size_flat = logits[:, tables.coord_logits_dim :]
    bins = int(size_flat.shape[-1] // 2)
    width_logits = size_flat[:, :bins]
    height_logits = size_flat[:, bins:]
    return coord_logits, width_logits, height_logits


def spatial_bins_to_values(
    coord_bins: Tensor,
    width_bins: Tensor,
    height_bins: Tensor,
    tables: SpatialDecodeTables,
    *,
    out_coord: Tensor,
    out_size: Tensor,
) -> tuple[Tensor, Tensor]:
    coord_out = out_coord.view(-1)
    torch.index_select(tables.coord_value_lut, 0, coord_bins, out=coord_out)
    torch.index_select(tables.size_value_lut, 0, width_bins, out=out_size[:, 0])
    torch.index_select(tables.size_value_lut, 0, height_bins, out=out_size[:, 1])
    return out_coord, out_size


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
    "SpatialDecodeTables",
    "fourier_features",
    "encode_coordinate",
    "encode_size",
    "build_spatial_decode_tables",
    "spatial_decode_logits",
    "spatial_bins_to_values",
    "build_region_module",
]
