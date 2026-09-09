"""Inference storage for block-scaled E4M3 linear projections."""

import torch
from torch import nn

from kestrel_kernels import get_runtime


_block_scaled_linear = get_runtime().linear.block_scaled_linear


class BlockScaledLinear(nn.Module):
    """Keep FP8 checkpoint bytes and an optional unquantized row suffix.

    Scales are stored as [parts, row blocks, column blocks]. With two parts,
    eight-row blocks alternate between the parts, as in a fused gated MLP.
    Eager inference dispatches the packed projection through the active runtime;
    generated decode binds the same packed storage directly.
    """

    block_size = 128

    def __init__(
        self, in_features: int, out_features: int, *,
        quantized_rows: int | None = None, interleaved_parts: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.quantized_rows = (
            self.out_features if quantized_rows is None else int(quantized_rows)
        )
        self.interleaved_parts = int(interleaved_parts)
        if min(self.in_features, self.out_features, self.quantized_rows) <= 0:
            raise ValueError("block-scaled linear dimensions must be positive")
        if self.quantized_rows > self.out_features:
            raise ValueError("quantized rows exceed the projection extent")
        if self.interleaved_parts not in (1, 2):
            raise ValueError("block-scaled linear supports one or two row parts")
        if self.interleaved_parts == 2 and (
            self.quantized_rows != self.out_features or self.out_features % 16
        ):
            raise ValueError("interleaved projections require complete eight-row pairs")
        rows_per_part = self.quantized_rows // self.interleaved_parts
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        self.register_buffer("weight_scale_inv", torch.empty(
            self.interleaved_parts,
            (rows_per_part + self.block_size - 1) // self.block_size,
            (self.in_features + self.block_size - 1) // self.block_size,
            dtype=torch.float32,
        ))
        tail_rows = self.out_features - self.quantized_rows
        self.register_buffer("weight_tail", (
            torch.empty(tail_rows, self.in_features, dtype=torch.bfloat16)
            if tail_rows else None
        ))
        self.bias = (
            nn.Parameter(torch.empty(self.out_features), requires_grad=False)
            if bias else None
        )

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) must preserve the checkpoint's FP32 scales and
        # BF16 suffix. Move their byte views so dtype conversion cannot round
        # them before the generated binder sees the original values.
        protected = {name: self._buffers.pop(name)
                     for name in ("weight_scale_inv", "weight_tail")}
        try:
            result = super()._apply(fn, recurse=recurse)
            for name, value in protected.items():
                self._buffers[name] = (
                    None if value is None else fn(value.view(torch.uint8)).view(value.dtype)
                )
        except BaseException:
            self._buffers.update(protected)
            raise
        return result

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("linear activation must be BF16, FP16, or FP32")
        rows = torch.arange(self.quantized_rows, device=self.weight.device)
        if self.interleaved_parts == 2:
            part = (rows // 8) % 2
            logical_row = (rows // 16) * 8 + rows % 8
        else:
            part = torch.zeros_like(rows)
            logical_row = rows
        scales = self.weight_scale_inv[part, logical_row // self.block_size]
        scales = scales.repeat_interleave(self.block_size, dim=1)[:, :self.in_features]
        prefix = (self.weight[:self.quantized_rows].view(torch.float8_e4m3fn).float() * scales).to(dtype)
        if self.weight_tail is None:
            return prefix
        return torch.cat((prefix, self.weight_tail.to(dtype)), dim=0)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return _block_scaled_linear(
            activation,
            self.weight,
            self.weight_scale_inv,
            quantized_rows=self.quantized_rows,
            interleaved_parts=self.interleaved_parts,
            weight_tail=self.weight_tail,
            bias=self.bias,
        )
