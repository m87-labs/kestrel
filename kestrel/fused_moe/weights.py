
import torch
import torch.nn as nn


class ExpertWeights(nn.Module):
    """Simple container for per-expert weight tensors."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size, dtype=dtype)
        )


class ExpertWeightsFp8E4M3FN(nn.Module):
    """Per-expert FP8 weights stored as raw uint8 bits + per-output-channel scales.

    Storage format:
      - weight: uint8 view of float8_e4m3fn values representing (W / scale)
      - scale: per-(expert, out_channel) scale to reconstruct W ≈ fp8(weight) * scale
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size, dtype=torch.uint8),
            requires_grad=False,
        )
        self.register_buffer(
            "scale",
            torch.empty(num_experts, output_size, dtype=scale_dtype),
            persistent=True,
        )

    @property
    def fp8_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def dequantize(self, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        w_fp8 = self.weight.view(self.fp8_dtype)
        return w_fp8.to(dtype) * self.scale.to(dtype).unsqueeze(-1)


__all__ = ["ExpertWeights", "ExpertWeightsFp8E4M3FN"]
