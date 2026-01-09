"""Quantize MoE weights to FP8 using HQQ and save to a new checkpoint.

This script takes a BF16 model checkpoint and produces a new checkpoint with:
- FP8 quantized MoE expert weights (up_experts and down_experts)
- Per-row scales for each expert weight tensor

The scales are stored as:
- text_model.transformer.h.{i}.moe_quant.up_scale  [E, N]
- text_model.transformer.h.{i}.moe_quant.down_scale  [E, N]

Usage:
  python scripts/quantize_moe_fp8.py model.pt model_fp8.pt
  python scripts/quantize_moe_fp8.py model.pt model_fp8.pt --hqq-iters 30
"""

import argparse
import sys
from pathlib import Path

import torch


FP8_MAX = 448.0  # E4M3 max representable value


def quantize_fp8_hqq(
    x: torch.Tensor,
    axis: int = -1,
    iters: int = 20,
    lp_norm: float = 0.7,
    beta: float = 10.0,
    kappa: float = 1.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HQQ-style optimized FP8 quantization.

    Uses proximal Lp optimization to find better scales that minimize
    quantization error, rather than just using max values.

    Args:
        x: Input tensor to quantize
        axis: Axis along which to compute scales (default: -1, per-row)
        iters: Number of optimization iterations
        lp_norm: Lp norm for shrinkage
        beta: Initial shrinkage strength
        kappa: Beta growth factor per iteration

    Returns:
        Tuple of (quantized FP8 tensor, scales)
    """
    abs_max = x.abs().amax(dim=axis, keepdim=True).clamp(min=1e-12)
    scale = abs_max / FP8_MAX

    x_f = x.float()
    best_scale = scale.clone()
    best_error = float("inf")

    for _ in range(iters):
        x_scaled = x_f / scale
        x_q = x_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        x_r = x_q.float() * scale

        error = x_f - x_r
        if lp_norm == 1:
            shrunk = torch.sign(error) * torch.relu(error.abs() - 1.0 / beta)
        else:
            shrunk = torch.sign(error) * torch.relu(
                error.abs() - (1.0 / beta) * error.abs().pow(lp_norm - 1)
            )

        x_adjusted = x_f - shrunk
        new_abs_max = x_adjusted.abs().amax(dim=axis, keepdim=True).clamp(min=1e-12)
        scale = new_abs_max / FP8_MAX

        current_error = error.abs().mean().item()
        if current_error < best_error:
            best_error = current_error
            best_scale = scale.clone()

        beta *= kappa

    x_fp8 = (x / best_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, best_scale.squeeze(axis).to(torch.float32)


def quantize_checkpoint(
    input_path: Path,
    output_path: Path,
    hqq_iters: int = 20,
    device: str = "cuda",
) -> None:
    """Quantize MoE weights in a checkpoint to FP8.

    Args:
        input_path: Path to input BF16 checkpoint
        output_path: Path to save FP8 checkpoint
        hqq_iters: Number of HQQ optimization iterations
        device: Device to use for quantization
    """
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=True)

    # Find MoE weight tensors
    moe_up_keys = sorted([k for k in checkpoint.keys() if "mlp.experts.weight" in k])
    moe_down_keys = sorted([k for k in checkpoint.keys() if "mlp.output_experts.weight" in k])

    print(f"Found {len(moe_up_keys)} up_experts tensors")
    print(f"Found {len(moe_down_keys)} down_experts tensors")

    if not moe_up_keys and not moe_down_keys:
        print("No MoE weights found in checkpoint!")
        sys.exit(1)

    # Quantize each MoE weight tensor
    new_tensors = {}

    for key in moe_up_keys:
        print(f"  Quantizing {key}...")
        weight = checkpoint[key].to(device).to(torch.bfloat16)
        # Shape: [E, N, K] where K is the input dim (quantize along K)
        E, N, K = weight.shape

        # Quantize along last dimension (K) with per-row scales
        weight_fp8, scale = quantize_fp8_hqq(weight, axis=-1, iters=hqq_iters)

        # Store FP8 weights and scales
        new_tensors[key] = weight_fp8.cpu()

        # Extract layer index for scale key
        # Key format: text_model.transformer.h.{i}.mlp.experts.weight
        parts = key.split(".")
        layer_idx = parts[3]  # The index after "h."
        scale_key = f"text_model.transformer.h.{layer_idx}.moe_quant.up_scale"
        new_tensors[scale_key] = scale.cpu()  # [E, N]

        # Report quantization stats
        weight_dq = weight_fp8.float() * scale.unsqueeze(-1)
        error = (weight.float() - weight_dq).abs()
        print(f"    Shape: {list(weight.shape)}, Scale shape: {list(scale.shape)}")
        print(f"    Quant error: mean={error.mean().item():.6f}, max={error.max().item():.4f}")

    for key in moe_down_keys:
        print(f"  Quantizing {key}...")
        weight = checkpoint[key].to(device).to(torch.bfloat16)
        # Shape: [E, N, K] where K is the input dim (quantize along K)
        E, N, K = weight.shape

        # Quantize along last dimension (K) with per-row scales
        weight_fp8, scale = quantize_fp8_hqq(weight, axis=-1, iters=hqq_iters)

        # Store FP8 weights and scales
        new_tensors[key] = weight_fp8.cpu()

        # Extract layer index for scale key
        parts = key.split(".")
        layer_idx = parts[3]
        scale_key = f"text_model.transformer.h.{layer_idx}.moe_quant.down_scale"
        new_tensors[scale_key] = scale.cpu()  # [E, N]

        # Report quantization stats
        weight_dq = weight_fp8.float() * scale.unsqueeze(-1)
        error = (weight.float() - weight_dq).abs()
        print(f"    Shape: {list(weight.shape)}, Scale shape: {list(scale.shape)}")
        print(f"    Quant error: mean={error.mean().item():.6f}, max={error.max().item():.4f}")

    # Build output checkpoint: copy non-MoE tensors, add quantized MoE tensors
    output_checkpoint = {}
    for key, tensor in checkpoint.items():
        if "mlp.experts.weight" in key or "mlp.output_experts.weight" in key:
            # Replace with FP8 version
            output_checkpoint[key] = new_tensors[key]
        else:
            # Keep original
            output_checkpoint[key] = tensor

    # Add scale tensors
    for key, tensor in new_tensors.items():
        if "moe_quant" in key:
            output_checkpoint[key] = tensor

    print(f"\nSaving quantized checkpoint: {output_path}")
    print(f"  Original keys: {len(checkpoint)}")
    print(f"  Output keys: {len(output_checkpoint)}")

    torch.save(output_checkpoint, str(output_path))
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize MoE weights to FP8 using HQQ"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input BF16 model checkpoint (.pt)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to save FP8 model checkpoint (.pt)",
    )
    parser.add_argument(
        "--hqq-iters",
        type=int,
        default=20,
        help="Number of HQQ optimization iterations (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for quantization (default: cuda)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    quantize_checkpoint(
        args.input,
        args.output,
        hqq_iters=args.hqq_iters,
        device=args.device,
    )


if __name__ == "__main__":
    main()
