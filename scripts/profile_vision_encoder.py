#!/usr/bin/env python3
"""Profile the vision encoder forward pass with NVTX ranges.

This script focuses on the GPU forward path for the vision encoder and
projection. It adds NVTX markers around patch embedding, each transformer
block component, reconstruction, and the projection MLP for Nsight Systems.

## Quick Start (nsys)

    cd ~/code/kestrel
    nsys profile --trace=cuda,nvtx --stats=true -o /tmp/vision_profile --force-overwrite=true \
        uv run python scripts/profile_vision_encoder.py \
          --weights ~/code/moondream/model.pt \
          --image-size 768

## Post-run reports

    # Per-NVTX GPU time (projected onto GPU timeline)
    nsys stats --report nvtx_gpu_proj_sum /tmp/vision_profile.nsys-rep

    # Per-kernel breakdown (top kernels)
    nsys stats --report cuda_gpu_kern_sum /tmp/vision_profile.nsys-rep

    # Filter to only the profiled iteration (excludes warmup/model load)
    nsys stats --report nvtx_gpu_proj_sum --filter-nvtx vision_forward /tmp/vision_profile.nsys-rep

## How to interpret

- Start with nvtx_gpu_proj_sum for GPU time per range (vision_encoder, layerX_*).
- Attention time splits into qkv + flash_attn + proj; MLP time is fc1+gelu+fc2.
- cuda_gpu_kern_sum highlights the heaviest kernels (often nvjet_* GEMMs and flash_attn).
- If H2D copies dominate, filter by `vision_forward` to avoid model load/warmup noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def patch_vision_encoder_with_nvtx(detailed: bool = True) -> None:
    """Monkey-patch vision_encoder to add NVTX markers."""
    import torch.nn.functional as F
    from kestrel.moondream import vision as vision_module

    def _vision_attn_detail(x: torch.Tensor, attn: torch.nn.ModuleDict, n_heads: int, prefix: str) -> torch.Tensor:
        nvtx.range_push(f"{prefix}_qkv")
        qkv = attn["qkv"](x)
        nvtx.range_pop()

        dim = x.shape[-1]
        head_dim = dim // n_heads
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(x.size(0), -1, n_heads, head_dim)
        k = k.view(x.size(0), -1, n_heads, head_dim)
        v = v.view(x.size(0), -1, n_heads, head_dim)

        nvtx.range_push(f"{prefix}_flash")
        out, _ = vision_module._flash_attn_fwd(q, k, v, causal=False)
        nvtx.range_pop()
        return out.reshape(x.size(0), -1, dim)

    def _vision_mlp_detail(x: torch.Tensor, mlp: torch.nn.ModuleDict, prefix: str) -> torch.Tensor:
        nvtx.range_push(f"{prefix}_fc1")
        x = mlp["fc1"](x)
        nvtx.range_pop()
        nvtx.range_push(f"{prefix}_gelu")
        x = F.gelu(x, approximate="tanh")
        nvtx.range_pop()
        nvtx.range_push(f"{prefix}_fc2")
        x = mlp["fc2"](x)
        nvtx.range_pop()
        return x

    def instrumented_vision_encoder(
        crops: torch.Tensor,
        module: torch.nn.Module,
        config: "vision_module.VisionConfig",
        *,
        early_layer: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        nvtx.range_push("vision_encoder")
        nvtx.range_push("create_patches")
        x = vision_module.create_patches(crops, config.enc_patch_size)
        nvtx.range_pop()

        nvtx.range_push("patch_embed")
        x = module.patch_emb(x)
        nvtx.range_pop()

        nvtx.range_push("add_pos_emb")
        x = x + module.pos_emb
        nvtx.range_pop()

        early = None
        use_fast_ln = x.is_cuda and x.dtype == torch.bfloat16 and not torch.is_grad_enabled()
        x_norm_buf: torch.Tensor | None = (
            torch.empty(x.shape, device=x.device, dtype=x.dtype) if use_fast_ln else None
        )

        def _layer_norm(x_in: torch.Tensor, ln: torch.nn.LayerNorm) -> torch.Tensor:
            if x_norm_buf is None:
                return F.layer_norm(x_in, ln.normalized_shape, ln.weight, ln.bias, float(ln.eps))
            vision_module.layernorm_bias_into(
                x=x_in,
                weight=ln.weight,
                bias=ln.bias,
                out=x_norm_buf,
                eps=float(ln.eps),
                fallback_to_torch=True,
            )
            return x_norm_buf

        for i, block in enumerate(module.blocks):
            nvtx.range_push(f"layer{i}")

            nvtx.range_push(f"layer{i}_ln1")
            x_norm = _layer_norm(x, block.ln1)
            nvtx.range_pop()

            nvtx.range_push(f"layer{i}_attn")
            if detailed:
                attn_out = _vision_attn_detail(x_norm, block.attn, config.enc_n_heads, f"layer{i}_attn")
            else:
                attn_out = vision_module._vision_attn(x_norm, block.attn, config.enc_n_heads)
            nvtx.range_pop()

            nvtx.range_push(f"layer{i}_attn_proj")
            b_proj = block.attn["proj"].bias
            if (
                x.is_cuda
                and not torch.is_grad_enabled()
                and x.dtype == torch.bfloat16
                and attn_out.dtype == x.dtype
                and x.is_contiguous()
                and attn_out.is_contiguous()
                and b_proj is not None
            ):
                vision_module.fused_linear_bias_residual_into(
                    x=attn_out,
                    w=block.attn["proj"].weight,
                    b=b_proj,
                    residual=x,
                    out=x,
                )
            else:
                x = x + block.attn["proj"](attn_out)
            nvtx.range_pop()

            nvtx.range_push(f"layer{i}_ln2")
            x_norm = _layer_norm(x, block.ln2)
            nvtx.range_pop()

            nvtx.range_push(f"layer{i}_mlp")
            b1 = block.mlp["fc1"].bias
            b2 = block.mlp["fc2"].bias
            if (
                x.is_cuda
                and not torch.is_grad_enabled()
                and x.dtype == torch.bfloat16
                and x_norm.dtype == x.dtype
                and x.is_contiguous()
                and x_norm.is_contiguous()
                and b1 is not None
                and b2 is not None
            ):
                vision_module.fused_mlp_gelu_bias_residual_into(
                    x=x_norm,
                    w1=block.mlp["fc1"].weight,
                    b1=b1,
                    w2=block.mlp["fc2"].weight,
                    b2=b2,
                    residual=x,
                    out=x,
                )
            else:
                if detailed:
                    x = x + _vision_mlp_detail(x_norm, block.mlp, f"layer{i}_mlp")
                else:
                    x = x + vision_module._vision_mlp(x_norm, block.mlp)
            nvtx.range_pop()

            if early_layer is not None and i == early_layer:
                early = x

            nvtx.range_pop()  # layer{i}

        nvtx.range_push("post_ln")
        x = _layer_norm(x, module.post_ln)
        nvtx.range_pop()
        nvtx.range_pop()  # vision_encoder
        if early_layer is not None:
            return x, early
        return x

    vision_module.vision_encoder = instrumented_vision_encoder


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load images. Install it or use --image-size.") from exc

    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)


def _synthetic_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile vision encoder forward pass")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional config JSON path")
    parser.add_argument("--image", type=Path, default=None, help="Path to an input image")
    parser.add_argument("--image-size", type=int, default=768, help="Synthetic image size (square)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1, help="Profiled iterations")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--encoder-only", action="store_true", help="Skip reconstruction/projection")
    parser.add_argument("--compact", action="store_true", help="Use fewer NVTX ranges")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic image")
    args = parser.parse_args()

    patch_vision_encoder_with_nvtx(detailed=not args.compact)

    from kestrel.config import ModelPaths, RuntimeConfig
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.moondream import vision as vision_module

    device = torch.device("cuda")
    dtype = _parse_dtype(args.dtype)

    runtime_cfg = RuntimeConfig(
        model_paths=ModelPaths(weights=args.weights.expanduser(), config_json=args.config_json),
        device="cuda",
        dtype=dtype,
        max_batch_size=2,
    )

    print(f"Loading model from {args.weights}...")
    runtime = MoondreamRuntime(runtime_cfg)
    vision_cfg = runtime.config.vision
    vision_model = runtime.model.vision
    print(
        "Model loaded. Vision config: "
        f"layers={vision_cfg.enc_n_layers}, dim={vision_cfg.enc_dim}, "
        f"ff_dim={vision_cfg.enc_ff_dim}, heads={vision_cfg.enc_n_heads}"
    )

    if args.image is not None:
        image = _load_image(args.image.expanduser())
        print(f"Loaded image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    else:
        image = _synthetic_image(args.image_size, args.image_size, args.seed)
        print(f"Using synthetic image: {args.image_size}x{args.image_size}")

    overlap = vision_module.compute_overlap_crops(image, vision_cfg)
    num_crops = overlap["crops"].shape[0]
    patches_per_crop = (vision_cfg.crop_size // vision_cfg.enc_patch_size) ** 2
    print(f"Tiling={overlap['tiling']}, num_crops={num_crops}, tokens_per_crop={patches_per_crop}")

    # Prepare crops on GPU (not part of profiled forward)
    crops, tiling = vision_module.prepare_crops_from_overlap(overlap, device, dtype)
    torch.cuda.synchronize()

    def run_forward() -> torch.Tensor:
        outputs = vision_module.vision_encoder(crops, vision_model, vision_cfg)
        if args.encoder_only:
            return outputs

        global_features = outputs[0]
        local = outputs[1:].reshape(
            -1,
            vision_cfg.enc_n_layers,
            vision_cfg.enc_n_layers,
            vision_cfg.enc_dim,
        )

        nvtx.range_push("vision_reconstruct")
        reconstructed = vision_module.reconstruct_from_crops(
            local.to(dtype=torch.float32),
            tiling,
            overlap_margin=vision_cfg.overlap_margin,
            patch_size=1,
        )
        reconstructed = reconstructed.to(device=device, dtype=outputs.dtype)
        nvtx.range_pop()

        nvtx.range_push("vision_projection")
        projected = vision_module.vision_projection(
            global_features,
            reconstructed,
            vision_model,
            vision_cfg,
        )
        nvtx.range_pop()
        return projected

    print(f"Running {args.warmup} warmup iterations...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_forward()
        torch.cuda.synchronize()

        print(f"Running {args.iters} profiled iterations...")
        for _ in range(args.iters):
            nvtx.range_push("vision_forward")
            run_forward()
            nvtx.range_pop()
        torch.cuda.synchronize()

    print("Done. Suggested commands:")
    print("  nsys stats --report nvtx_gpu_proj_sum /tmp/vision_profile.nsys-rep")
    print("  nsys stats --report cuda_gpu_kern_sum /tmp/vision_profile.nsys-rep")
    print("  nsys stats --report nvtx_gpu_proj_sum --filter-nvtx vision_forward /tmp/vision_profile.nsys-rep")


if __name__ == "__main__":
    main()
