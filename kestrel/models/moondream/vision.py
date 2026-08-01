"""Vision encoder components ported from the Moondream reference implementation."""


from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import VisionConfig
from .image_crops import OverlapCropOutput, overlap_crop_image, reconstruct_from_crops
from kestrel.utils.image import ensure_srgb
from kestrel.ops.fused_mlp import fused_mlp_gelu_bias_residual_into
from kestrel_kernels import get_runtime

_KERNELS = get_runtime()
fused_linear_bias_residual_into = _KERNELS.vision.fused_linear_bias_residual_into
_flash_attn_fwd = _KERNELS.attention.flash_attn_fwd
_layernorm_bias_into = _KERNELS.dense.layernorm_bias_into


def prepare_crops(
    image: np.ndarray,
    config: VisionConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    overlap = compute_overlap_crops(image, config)
    return prepare_crops_from_overlap(overlap, device, dtype)


def prepare_crops_from_overlap(
    overlap: OverlapCropOutput,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    crops_cpu = torch.from_numpy(overlap["crops"])
    crops_cpu = crops_cpu.permute(0, 3, 1, 2).contiguous()
    # NOTE: Use async H2D for performance.
    #
    # Important: non_blocking=True enqueues the H2D copy on the *current* CUDA stream.
    # Ensure the copy is enqueued on the same stream as the consumer (e.g. CUDA graph
    # replay), or add an explicit dependency (wait_stream/event). Otherwise, the
    # consumer can observe stale/partially-copied inputs. pin_memory is a
    # CUDA-only optimization; skip on other backends (MPS DispatchStub raises).
    if device.type == "cuda":
        crops_cpu = crops_cpu.pin_memory()
    crops = crops_cpu.to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    crops = crops.div_(255.0)
    crops = crops.sub_(0.5).div_(0.5)
    return crops, overlap["tiling"]


# Hopper `wgmma` operand descriptors require 16-byte alignment on the K-contiguous
# operand; in bf16 that is 8 elements.  The natural SigLIP patch dimension is
# 14*14*3 = 588, which is only 4-element aligned, so NO sm90 tile in cuBLAS's kernel
# set is eligible and the heuristic falls back to an Ampere `s16816` kernel running on
# a Hopper part.  Measured on H100 at nc=13 (M=9477, N=1152): 130.56 us for the
# Ampere kernel against 36.96 us for the sm90 one it binds at K=592 -- 3.53x, or
# +93.60 us per forward, on the single largest lever found on this path.
#
# Padding K to the next multiple of 8 is EXACT, not an approximation: the added
# columns are zero on both operands, and a zero contributes exactly zero to every dot
# product.  592 rather than 608 because a measured sweep put their walls inside each
# other's spread while 592 costs +0.68% extra FLOPs against 608's +3.40%.
PATCH_DIM_ALIGN = 8


def aligned_patch_dim(raw_patch_dim: int) -> int:
    """Round a patch dimension up to the tensor-core alignment (588 -> 592).

    One owner for the rule: :func:`create_patches` and :func:`build_vision_model`
    both derive their width from this, so the activation and the weight cannot
    disagree about how wide the operand is.
    """
    return -(-raw_patch_dim // PATCH_DIM_ALIGN) * PATCH_DIM_ALIGN


def create_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Fold an image batch into patch rows, zero-padded to the aligned width.

    The padding is intrinsic rather than opt-in: this function exists to produce the
    operand `patch_emb` consumes, and an unpadded caller would silently build the
    misaligned GEMM this alignment exists to avoid.  Callers that want the raw width
    should slice ``[..., :channels * patch_size**2]``.
    """
    bsz, channels, height, width = x.shape
    p1 = p2 = patch_size
    gh, gw = height // p1, width // p2
    n = gh * gw
    raw = channels * p1 * p2
    dim = aligned_patch_dim(raw)

    x = x.reshape(bsz, channels, gh, p1, gw, p2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    if dim == raw:
        return x.reshape(bsz, n, raw)

    # The permute makes `x` non-contiguous, so materializing it costs one copy no
    # matter what.  Writing THROUGH a strided view of the padded buffer spends
    # exactly that one copy -- a `reshape(...)` followed by a slice-assign would pay
    # it twice, and this op moves ~11 MiB per forward at nc=13.  Only the 4 pad
    # columns are zeroed, not the whole buffer.
    out = x.new_empty(bsz, n, dim)
    out.as_strided(
        size=(bsz, gh, gw, channels, p1, p2),
        stride=(n * dim, gw * dim, dim, p1 * p2, p2, 1),
    ).copy_(x)
    out[:, :, raw:].zero_()
    return out


def vision_encoder(
    crops: torch.Tensor,
    module: nn.Module,
    config: VisionConfig,
    *,
    early_layer: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x = create_patches(crops, config.enc_patch_size)
    x = module.patch_emb(x)
    x = x + module.pos_emb
    early = None
    # Cross-arch: ``_layernorm_bias_into`` dispatches to the .so kernel
    # on CUDA / the Metal kernel on MPS. The kernel handles the SigLIP
    # vision config directly and raises on anything unsupported.
    x_norm_buf = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    def _layer_norm(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        _layernorm_bias_into(
            x_norm_buf, x, ln.weight, ln.bias, float(ln.eps),
        )
        return x_norm_buf

    for i, block in enumerate(module.blocks):
        x_norm = _layer_norm(x, block.ln1)
        attn_out = _vision_attn(x_norm, block.attn, config.enc_n_heads)
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
            fused_linear_bias_residual_into(
                x=attn_out,
                w=block.attn["proj"].weight,
                b=b_proj,
                residual=x,
                out=x,
            )
        else:
            x = x + block.attn["proj"](attn_out)
        x_norm = _layer_norm(x, block.ln2)
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
            fused_mlp_gelu_bias_residual_into(
                x=x_norm,
                w1=block.mlp["fc1"].weight,
                b1=b1,
                w2=block.mlp["fc2"].weight,
                b2=b2,
                residual=x,
                out=x,
            )
        else:
            x = x + _vision_mlp(x_norm, block.mlp)
        if early_layer is not None and i == early_layer:
            early = x
    x = _layer_norm(x, module.post_ln)
    if early_layer is not None:
        return x, early
    return x


def _vision_attn(
    x: torch.Tensor,
    attn: nn.ModuleDict,
    n_heads: int,
) -> torch.Tensor:
    qkv = attn["qkv"](x)
    dim = x.shape[-1]
    head_dim = dim // n_heads
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(x.size(0), -1, n_heads, head_dim)
    k = k.view(x.size(0), -1, n_heads, head_dim)
    v = v.view(x.size(0), -1, n_heads, head_dim)
    out, _ = _flash_attn_fwd(q, k, v, causal=False)
    return out.reshape(x.size(0), -1, dim)


def _vision_mlp(x: torch.Tensor, mlp: nn.ModuleDict) -> torch.Tensor:
    x = F.gelu(mlp["fc1"](x), approximate="tanh")
    return mlp["fc2"](x)


def vision_projection(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    module: nn.Module,
    config: VisionConfig,
) -> torch.Tensor:
    reconstructed = local_features.permute(2, 0, 1)
    # MPS has no adaptive-pool implementation for non-divisible input/output
    # ratios (pytorch#96056). Detour through CPU — the pooled output is small
    # (enc_n_layers × enc_n_layers × enc_dim) so the round-trip is cheap.
    if reconstructed.device.type == "mps":
        pooled = F.adaptive_avg_pool2d(
            reconstructed.to("cpu"),
            output_size=(config.enc_n_layers, config.enc_n_layers),
        )
        reconstructed = pooled.to(reconstructed.device)
    else:
        reconstructed = F.adaptive_avg_pool2d(
            reconstructed,
            output_size=(config.enc_n_layers, config.enc_n_layers),
        )
    reconstructed = reconstructed.permute(1, 2, 0).reshape(-1, config.enc_dim)
    features = torch.cat([global_features, reconstructed], dim=-1)
    hidden = F.gelu(module.proj_mlp["fc1"](features), approximate="tanh")
    output = module.proj_mlp["fc2"](hidden)
    return output


def build_vision_model(
    config: VisionConfig,
    dtype: torch.dtype,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    # Padded to the tensor-core alignment; `create_patches` pads the activation to
    # the same width from the same rule.  Checkpoints stay canonical at the RAW width
    # and are padded on load (see `weights._copy_patch_emb_weight`), so no re-export
    # is needed and a 588-column checkpoint keeps working unchanged.
    patch_dim = aligned_patch_dim(
        config.enc_patch_size * config.enc_patch_size * config.in_channels
    )
    grid_size = config.crop_size // config.enc_patch_size
    num_patches = grid_size * grid_size

    model = nn.ModuleDict(
        {
            "patch_emb": nn.Linear(patch_dim, config.enc_dim, dtype=dtype, device=device),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype, device=device),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(config.enc_dim, 3 * config.enc_dim, dtype=dtype, device=device),
                                    "proj": nn.Linear(config.enc_dim, config.enc_dim, dtype=dtype, device=device),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype, device=device),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(config.enc_dim, config.enc_ff_dim, dtype=dtype, device=device),
                                    "fc2": nn.Linear(config.enc_ff_dim, config.enc_dim, dtype=dtype, device=device),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype, device=device),
            "proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(config.enc_dim * 2, config.proj_inner_dim, dtype=dtype, device=device),
                    "fc2": nn.Linear(config.proj_inner_dim, config.proj_out_dim, dtype=dtype, device=device),
                }
            ),
        }
    )
    model.pos_emb = nn.Parameter(torch.zeros(1, num_patches, config.enc_dim, dtype=dtype, device=device))

    # Zero the alignment pad on BOTH operands, not just the activation.
    # `nn.Linear(592, ...)` random-inits all 592 columns, so a freshly-built model
    # carries garbage where the pad is. It is numerically harmless *today* only
    # because `create_patches` zeroes the matching activation columns -- a one-sided
    # invariant that holds until someone feeds this Linear a differently-built
    # operand. Zeroing here makes the pad inert from either side, and matches what a
    # checkpoint load produces (`weights._copy_patch_emb_weight` zeroes it too), so a
    # built-then-loaded model and a built-only model agree.
    raw_patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
    if patch_dim != raw_patch_dim:
        with torch.no_grad():
            model["patch_emb"].weight[:, raw_patch_dim:].zero_()
    return model


def encode_image(
    image: Optional[np.ndarray],
    module: nn.Module,
    config: VisionConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    overlap: Optional[OverlapCropOutput] = None,
) -> torch.Tensor:
    with torch.inference_mode():
        if overlap is not None:
            crops, tiling = prepare_crops_from_overlap(overlap, device, dtype)
        else:
            if image is None:
                raise ValueError("image must be provided when overlap is not supplied")
            crops, tiling = prepare_crops(image, config, device, dtype)
        torch._dynamo.mark_dynamic(crops, 0)
        outputs = vision_encoder(crops, module, config)
        global_features = outputs[0]
        local = outputs[1:].reshape(
            -1,
            config.enc_n_layers,
            config.enc_n_layers,
            config.enc_dim,
        )
        reconstructed = reconstruct_from_crops(
            local,
            tiling,
            overlap_margin=config.overlap_margin,
            patch_size=1,
        )
        projected = vision_projection(
            global_features,
            reconstructed,
            module,
            config,
        )
    return projected


def compute_overlap_crops(
    image: np.ndarray, config: VisionConfig
) -> OverlapCropOutput:
    normalized = ensure_srgb(image)
    return overlap_crop_image(
        normalized,
        overlap_margin=config.overlap_margin,
        max_crops=config.max_crops,
        base_size=(config.crop_size, config.crop_size),
        patch_size=config.enc_patch_size,
    )


__all__ = [
    "prepare_crops",
    "prepare_crops_from_overlap",
    "create_patches",
    "aligned_patch_dim",
    "vision_encoder",
    "vision_projection",
    "build_vision_model",
    "encode_image",
    "compute_overlap_crops",
]
