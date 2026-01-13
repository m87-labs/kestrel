"""Tests for bf16 LayerNorm CUDA kernel."""

import pytest
import torch
import torch.nn.functional as F

from kestrel_kernels.layernorm_cuda import layernorm_bias_cuda, layernorm_bias_reload_cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


DTYPE = torch.bfloat16
EPS = 1e-5
RTOL = 1e-2
ATOL = 1e-2


def layernorm_bias_into(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
    variant: str = "auto",
    fallback_to_torch: bool = False,
) -> None:
    """Compute LayerNorm forward: out = (x - mean) / sqrt(var + eps) * weight + bias."""
    if x.ndim == 3:
        b, t, n = x.shape
        x2 = x.reshape(b * t, n)
        out2 = out.reshape(b * t, n)
    else:
        x2 = x
        out2 = out

    try:
        if variant == "auto":
            variant = "reload_x" if int(x2.shape[1]) == 1152 else "default"
        if variant == "default":
            layernorm_bias_cuda(out2, x2, weight, bias, float(eps))
        elif variant == "reload_x":
            layernorm_bias_reload_cuda(out2, x2, weight, bias, float(eps))
        else:
            raise ValueError(f"Unknown layernorm_cuda variant: {variant!r}")
    except Exception:
        if not fallback_to_torch:
            raise
        out2.copy_(F.layer_norm(x2, (x2.shape[1],), weight, bias, float(eps)))


@pytest.mark.parametrize("crops", list(range(1, 13)))
@pytest.mark.parametrize("variant", ["default", "reload_x"])
def test_layernorm_cuda_supports_vision_crop_sizes(device, crops: int, variant: str):
    torch.manual_seed(0)

    m = crops * 729
    n = 1152
    x = torch.randn((m, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)

    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, variant=variant)

    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m", [1, 32, 128, 1024])
@pytest.mark.parametrize("variant", ["default", "reload_x"])
def test_layernorm_cuda_supports_dim_2048(device, m: int, variant: str):
    torch.manual_seed(0)

    n = 2048
    x = torch.randn((m, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)

    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, variant=variant)

    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("variant", ["default", "reload_x"])
def test_layernorm_cuda_supports_3d_input(device, variant: str):
    torch.manual_seed(0)

    bsz, t, n = 2, 5, 2048
    x = torch.randn((bsz, t, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)

    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, variant=variant)

    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_layernorm_cuda_rejects_non_bf16(device):
    n = 1152
    x = torch.randn((2, n), device=device, dtype=torch.float16)
    w = torch.randn((n,), device=device, dtype=torch.float16)
    b = torch.randn((n,), device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    with pytest.raises(RuntimeError, match="must be bf16"):
        layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS)


def test_layernorm_cuda_fallback_to_torch(device):
    torch.manual_seed(0)

    n = 10  # not a multiple of 8
    x = torch.randn((4, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)

    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, fallback_to_torch=True)
    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_layernorm_cuda_auto_variant(device):
    torch.manual_seed(0)

    # N=1152 (vision) should use the reload_x path by default.
    m, n = 729, 1152
    x = torch.randn((m, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)
    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, variant="auto")
    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    # N=2048 should default to the register-caching variant.
    m, n = 730, 2048
    x = torch.randn((m, n), device=device, dtype=DTYPE)
    w = torch.randn((n,), device=device, dtype=DTYPE)
    b = torch.randn((n,), device=device, dtype=DTYPE)
    out = torch.empty_like(x)
    layernorm_bias_into(x=x, weight=w, bias=b, out=out, eps=EPS, variant="auto")
    expected = F.layer_norm(x, (n,), w, b, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
