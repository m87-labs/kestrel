"""The CPU/MPS depthwise-conv path (shifted multiply-adds in the [B, T, C] layout) must match nn.Conv1d + BatchNorm."""

import pytest
import torch

from kestrel.models.parakeet_tdt.config import ParakeetEncoderConfig
from kestrel.models.parakeet_tdt.model import Convolution


def _config(channels: int = 64, kernel: int = 9) -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=channels, intermediate_size=4 * channels, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, num_mel_bins=128, conv_kernel_size=kernel, subsampling_conv_channels=32,
        subsampling_conv_kernel_size=3, subsampling_conv_stride=2, subsampling_factor=8,
        max_position_embeddings=5000, hidden_act="silu",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_layout_depthwise_matches_conv1d(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    conv = Convolution(_config())
    with torch.no_grad():
        conv.depthwise_conv.weight.normal_()
        conv.norm.weight.uniform_(0.5, 1.5)
        conv.norm.bias.normal_()
        conv.norm.running_mean.normal_()
        conv.norm.running_var.uniform_(0.5, 2.0)
    conv.eval()
    hidden = torch.randn(2, 37, 64)
    reference = torch.nn.functional.silu(conv.norm(conv.depthwise_conv(hidden.transpose(1, 2)))).transpose(1, 2)
    out = torch.nn.functional.silu(conv._depthwise_norm_linear_layout(hidden.to(dtype))).float()
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    assert torch.allclose(out, reference, atol=tol, rtol=tol)


def test_forward_paths_agree_in_eval(monkeypatch) -> None:
    import kestrel.models.parakeet_tdt.model as pm

    torch.manual_seed(1)
    conv = Convolution(_config()).eval()
    with torch.no_grad():
        conv.norm.running_mean.normal_()
        conv.norm.running_var.uniform_(0.5, 2.0)
    hidden = torch.randn(1, 20, 64)
    monkeypatch.setattr(pm, "DEPTHWISE_LINEAR_LAYOUT_DEVICES", frozenset({"cpu"}))
    linear_layout = conv(hidden, None)
    monkeypatch.setattr(pm, "DEPTHWISE_LINEAR_LAYOUT_DEVICES", frozenset())
    reference = conv(hidden, None)
    assert torch.allclose(linear_layout, reference, atol=1e-5, rtol=1e-5)
