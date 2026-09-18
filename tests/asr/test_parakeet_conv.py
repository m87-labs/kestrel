"""The conformer convolution module: a checkpoint's Conv1d weights load into it, and it computes exactly what
``nn.Conv1d`` + eval-mode ``BatchNorm1d`` + SiLU do — with either depthwise implementation behind the op."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip(
    "kestrel_kernels.conformer_ops", reason="the conformer ops ship with kestrel-kernels"
)

from kestrel_kernels.conformer_ops import DEPTHWISE_IMPLS, torch_conformer_runtime

import kestrel.models.parakeet_tdt.model as parakeet_model
from kestrel.models.parakeet_tdt.config import ParakeetEncoderConfig
from kestrel.models.parakeet_tdt.model import Convolution

CHANNELS = 64


def _config(kernel: int = 9) -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=CHANNELS, intermediate_size=4 * CHANNELS, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, num_mel_bins=128, conv_kernel_size=kernel, subsampling_conv_channels=32,
        subsampling_conv_kernel_size=3, subsampling_conv_stride=2, subsampling_factor=8,
        max_position_embeddings=5000, hidden_act="silu",
    )


@pytest.fixture(params=sorted(DEPTHWISE_IMPLS))
def conv(request, monkeypatch) -> Convolution:
    """A loaded ``Convolution``, once per depthwise implementation the kernels publish."""
    monkeypatch.setattr(
        parakeet_model,
        "get_runtime",
        lambda: SimpleNamespace(conformer=torch_conformer_runtime(request.param)),
    )
    torch.manual_seed(0)
    module = Convolution(_config()).eval()
    with torch.no_grad():
        module.depthwise_conv.weight.normal_()
        module.norm.weight.uniform_(0.5, 1.5)
        module.norm.bias.normal_()
        module.norm.running_mean.normal_()
        module.norm.running_var.uniform_(0.5, 2.0)
    return module


@pytest.mark.parametrize("masked", [False, True])
def test_matches_conv1d_and_batchnorm(conv: Convolution, masked: bool) -> None:
    hidden = torch.randn(2, 37, CHANNELS)
    valid = torch.ones(2, 37, dtype=torch.bool)
    if masked:
        valid[1, 20:] = False
    gated = F.glu(F.linear(hidden, conv.pointwise_conv1.weight), dim=-1)
    gated = gated.masked_fill(~valid[..., None], 0)
    expected = F.silu(conv.norm(conv.depthwise_conv(gated.transpose(1, 2)))).transpose(1, 2)
    expected = F.linear(expected, conv.pointwise_conv2.weight)
    torch.testing.assert_close(conv(hidden, valid), expected, atol=1e-5, rtol=1e-5)


def test_checkpoint_1x1_conv_weights_load_into_the_linears() -> None:
    module = Convolution(_config())
    state = module.state_dict()
    state["pointwise_conv1.weight"] = torch.randn(2 * CHANNELS, CHANNELS, 1)
    state["pointwise_conv2.weight"] = torch.randn(CHANNELS, CHANNELS, 1)
    expected = state["pointwise_conv2.weight"][..., 0].clone()
    module.load_state_dict(state, strict=True)
    assert module.pointwise_conv1.weight.shape == (2 * CHANNELS, CHANNELS)
    torch.testing.assert_close(module.pointwise_conv2.weight, expected)
