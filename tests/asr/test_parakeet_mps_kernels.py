"""The encoder must give the same answer whichever conformer backend the runtime resolves to.

The model calls ``get_runtime().conformer`` and never branches on the device, so the check is: run the encoder
against the backend this host has, then against the torch reference, and compare. On a Mac that exercises the
fused Metal kernels; anywhere else both sides are the reference and the test still guards the wiring.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "kestrel_kernels.conformer_ops", reason="the conformer ops ship with kestrel-kernels"
)

from kestrel_kernels.conformer_ops import torch_conformer_runtime

import kestrel.models.parakeet_tdt.model as parakeet_model
from kestrel.models.parakeet_tdt.config import ParakeetEncoderConfig
from kestrel.models.parakeet_tdt.model import Encoder


def _config(channels: int = 64, layers: int = 2) -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=channels, intermediate_size=2 * channels, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=4, num_mel_bins=128, conv_kernel_size=9,
        subsampling_conv_channels=32, subsampling_conv_kernel_size=3, subsampling_conv_stride=2,
        subsampling_factor=8, max_position_embeddings=5000, hidden_act="silu",
    )


def _encoder(device: torch.device, dtype: torch.dtype) -> Encoder:
    torch.manual_seed(7)
    encoder = Encoder(_config())
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.05)
        for block in encoder.layers:
            block.conv.norm.running_mean.normal_(0.0, 0.1)
            block.conv.norm.running_var.uniform_(0.5, 2.0)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    encoder.reset_nonpersistent_buffers()
    return encoder


def _run(encoder: Encoder, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return encoder.forward_subsampled(hidden, valid)[0].float().clone()


def _use_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        parakeet_model,
        "get_runtime",
        lambda: SimpleNamespace(conformer=torch_conformer_runtime("shifted")),
    )


def test_the_model_never_asks_what_device_it_is_on() -> None:
    with open(parakeet_model.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "device.type" not in text, "the encoder goes through the kernel runtime, not a device check"


@pytest.mark.parametrize(
    "device_name,dtype",
    [
        pytest.param("cpu", torch.float32, id="cpu-fp32"),
        pytest.param("mps", torch.float16, id="mps-fp16"),
    ],
)
def test_encoder_matches_the_torch_reference(monkeypatch, device_name, dtype) -> None:
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS backend not available on this host")
    device = torch.device(device_name)
    encoder = _encoder(device, dtype)
    hidden = torch.randn(1, 53, 64, device=device, dtype=dtype) * 0.5
    valid = torch.ones(1, 53, dtype=torch.bool, device=device)

    backend = _run(encoder, hidden, valid)
    _use_reference(monkeypatch)
    reference = _run(encoder, hidden, valid)

    tolerance = 5e-2 if dtype is torch.float16 else 1e-5
    torch.testing.assert_close(backend, reference, rtol=tolerance, atol=tolerance / 10)


def test_padded_batch_matches_the_torch_reference(monkeypatch) -> None:
    """A padded row softmaxes to NaN on both sides; every valid row must stay finite and equal."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend not available on this host")
    device = torch.device("mps")
    encoder = _encoder(device, torch.float16)
    hidden = torch.randn(2, 40, 64, device=device, dtype=torch.float16) * 0.5
    valid = torch.ones(2, 40, dtype=torch.bool, device=device)
    valid[1, 25:] = False

    backend = _run(encoder, hidden, valid)
    _use_reference(monkeypatch)
    reference = _run(encoder, hidden, valid)

    finite = torch.isfinite(reference)
    torch.testing.assert_close(torch.isfinite(backend).to(torch.uint8), finite.to(torch.uint8))
    torch.testing.assert_close(backend[finite], reference[finite], rtol=5e-2, atol=5e-3)
