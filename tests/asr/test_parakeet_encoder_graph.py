from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from kestrel.models.parakeet_tdt.config import ParakeetEncoderConfig, ParakeetTdtConfig
from kestrel.models.parakeet_tdt.encoder_graph import ParakeetEncoderGraph
from kestrel.models.parakeet_tdt.model import ParakeetTdt


def _model(device="cpu", dtype=torch.float32) -> ParakeetTdt:
    torch.manual_seed(0)
    config = ParakeetTdtConfig(
        encoder=ParakeetEncoderConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_mel_bins=128,
            conv_kernel_size=31,
            subsampling_conv_channels=4,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            subsampling_factor=8,
            max_position_embeddings=5000,
            hidden_act="silu",
        ),
        blank_token_id=8,
        decoder_hidden_size=16,
        durations=(0, 1, 2),
        max_symbols_per_step=10,
        num_decoder_layers=1,
        pad_token_id=8,
        vocab_size=9,
        hidden_act="relu",
    )
    return ParakeetTdt(config).to(device=device, dtype=dtype).eval()


def _features(valid_frames, *, batch=1, device="cpu", dtype=torch.float32):
    features = torch.randn(batch, valid_frames + 1, 128, device=device, dtype=dtype)
    mask = torch.arange(valid_frames + 1, device=device)[None] < valid_frames
    mask = mask.expand(batch, -1)
    return features.masked_fill(~mask[..., None], 0), mask


@pytest.mark.parametrize("buckets", [(0,), (32, 32), (64, 32)])
def test_encoder_rejects_invalid_buckets(buckets) -> None:
    with pytest.raises(ValueError, match="encoder graph buckets"):
        ParakeetEncoderGraph(
            _model(), enabled=False, device=torch.device("cpu"), stream=None,
            buckets=buckets,
        )


@torch.inference_mode()
def test_padding_after_subsampling_preserves_boundary_outputs() -> None:
    model = _model()
    for valid_frames in range(128, 144):
        features, mask = _features(valid_frames)
        expected, expected_valid = model.encode(features, mask)
        hidden, valid = model.encoder.subsampling(features, mask)
        padding = 48 - hidden.shape[1]
        actual, actual_valid = model.encode_subsampled(
            F.pad(hidden, (0, 0, 0, padding)), F.pad(valid, (0, padding))
        )
        assert torch.equal(expected_valid.sum(-1), actual_valid.sum(-1))
        torch.testing.assert_close(actual[actual_valid], expected[expected_valid])


@torch.inference_mode()
@pytest.mark.parametrize("enabled, buckets", [(False, (48, 80)), (True, ())])
def test_disabled_encoder_keeps_original_shapes(enabled, buckets) -> None:
    model = _model()
    session = ParakeetEncoderGraph(
        model, enabled=enabled, device=torch.device("cpu"), stream=None, buckets=buckets
    )
    features, mask = _features(129)
    with session.launch(features, mask) as (actual, valid):
        expected, expected_valid = model.encode(features, mask)
        torch.testing.assert_close(actual, expected, equal_nan=True)
        assert torch.equal(valid, expected_valid)
        with pytest.raises(RuntimeError, match="during an active lease"):
            session.shutdown()
    assert not session._graphs._entries
    session.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        with session.launch(features, mask):
            pass


@torch.inference_mode()
@pytest.mark.parametrize(
    "buckets, expected_calls",
    [(None, [48, 48, 48, 48, 80, 128, 224, 48]), ((24, 48), [24, 24, 48, 48, 24])],
)
def test_bucket_routing_preserves_lengths_and_bypasses_outliers(
    monkeypatch, buckets, expected_calls,
) -> None:
    model = _model()
    session = ParakeetEncoderGraph(
        model,
        enabled=False,
        device=torch.device("cpu"),
        stream=None,
        **({} if buckets is None else {"buckets": buckets}),
    )
    calls = []

    @contextmanager
    def replay(hidden, valid):
        calls.append(hidden.shape[1])
        yield model.encode_subsampled(hidden, valid)

    monkeypatch.setattr(session._graphs, "enabled", True)
    monkeypatch.setattr(session._graphs, "launch", replay)
    try:
        shapes = [(1, 129), (1, 191), (1, 192), (1, 383), (1, 384),
                  (1, 640), (1, 1024), (1, 1792), (8, 129)]
        for batch, length in shapes:
            features, mask = _features(length, batch=batch)
            with session.launch(features, mask) as (actual, valid):
                expected, expected_valid = model.encode(features, mask)
                assert actual.shape == expected.shape
                assert torch.equal(valid, expected_valid)
                torch.testing.assert_close(actual[valid], expected[valid])
        assert calls == expected_calls
    finally:
        session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("buckets", [None, (32, 64)])
@torch.inference_mode()
def test_cuda_buckets_replay_without_capturing_outliers(buckets) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    model = _model(device, torch.bfloat16)
    calls = []
    original = model.encode_subsampled

    def counted(hidden, valid):
        calls.append(tuple(hidden.shape))
        return original(hidden, valid)

    model.encode_subsampled = counted
    session = ParakeetEncoderGraph(
        model, enabled=True, device=device, stream=stream,
        **({} if buckets is None else {"buckets": buckets}),
    )
    buckets = session.buckets
    try:
        lengths = [size * 8 - 1 for size in buckets]
        lengths += [129, 137, 383, 384, 639, 640, 1023, 1024, 1791, 1792, 2000]
        for batch in (1, 8, 1):
            for length in lengths:
                features, mask = _features(
                    length, batch=batch, device=device, dtype=torch.bfloat16
                )
                with session.launch(features, mask) as (actual, valid):
                    assert torch.cuda.current_stream(device) == stream
                    expected, expected_valid = model.encode(features, mask)
                    assert actual.shape == expected.shape
                    assert torch.equal(valid, expected_valid)
                    torch.testing.assert_close(
                        actual[valid], expected[valid], atol=0.02, rtol=0.02
                    )
                assert len(session._graphs._entries) <= len(buckets)
        assert len(session._graphs._entries) == len(buckets)
        assert len(calls) == 6 * len(buckets)
        assert {shape[1] for shape in calls} == set(buckets)
    finally:
        session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("provide_stream", [False, True])
@torch.inference_mode()
def test_cuda_bucket_waits_for_producer_and_leases_outputs_to_consumers(
    provide_stream,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    producer = torch.cuda.Stream(device=device)
    model = _model(device)
    session = ParakeetEncoderGraph(
        model,
        enabled=True,
        device=device,
        stream=stream if provide_stream else None,
        buckets=(48, 80),
    )
    stream = session._session._stream
    features, mask = _features(129, device=device)
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(producer):
            torch.cuda._sleep(5_000_000)
            features.mul_(2)
            with session.launch(features, mask) as (actual, valid):
                assert torch.cuda.current_stream(device) == stream
                consumed = actual[valid] + 1
                expected, expected_valid = model.encode(features, mask)
                torch.testing.assert_close(consumed, expected[expected_valid] + 1)
                with pytest.raises(RuntimeError, match="cannot be nested"):
                    with session.launch(features, mask):
                        pass
    finally:
        session.shutdown()
