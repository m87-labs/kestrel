from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime
from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disabled_encoder_graph_keeps_generated_decode_on_configured_stream() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    compute_stream = torch.cuda.Stream(device=device)
    producer_stream = torch.cuda.Stream(device=device)
    streams = []

    class _Model:
        def encode(
            self, features: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            streams.append(("encode", torch.cuda.current_stream(device)))
            return features + 1, mask

    class _Decoder:
        minimum_batch = 1

        def generate(
            self,
            encoded: torch.Tensor,
            valid: torch.Tensor,
            *,
            max_tokens: int,
        ) -> SimpleNamespace:
            del max_tokens
            with torch.cuda.stream(compute_stream):
                streams.append(("decode", torch.cuda.current_stream(device)))
                torch.testing.assert_close(encoded, torch.full_like(encoded, 4))
                batch = encoded.shape[0]
                return SimpleNamespace(
                    lengths=torch.ones(batch, dtype=torch.long, device=device),
                    sequences=torch.zeros(
                        (batch, 2), dtype=torch.long, device=device
                    ),
                    durations=torch.zeros(
                        (batch, 2), dtype=torch.long, device=device
                    ),
                    encoder_frame_seconds=0.08,
                )

    class _Tokenizer:
        def decode(self, _token_ids: list[int]) -> str:
            return "ok"

    model = _Model()
    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = device
    runtime.dtype = torch.bfloat16
    runtime.model = model
    runtime.tokenizer = _Tokenizer()
    runtime._batch_decoder = _Decoder()
    runtime._encoder_graph = FixedShapeSinglePassGraph(
        enabled=False,
        device=device,
        stream=compute_stream,
        run_forward=model.encode,
    )

    features = torch.zeros((1, 4), dtype=torch.bfloat16, device=device)
    mask = torch.ones((1, 4), dtype=torch.bool, device=device)
    runtime._batch_audio_features = lambda _rows: (features, mask)
    waveform = np.ones(320, dtype=np.float32)
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(5_000_000)
        features.fill_(3)
        result = runtime.forward(
            "transcribe",
            (
                {
                    "audio": waveform,
                    "sample_rate": 16_000,
                    "timestamps": "none",
                },
            ),
        )

    assert result[0]["text"] == "ok"
    assert streams == [("encode", compute_stream), ("decode", compute_stream)]
    runtime.shutdown()
