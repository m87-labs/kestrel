"""Request ownership across native codec graph buckets and eager tails."""

from dataclasses import replace
from threading import RLock

import pytest
import torch

from kestrel.models.qwen3_tts.codec import (
    IncrementalCodecState,
    Qwen3TTSCodecDecoder,
    Qwen3TTSIncrementalDecoder,
)
from kestrel.models.qwen3_tts.codec_replay import TorchCodec, _state_tensors
from kestrel.models.qwen3_tts.config import Qwen3TTSCodecConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_replay_preserves_cold_mixed_reordered_and_partial_requests():
    config = replace(
        Qwen3TTSCodecConfig(),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        latent_dim=8,
        codebook_dim=8,
        codebook_size=16,
        decoder_dim=16,
        num_quantizers=2,
        sliding_window=4,
        upsampling_ratios=(2,),
        upsample_rates=(2,),
    )
    torch.manual_seed(3)
    model = Qwen3TTSCodecDecoder(config).cuda().bfloat16().eval()
    model.prepare_for_inference()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        replay = TorchCodec(model, stream, RLock(), 5)
        eager = Qwen3TTSIncrementalDecoder(model)
        actual = [IncrementalCodecState() for _ in range(5)]
        expected = [IncrementalCodecState() for _ in range(5)]
        for rows, frames in [
            ((0, 1, 2), 3),
            ((2, 3, 0, 1, 4), 12),
            ((1, 2), 7),
            ((4, 0, 2), 3),
            ((0,), 1),
        ]:
            codes = torch.randint(
                config.codebook_size,
                (len(rows), config.num_quantizers, frames),
                device="cuda",
            )
            untouched = [
                (row, group, key, value.clone())
                for row, group, key, value in _state_tensors(actual)
                if row not in rows
            ]
            want = eager(codes, [expected[i] for i in rows]).squeeze(1)
            got = replay.run(codes, [actual[i] for i in rows])
            # Use dtype-aware numerical tolerances, not a waveform quality gate.
            torch.testing.assert_close(got, want)
            for row, group, key, saved in untouched:
                torch.testing.assert_close(
                    getattr(actual[row], group)[key], saved, rtol=0, atol=0
                )
            for left, right in zip(actual, expected):
                assert left.frame_position == right.frame_position
                assert (
                    left.transformer_context_length == right.transformer_context_length
                )
            for left, right in zip(
                _state_tensors(actual), _state_tensors(expected), strict=True
            ):
                assert left[:3] == right[:3]
                torch.testing.assert_close(left[3], right[3])
        stream.synchronize()
