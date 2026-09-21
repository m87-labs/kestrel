"""ParakeetTdtRuntime.launch/collect: what stops after the encoder, and what
finishes in place.

A cohort of one-segment requests is the case worth pipelining: ``launch``
leaves with the encoding on the device and ``collect`` decodes it, so the next
cohort's host work runs while this one's encoder does. Anything whose segments
have to be decoded before the next can be cut -- several segments, a live
stream window -- is finished inside ``launch`` and ``collect`` only hands the
results back.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models.asr.contract import Word
from kestrel.models.parakeet_tdt.model import TdtOutput
from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime


class _Tokenizer:
    blank_token_id = 0

    def decode(self, token_ids):
        return f"part {token_ids[1]}."

    def words(self, token_ids, durations, frame_seconds):
        del durations, frame_seconds
        return (Word(f"part{token_ids[1]}", 0.0, 1.0),)


class _Decoder:
    """Stands in for the batched decoder: one distinct token per row."""

    minimum_batch = 2

    def __init__(self) -> None:
        self.batches: list[int] = []
        self._next = 0

    def generate(self, encoded, valid, *, max_tokens):
        del valid, max_tokens
        batch = encoded.shape[0]
        self.batches.append(batch)
        ids = []
        for _ in range(batch):
            self._next += 1
            ids.append(self._next)
        return TdtOutput(
            sequences=torch.tensor([[0, value] for value in ids]),
            durations=torch.tensor([[0, 1] for _ in ids]),
            lengths=torch.tensor([2 for _ in ids]),
            encoder_frame_seconds=0.08,
        )


def _runtime() -> ParakeetTdtRuntime:
    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.compute_stream = None
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8)),
        vad_head=None,
    )
    runtime.tokenizer = _Tokenizer()
    decoder = _Decoder()
    runtime._batch_decoder = decoder

    def encode(features, mask):
        return features[:, :, :2].contiguous(), mask

    @contextmanager
    def launch(features, mask):
        yield encode(features, mask)

    runtime.model.generate = lambda features, mask, *, max_tokens: decoder.generate(
        encode(features, mask)[0], mask, max_tokens=max_tokens
    )
    runtime._encoder_graph = SimpleNamespace(encode=encode, launch=launch)
    return runtime


def _speech(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rng.standard_normal(int(seconds * 16_000)) * 0.1, -1, 1).astype(
        np.float32
    )


def _clip(seconds: float) -> dict[str, object]:
    return {"audio": _speech(seconds), "sample_rate": 16_000, "timestamps": "none"}


def test_one_segment_cohort_stops_after_the_encoder() -> None:
    runtime = _runtime()

    batch = runtime.launch("transcribe", (_clip(4.0), _clip(5.0)))

    assert batch.encoded is not None  # the decode is still owed
    assert runtime._batch_decoder.batches == []
    assert len(batch.rows) == 2

    results = runtime.collect(batch)

    assert runtime._batch_decoder.batches == [2]
    assert [result["text"] for result in results] == ["part 1.", "part 2."]
    assert batch.encoded is None  # the device tensors are released on collect


def test_multi_segment_cohort_finishes_inside_launch() -> None:
    """Segment two cannot be cut until segment one is decoded, so a long
    request is not deferred -- but it still produces the same transcript."""
    runtime = _runtime()
    parts = []
    for _ in range(4):
        parts.append(_speech(9.0))
        parts.append(np.zeros(16_000, dtype=np.float32))
    long_audio = np.concatenate(parts)

    batch = runtime.launch(
        "transcribe",
        ({"audio": long_audio, "sample_rate": 16_000, "timestamps": "none"}, _clip(4.0)),
    )

    assert batch.encoded is None
    assert runtime._batch_decoder.batches  # decoded in place
    assert runtime.collect(batch)[1]["text"] == "part 2."


def test_forward_is_launch_then_collect() -> None:
    runtime = _runtime()

    results = runtime.forward("transcribe", (_clip(4.0), _clip(5.0)))

    assert [result["text"] for result in results] == ["part 1.", "part 2."]


def test_a_batch_the_decoder_will_not_take_is_not_deferred() -> None:
    """Below the decoder's minimum batch the encoder and decode stay joined."""
    runtime = _runtime()

    batch = runtime.launch("transcribe", (_clip(4.0),))

    assert batch.encoded is None
    assert runtime.collect(batch)[0]["text"] == "part 1."


def test_a_bad_input_does_not_stop_the_cohort_from_deferring() -> None:
    runtime = _runtime()

    batch = runtime.launch(
        "transcribe", (_clip(4.0), "not a mapping", _clip(5.0))
    )

    assert batch.encoded is not None
    results = runtime.collect(batch)
    assert isinstance(results[1], ValueError)
    assert [results[0]["text"], results[2]["text"]] == ["part 1.", "part 2."]


def test_rejects_a_task_it_does_not_serve() -> None:
    with pytest.raises(ValueError, match="transcribe"):
        _runtime().launch("segment", ({},))
