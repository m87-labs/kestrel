"""The speech head some Parakeet checkpoints carry on the encoder's subsampler.

A checkpoint that ships ``vad_head.*`` tensors segments its own long audio: the
head is three small convolutions over the convolutional subsampler's output, so
a frame costs a fraction of one encoder layer and the whole recording can be
scanned before a second of it is transcribed. The subsampler is purely local,
which is what makes the scan valid in blocks -- a frame's features are the same
whether the model ran on that block or on the whole file.

Stock checkpoints carry no such tensors and fall back to the energy pauses in
``segment.py``; nothing here is bundled as a default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .features import parakeet_features


VAD_HEAD_PREFIX = "vad_head."
_SPEECH_THRESHOLD = 0.5
_MIN_SPEECH_SECONDS = 0.1
_MIN_GAP_SECONDS = 0.1
_HIDDEN_SIZE = 128
_CONTEXT_KERNEL = 5
# `parakeet_features` refuses to normalize anything shorter; a block that small
# is claimed rather than dropped, so no audio goes missing over it.
_MIN_FEATURE_SAMPLES = 320


class VadHead(nn.Module):
    """Speech logits per subsampled frame, from `[B, T, hidden]` to `[B, T]`."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(hidden_size, _HIDDEN_SIZE, 1)
        self.ctx = nn.Conv1d(
            _HIDDEN_SIZE, _HIDDEN_SIZE, _CONTEXT_KERNEL, padding=_CONTEXT_KERNEL // 2
        )
        self.out = nn.Conv1d(_HIDDEN_SIZE, 1, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        value = F.silu(self.proj(hidden.transpose(1, 2)))
        value = F.silu(self.ctx(value))
        return self.out(value).squeeze(1)


def speech_regions(
    probabilities: np.ndarray, *, frame_seconds: float
) -> list[tuple[float, float]]:
    """`(start, end)` seconds for each run of frames the head calls speech.

    Gaps shorter than `_MIN_GAP_SECONDS` are bridged and runs shorter than
    `_MIN_SPEECH_SECONDS` dropped, the post-processing any voice-activity
    detector applies before its output is used as a boundary.
    """

    speaking = np.asarray(probabilities) >= _SPEECH_THRESHOLD
    edges = np.flatnonzero(np.diff(np.r_[0, speaking.astype(np.int8), 0]))
    regions: list[list[float]] = []
    for start, end in edges.reshape(-1, 2) * frame_seconds:
        if regions and start - regions[-1][1] < _MIN_GAP_SECONDS:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    return [
        (start, end) for start, end in regions if end - start >= _MIN_SPEECH_SECONDS
    ]


@torch.inference_mode()
def head_speech(
    model: object, waveform: np.ndarray, sample_rate: int
) -> list[tuple[float, float]]:
    """Speech regions from the loaded checkpoint's own head.

    Device and dtype come from the subsampler that consumes the features, so
    this follows the model wherever it was loaded without asking.
    """

    duration = waveform.size / sample_rate
    if waveform.size < _MIN_FEATURE_SAMPLES:
        return [(0.0, duration)]
    weight = model.encoder.subsampling.linear.weight
    samples = torch.from_numpy(np.ascontiguousarray(waveform)).to(weight.device)
    features, mask = parakeet_features(samples)
    probabilities, valid = model.speech_probabilities(features.to(weight.dtype), mask)
    frames = probabilities[0][valid[0]].float().cpu().numpy()
    return [
        (start, min(end, duration))
        for start, end in speech_regions(
            frames, frame_seconds=model.encoder_frame_seconds
        )
    ]


__all__ = ["VAD_HEAD_PREFIX", "VadHead", "head_speech", "speech_regions"]
