"""The speech head some Parakeet checkpoints carry on the encoder's subsampler.

A checkpoint that ships ``vad_head.*`` tensors segments its own long audio: the
head is three small convolutions over the convolutional subsampler's output, so
a frame costs a fraction of one encoder layer and the whole recording can be
scanned before a single second of it is transcribed. The subsampler is purely
local, which is what makes the scan valid in blocks -- a frame's features are
the same whether the model ran on that block or on the whole file.

Stock checkpoints carry no such tensors and fall back to the energy detector in
``segment.py``; nothing here is bundled as a default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .features import parakeet_features


VAD_HEAD_PREFIX = "vad_head."
SPEECH_THRESHOLD = 0.5
MIN_SPEECH_SECONDS = 0.1
MIN_GAP_SECONDS = 0.1
_HIDDEN_SIZE = 128
_CONTEXT_KERNEL = 5
# `parakeet_features` refuses to normalize anything shorter; a tail that small
# is claimed rather than dropped, so no audio goes missing over it.
_MIN_FEATURE_SAMPLES = 320


class VadHead(nn.Module):
    """Speech logits per subsampled frame, from ``[B, T, hidden]`` to ``[B, T]``."""

    def __init__(
        self,
        hidden_size: int,
        *,
        width: int = _HIDDEN_SIZE,
        kernel: int = _CONTEXT_KERNEL,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(hidden_size, width, 1)
        self.ctx = nn.Conv1d(width, width, kernel, padding=kernel // 2)
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        value = F.silu(self.proj(hidden.transpose(1, 2)))
        value = F.silu(self.ctx(value))
        return self.out(value).squeeze(1)


def speech_regions(
    probabilities: np.ndarray,
    *,
    frame_seconds: float,
    threshold: float = SPEECH_THRESHOLD,
    min_speech: float = MIN_SPEECH_SECONDS,
    min_gap: float = MIN_GAP_SECONDS,
) -> list[tuple[float, float]]:
    """``(start, end)`` seconds for each run of frames the head calls speech.

    Gaps shorter than ``min_gap`` are bridged and runs shorter than ``min_speech``
    are dropped, the post-processing any voice-activity detector applies before
    its output is used as a boundary.
    """

    speaking = np.asarray(probabilities) >= threshold
    regions: list[list[float]] = []
    index = 0
    while index < speaking.size:
        if not speaking[index]:
            index += 1
            continue
        end = index
        while end < speaking.size and speaking[end]:
            end += 1
        start_seconds, end_seconds = index * frame_seconds, end * frame_seconds
        if regions and start_seconds - regions[-1][1] < min_gap:
            regions[-1][1] = end_seconds
        else:
            regions.append([start_seconds, end_seconds])
        index = end
    return [
        (start, end) for start, end in regions if end - start >= min_speech
    ]


class VadHeadSpeech:
    """Pause source (a) -- the loaded checkpoint's own head.

    Features and the subsampler run over the block handed to ``__call__``; the
    caller keeps those blocks small enough to hold, which is exactly why the
    head sits below the conformer layers rather than above them.
    """

    def __init__(
        self,
        model: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._model = model
        self._device = device
        self._dtype = dtype

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        duration = waveform.size / sample_rate
        if waveform.size < _MIN_FEATURE_SAMPLES:
            return [(0.0, duration)]
        samples = torch.from_numpy(np.ascontiguousarray(waveform)).to(self._device)
        features, mask = parakeet_features(samples)
        probabilities, valid = self._model.speech_probabilities(
            features.to(self._dtype), mask
        )
        frames = probabilities[0][valid[0]].float().cpu().numpy()
        regions = speech_regions(
            frames, frame_seconds=self._model.encoder_frame_seconds
        )
        return [(start, min(end, duration)) for start, end in regions]


__all__ = [
    "MIN_GAP_SECONDS",
    "MIN_SPEECH_SECONDS",
    "SPEECH_THRESHOLD",
    "VAD_HEAD_PREFIX",
    "VadHead",
    "VadHeadSpeech",
    "speech_regions",
]
