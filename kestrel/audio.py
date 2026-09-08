"""Small, model-independent PCM helpers."""

from __future__ import annotations

import numpy as np


class SpeechOnsetTrimmer:
    """Buffer PCM until three consecutive 10 ms blocks contain audible speech."""

    def __init__(self, sample_rate: int) -> None:
        self._frame = sample_rate * 10 // 1_000
        self._hop = self._frame
        self._threshold = 10.0 ** (-45.0 / 20.0)
        self._pending = np.empty(0, dtype=np.float32)
        self._scan = 0
        self._run_start: int | None = None
        self._run_length = 0
        self._found = False

    def push(self, pcm: np.ndarray) -> np.ndarray:
        """Return newly audible PCM, or an empty array while onset is unknown."""

        if not isinstance(pcm, np.ndarray) or pcm.dtype != np.float32:
            raise TypeError("PCM must be a float32 numpy array")
        pcm = np.ascontiguousarray(pcm).reshape(-1)
        if not pcm.size or self._found:
            return pcm
        self._pending = np.concatenate((self._pending, pcm))
        while self._scan + self._frame <= self._pending.size:
            window = self._pending[self._scan : self._scan + self._frame]
            centered = window - np.mean(window)
            rms = float(np.sqrt(np.mean(centered * centered)))
            if rms >= self._threshold:
                if self._run_length == 0:
                    self._run_start = self._scan
                self._run_length += 1
                # A click crossing a block boundary can activate two blocks;
                # three requires 30 ms of sustained audio.
                if self._run_length == 3:
                    assert self._run_start is not None
                    audible = self._pending[self._run_start :]
                    self._pending = np.empty(0, dtype=np.float32)
                    self._found = True
                    return audible
            else:
                self._run_start = None
                self._run_length = 0
            self._scan += self._hop
        return np.empty(0, dtype=np.float32)

    def finish(self) -> np.ndarray:
        """Return buffered PCM unchanged when no sustained onset was found."""

        if self._found or not self._pending.size:
            return np.empty(0, dtype=np.float32)
        pending = self._pending
        self._pending = np.empty(0, dtype=np.float32)
        self._found = True
        return pending


__all__ = ["SpeechOnsetTrimmer"]
