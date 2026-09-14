from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F
from torch import Tensor

from kestrel.device import make_stream
from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph

from .model import ParakeetTdt


_ENCODER_GRAPH_BUCKETS = (48, 80, 128, 224)


def _normalize_buckets(buckets: tuple[int, ...]) -> tuple[int, ...]:
    message = "encoder graph buckets must be strictly increasing positive integers"
    try:
        normalized = tuple(buckets)
    except TypeError as exc:
        raise ValueError(message) from exc
    if any(type(size) is not int or size <= 0 for size in normalized) or any(
        left >= right for left, right in zip(normalized, normalized[1:])
    ):
        raise ValueError(message)
    return normalized


class ParakeetEncoderGraph:
    def __init__(
        self,
        model: ParakeetTdt,
        *,
        enabled: bool,
        device: torch.device,
        stream: torch.cuda.Stream | None,
        buckets: tuple[int, ...] = _ENCODER_GRAPH_BUCKETS,
    ) -> None:
        self.buckets = _normalize_buckets(buckets)
        self._model = model
        if stream is None:
            stream = make_stream(device)
        self._graphs = FixedShapeSinglePassGraph(
            enabled=enabled and bool(self.buckets),
            device=device,
            stream=stream,
            run_forward=model.encode_subsampled,
            max_entries=max(1, len(self.buckets)),
        )
        # The outer session keeps stream ordering and the output lease through
        # the caller's decoding; the inner cache reuses its output buffers.
        self._session = FixedShapeSinglePassGraph(
            enabled=False,
            device=device,
            stream=stream,
            run_forward=self._encode,
        )

    @property
    def enabled(self) -> bool:
        return self._graphs.enabled

    def _encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        factor = self._model.config.encoder.subsampling_factor
        length = (features.shape[1] + factor - 1) // factor
        bucket = next((size for size in self.buckets if size >= length), None)
        if not self.enabled or bucket is None:
            return self._model.encode(features, mask)

        hidden, valid = self._model.encoder.subsampling(features, mask)
        hidden = F.pad(hidden, (0, 0, 0, bucket - length)).contiguous()
        valid = F.pad(valid, (0, bucket - length)).contiguous()
        with self._graphs.launch(hidden, valid) as (encoded, valid):
            return encoded[:, :length], valid[:, :length]

    def launch(
        self, features: Tensor, mask: Tensor
    ) -> AbstractContextManager[tuple[Tensor, ...]]:
        return self._session.launch(features, mask)

    def shutdown(self) -> None:
        self._session.shutdown()
        self._graphs.shutdown()
