from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F
from torch import Tensor

from kestrel.device import make_stream
from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph

from .model import ParakeetTdt


_ENCODER_GRAPH_BUCKETS = (48, 80, 128, 224)
# A captured graph saves the encoder's kernel launches, which only matter while the batch is small; past that the
# bucket padding (1.25x the real frames on LibriSpeech test-clean, 1.95x on AMI, whose median segment is 1.5 s
# against the 3.8 s smallest bucket) and one capture per (batch, bucket) pair cost more than the launches saved.
# Measured on a B200 with parakeet-tdt-0.6b-v3, graphed over eager real-time factor, LibriSpeech test-clean / AMI
# test: batch 1 1.46x / 1.64x, 2 1.40x / 1.56x, 4 1.30x / 1.47x, 8 1.22x / 1.36x, 16 1.18x / 1.15x, 32 0.99x / 1.00x,
# 128 0.70x / 0.74x. Batches above this size run eagerly, padded only to their own longest row.
_ENCODER_GRAPH_MAX_BATCH = 16


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
        max_batch: int,
        device: torch.device,
        stream: torch.cuda.Stream | None,
        buckets: tuple[int, ...] = _ENCODER_GRAPH_BUCKETS,
        graph_max_batch: int = _ENCODER_GRAPH_MAX_BATCH,
    ) -> None:
        self.buckets = _normalize_buckets(buckets)
        if type(graph_max_batch) is not int or graph_max_batch <= 0:
            raise ValueError("encoder graph_max_batch must be a positive integer")
        self.graph_max_batch = graph_max_batch
        self._model = model
        if stream is None:
            stream = make_stream(device)
        self._stream = stream
        # Retain every batch/bucket graph until shutdown: evicting one can free
        # cuBLAS workspace still used by other graphs on the same stream.
        # https://github.com/pytorch/pytorch/issues/193402
        self._graphs = FixedShapeSinglePassGraph(
            enabled=enabled and bool(self.buckets),
            device=device,
            stream=stream,
            run_forward=model.encode_subsampled,
            max_entries=max(1, min(max_batch, graph_max_batch) * len(self.buckets)),
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

    def _replay(self, features: Tensor) -> tuple[int, int] | None:
        """Subsampled length and the bucket to pad it to, or None to run eagerly."""
        if not self.enabled or features.shape[0] > self.graph_max_batch:
            return None
        factor = self._model.config.encoder.subsampling_factor
        length = (features.shape[1] + factor - 1) // factor
        bucket = next((size for size in self.buckets if size >= length), None)
        return None if bucket is None else (length, bucket)

    def _encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        replay = self._replay(features)
        if replay is None:
            return self._model.encode(features, mask)
        length, bucket = replay

        hidden, valid = self._model.encoder.subsampling(features, mask)
        hidden = F.pad(hidden, (0, 0, 0, bucket - length)).contiguous()
        valid = F.pad(valid, (0, bucket - length)).contiguous()
        with self._graphs.launch(hidden, valid) as (encoded, valid):
            return encoded[:, :length], valid[:, :length]

    def launch(
        self, features: Tensor, mask: Tensor
    ) -> AbstractContextManager[tuple[Tensor, ...]]:
        return self._session.launch(features, mask)

    def encode(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode into caller-owned tensors that outlive this call.

        ``launch`` leases session-owned buffers: under replay its outputs are
        views of a captured graph's static outputs, which the next replay of
        that shape overwrites. A caller holding an encoding while the next
        batch is enqueued needs a copy of its own -- and, when it consumes the
        encoding on a different stream than the session's, that stream ordered
        after the session's.
        """
        with self.launch(features, mask) as leased:
            encoded, valid = leased
            if self._replay(features) is not None:
                encoded, valid = encoded.clone(), valid.clone()
        stream = self._stream
        if stream is not None:
            current = torch.cuda.current_stream(stream.device)
            if current != stream:
                current.wait_stream(stream)
        return encoded, valid

    def shutdown(self) -> None:
        self._session.shutdown()
        self._graphs.shutdown()
