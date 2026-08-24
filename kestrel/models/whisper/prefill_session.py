"""Native Whisper prefill graph owned by the public model runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from kestrel_kernels.cubin_runtime import capture_packed_artifact_receipts

from .prefill_decoder_prefix import (
    PreparedWhisperDecoderWeights,
    WhisperDecoderPrefixWorkspace,
    prepare_whisper_decoder_weights,
    whisper_decoder_prefix,
)
from .prefill_encoder import (
    PreparedWhisperEncoderWeights,
    WhisperEncoderWorkspace,
    prepare_whisper_encoder_weights,
    whisper_cross_kv,
    whisper_encoder,
)
from .runtime_abi import (
    WhisperExecutionBindings,
    WhisperCrossArenas,
    WhisperPrefillBuffers,
    WhisperPrefillSession,
    WhisperSelfKVArenas,
)
from .weights import WhisperModelWeights


@dataclass(frozen=True, slots=True)
class _PreparedPrefillWeights:
    encoder: PreparedWhisperEncoderWeights
    decoder: PreparedWhisperDecoderWeights


class NativeWhisperPrefillSession(WhisperPrefillSession):
    """Direct custom-kernel orchestration over fixed resident slot buffers."""

    def __init__(
        self,
        bindings: WhisperExecutionBindings,
        weights: WhisperModelWeights,
        *,
        require_packed: bool,
    ) -> None:
        self._device = bindings.cross_kv.keys.device
        self._stream = bindings.compute_stream
        self._max_batch_size = int(bindings.max_batch_size)
        self._buffers = {slot.slot_id: slot for slot in bindings.prefill_buffers}
        self._weights: _PreparedPrefillWeights | None = _PreparedPrefillWeights(
            encoder=prepare_whisper_encoder_weights(weights),
            decoder=prepare_whisper_decoder_weights(weights),
        )
        self._cross_kv: WhisperCrossArenas | None = bindings.cross_kv
        self._self_kv: WhisperSelfKVArenas | None = bindings.self_kv
        self._workspaces: dict[
            int, tuple[WhisperEncoderWorkspace, WhisperDecoderPrefixWorkspace]
        ] = {}
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._require_packed = bool(require_packed)
        self._artifact_receipts: tuple[dict[str, object], ...] = ()
        self._lock = threading.Lock()
        self._warmed = False
        self._closed = False

    def _workspace(
        self, batch_size: int
    ) -> tuple[WhisperEncoderWorkspace, WhisperDecoderPrefixWorkspace]:
        workspace = self._workspaces.get(batch_size)
        if workspace is None:
            workspace = (
                WhisperEncoderWorkspace.allocate(batch_size, device=self._device),
                WhisperDecoderPrefixWorkspace.allocate(batch_size, device=self._device),
            )
            self._workspaces[batch_size] = workspace
        return workspace

    def _run(self, slot: WhisperPrefillBuffers, batch_size: int) -> None:
        weights = self._weights
        cross_kv = self._cross_kv
        self_kv = self._self_kv
        if weights is None or cross_kv is None or self_kv is None:
            raise RuntimeError("Whisper prefill session is shut down")
        encoder_workspace, decoder_workspace = self._workspace(batch_size)
        encoder_hidden = whisper_encoder(
            slot.input_features[:batch_size],
            weights.encoder,
            encoder_workspace,
            require_packed=self._require_packed,
        )
        whisper_cross_kv(
            encoder_hidden,
            weights.encoder,
            encoder_workspace,
            cross_kv,
            slot.batch_idx[:batch_size],
        )
        whisper_decoder_prefix(
            slot.control_token_ids[:batch_size],
            slot.prefix_lengths[:batch_size],
            slot.slot_mapping[:batch_size],
            encoder_workspace.compact_cross_keys,
            encoder_workspace.compact_cross_values,
            weights.decoder,
            decoder_workspace,
            self_kv,
            logits_out=slot.logits_out[:batch_size],
            require_packed=self._require_packed,
        )

    def _stage_warmup(self, slot: WhisperPrefillBuffers, batch_size: int) -> None:
        slot.input_features[:batch_size].zero_()
        slot.control_token_ids[:batch_size].fill_(50258)
        slot.prefix_lengths[:batch_size].fill_(1)
        slot.slot_mapping[:batch_size].zero_()
        slot.batch_idx[:batch_size].copy_(
            torch.arange(
                1,
                batch_size + 1,
                device=self._device,
                dtype=torch.int64,
            )
        )

    @property
    def artifact_receipts(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(receipt) for receipt in self._artifact_receipts)

    @torch.inference_mode()
    def warmup(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Whisper prefill session is shut down")
            if self._warmed:
                return
            with capture_packed_artifact_receipts() as receipt_capture:
                self._graphs.clear()
                warmup_slot = self._buffers[0]
                with torch.cuda.stream(self._stream):
                    for batch_size in range(1, self._max_batch_size + 1):
                        self._stage_warmup(warmup_slot, batch_size)
                        self._run(warmup_slot, batch_size)
                self._stream.synchronize()

                for batch_size in range(1, self._max_batch_size + 1):
                    for slot_id, slot in self._buffers.items():
                        with torch.cuda.stream(self._stream):
                            self._stage_warmup(slot, batch_size)
                        self._stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=self._stream):
                            self._run(slot, batch_size)
                        self._graphs[(slot_id, batch_size)] = graph
                self._stream.synchronize()
            receipts = receipt_capture.receipts
            if self._require_packed and not receipts:
                raise RuntimeError(
                    "Whisper prefill warmup resolved no packed artifact receipts"
                )
            self._artifact_receipts = receipts
            self._warmed = True

    @torch.inference_mode()
    def launch(self, slot_id: int, batch_size: int) -> None:
        if self._closed:
            raise RuntimeError("Whisper prefill session is shut down")
        if not self._warmed:
            raise RuntimeError("Whisper prefill session must be warmed before launch")
        if not 0 < int(batch_size) <= self._max_batch_size:
            raise ValueError("Whisper prefill batch is outside session capacity")
        key = (int(slot_id), int(batch_size))
        try:
            graph = self._graphs[key]
        except KeyError as exc:
            raise ValueError(
                f"No captured Whisper prefill graph for slot/batch {key}"
            ) from exc
        graph.replay()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._graphs.clear()
            self._workspaces.clear()
            self._buffers.clear()
            self._weights = None
            self._cross_kv = None
            self._self_kv = None


__all__ = ["NativeWhisperPrefillSession"]
