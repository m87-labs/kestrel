"""CUDA-graph replay for batched Parakeet TDT decoding."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch import Tensor

from kestrel.runtime.decode_graph import DecodeGraphManager

from .model import ParakeetTdt, TdtOutput, _decode_batch


@dataclass(slots=True)
class _TdtDecodeSlot:
    encoded: Tensor
    frames: Tensor
    active: Tensor
    decoder_hidden: Tensor
    state: tuple[Tensor, Tensor]
    decisions: Tensor


class _TdtBatchGraphDecoder:
    """Replay one exact TDT step while the host owns stopping and results."""

    minimum_batch = 2

    def __init__(
        self,
        model: ParakeetTdt,
        *,
        max_batch: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        self.model = model
        self.device = model.encoder_projector.weight.device
        self.max_batch = max_batch
        config = model.config
        dtype = model.encoder_projector.weight.dtype
        token = torch.full(
            (max_batch, 1),
            config.blank_token_id,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            initial_decoder_hidden, initial_state = model.decoder(token, None)
        self._initial_decoder_hidden = initial_decoder_hidden
        self._initial_state = initial_state
        self.slot = _TdtDecodeSlot(
            encoded=torch.empty(
                (
                    max_batch,
                    config.encoder.max_position_embeddings,
                    config.decoder_hidden_size,
                ),
                device=self.device,
                dtype=dtype,
            ),
            frames=torch.zeros(max_batch, dtype=torch.long, device=self.device),
            active=torch.zeros(max_batch, dtype=torch.bool, device=self.device),
            decoder_hidden=torch.empty_like(initial_decoder_hidden),
            state=tuple(torch.empty_like(value) for value in initial_state),
            decisions=torch.empty(
                (max_batch, 2), dtype=torch.long, device=self.device
            ),
        )
        self._batch_indices = torch.arange(max_batch, device=self.device)
        self._graphs = DecodeGraphManager[_TdtDecodeSlot](
            enabled=True,
            device=self.device,
            max_batch=max_batch,
            graph_capture_lock=threading.RLock(),
            compute_stream=compute_stream,
            run_forward=self._step,
            prepare_step=lambda _slot, _batch_size: None,
            zero_padding=self._zero_padding,
            zero_for_capture=self._zero_for_capture,
            eager_batch_sizes=(1,),
        )
        self._graphs.ensure_ready((self.slot,))

    def _step(self, slot: _TdtDecodeSlot, batch_size: int) -> None:
        model = self.model
        logits = model.joint(
            slot.encoded[
                self._batch_indices[:batch_size],
                slot.frames[:batch_size].clamp_max(slot.encoded.shape[1] - 1),
            ][:, None],
            slot.decoder_hidden[:batch_size],
        )
        token_ids = logits[..., : model.config.vocab_size].argmax(-1).flatten()
        duration_indices = logits[..., model.config.vocab_size :].argmax(
            -1
        ).flatten()
        candidate_hidden, candidate_state = model.decoder(
            token_ids[:, None],
            tuple(value[:, :batch_size] for value in slot.state),
        )
        emitted = slot.active[:batch_size] & (
            token_ids != model.config.blank_token_id
        )
        hidden_mask = emitted[:, None, None]
        state_mask = hidden_mask.transpose(0, 1)
        slot.decoder_hidden[:batch_size].copy_(
            torch.where(
                hidden_mask,
                candidate_hidden,
                slot.decoder_hidden[:batch_size],
            )
        )
        for current, candidate in zip(slot.state, candidate_state, strict=True):
            current[:, :batch_size].copy_(
                torch.where(
                    state_mask,
                    candidate,
                    current[:, :batch_size],
                )
            )
        slot.decisions[:batch_size].copy_(
            torch.stack((token_ids, duration_indices), dim=1)
        )

    @staticmethod
    def _zero_padding(
        slot: _TdtDecodeSlot, batch_size: int, graph_batch_size: int
    ) -> None:
        slot.active[batch_size:graph_batch_size].zero_()

    @staticmethod
    def _zero_for_capture(slot: _TdtDecodeSlot) -> None:
        slot.encoded.zero_()
        slot.frames.zero_()
        slot.active.zero_()
        slot.decoder_hidden.zero_()
        for value in slot.state:
            value.zero_()
        slot.decisions.zero_()

    def generate(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None,
    ) -> TdtOutput:
        batch, width, _ = encoded.shape
        if batch < 2 or batch > self.max_batch:
            raise ValueError("TDT graph decode requires a supported batched input")
        if width > self.slot.encoded.shape[1]:
            return self.model._generate_batch(
                encoded, valid, max_tokens=max_tokens
            )

        valid_lengths = valid.sum(-1).tolist()
        slot = self.slot
        slot.encoded[:batch, :width].copy_(encoded)
        slot.decoder_hidden.copy_(self._initial_decoder_hidden)
        for current, initial in zip(slot.state, self._initial_state, strict=True):
            current.copy_(initial)

        def step(frames: list[int], active: list[bool]) -> list[list[int]]:
            slot.frames[:batch].copy_(torch.tensor(frames, device=self.device))
            slot.active[:batch].copy_(torch.tensor(active, device=self.device))
            self._graphs.run(slot, batch)
            return slot.decisions[:batch].tolist()

        return _decode_batch(
            self.model.config,
            valid_lengths,
            max_tokens,
            self.device,
            step,
        )
