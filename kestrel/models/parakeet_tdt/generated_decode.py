"""Generated one-decision execution for Parakeet TDT."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from kestrel.device import make_event, stream_context
from kestrel.runtime.decode_graph import DecodeGraphManager
from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec
from kestrel.utils import CpuGpuBuffer

from .model import ParakeetTdt, TdtOutput, _TdtBatchState


@dataclass(slots=True)
class _TdtDecodeSlot:
    slot_id: int
    compute_stream: torch.cuda.Stream
    encoded: Tensor
    frame_indices: Tensor
    active: Tensor
    decoder_hidden: Tensor
    hidden: tuple[Tensor, ...]
    cell: tuple[Tensor, ...]
    decisions: CpuGpuBuffer
    frames: Tensor
    valid_lengths: Tensor
    steps_remaining: Tensor
    tokens_remaining: Tensor
    batch_offsets: Tensor
    duration_values: Tensor
    read_done: Any


@dataclass(frozen=True, slots=True)
class _TdtDecodeBindings:
    def is_eligible(self, runtime: Any) -> bool:
        config = runtime.model.config
        return (
            config.decoder_hidden_size == 640
            and config.vocab_size == 8193
            and config.blank_token_id == 8192
            and len(config.durations) == 5
            and config.num_decoder_layers == 2
            and config.encoder.max_position_embeddings == 5000
        )

    def runtime_inputs(self, _runtime: Any) -> Mapping[str, Any]:
        return {}

    def slot_inputs(self, slot: _TdtDecodeSlot, capacity: int) -> Mapping[str, Any]:
        state = {
            "decoder_hidden": slot.decoder_hidden[:capacity],
            **{
                f"hidden_{layer}": value[:capacity]
                for layer, value in enumerate(slot.hidden)
            },
            **{
                f"cell_{layer}": value[:capacity]
                for layer, value in enumerate(slot.cell)
            },
        }
        return {
            "encoded": slot.encoded[:capacity].flatten(0, 1),
            "frame_indices": slot.frame_indices[:capacity],
            "active": slot.active[:capacity],
            "token": slot.decisions.gpu[0, :capacity],
            "duration": slot.decisions.gpu[1, :capacity],
            **state,
            **{f"{name}_next": value for name, value in state.items()},
        }

    @staticmethod
    def launch_extents(
        _slot: _TdtDecodeSlot, batch_size: int
    ) -> Mapping[str, int]:
        return {"active_batch": int(batch_size)}


class _TdtBatchGeneratedDecoder:
    """Pipeline generated TDT decisions over two asynchronous readback slots."""

    minimum_batch = 1

    @classmethod
    def create(
        cls,
        model: ParakeetTdt,
        *,
        max_batch: int,
        compute_stream: torch.cuda.Stream,
        required: bool,
    ) -> _TdtBatchGeneratedDecoder | None:
        decoder = cls(model, max_batch=max_batch, compute_stream=compute_stream)
        spec = GeneratedDecodeSpec(
            label="Parakeet TDT",
            weight_root=model,
            weight_layer_prefix="",
            bindings=_TdtDecodeBindings(),
        )
        batch_sizes = range(cls.minimum_batch, max_batch + 1)
        generated = (
            GeneratedDecode.require(decoder, spec, batch_sizes=batch_sizes)
            if required
            else GeneratedDecode.try_create(
                decoder, spec, required_batch_sizes=batch_sizes
            )
        )
        if generated is None:
            return None
        decoder._generated = generated
        decoder._graphs = DecodeGraphManager[_TdtDecodeSlot](
            enabled=True,
            device=decoder.device,
            max_batch=max_batch,
            graph_capture_lock=threading.RLock(),
            compute_stream=compute_stream,
            run_forward=decoder._run_step,
            prepare_step=lambda _slot, _batch_size: None,
            zero_padding=decoder._zero_padding,
            zero_for_capture=decoder._zero_for_capture,
        )
        decoder._graphs.ensure_ready(decoder.decode_slots)
        return decoder

    def __init__(
        self,
        model: ParakeetTdt,
        *,
        max_batch: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        self.model = model
        self.device = model.encoder_projector.weight.device
        self.dtype = model.encoder_projector.weight.dtype
        self.max_batch_size = int(max_batch)
        self.compute_stream = compute_stream
        config = model.config
        token = torch.full(
            (max_batch, 1),
            config.blank_token_id,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            initial_decoder_hidden, initial_state = model.decoder(token, None)
        initial_hidden, initial_cell = initial_state
        self._initial_decoder_hidden = initial_decoder_hidden[:, 0]
        self._initial_hidden = tuple(initial_hidden.unbind())
        self._initial_cell = tuple(initial_cell.unbind())
        max_frames = config.encoder.max_position_embeddings
        encoded = torch.empty(
            (max_batch, max_frames, config.decoder_hidden_size),
            device=self.device,
            dtype=self.dtype,
        )
        frame_indices = torch.zeros(
            max_batch, dtype=torch.long, device=self.device
        )
        active = torch.zeros(max_batch, dtype=torch.bool, device=self.device)
        decoder_hidden = torch.empty_like(self._initial_decoder_hidden)
        hidden = tuple(torch.empty_like(value) for value in self._initial_hidden)
        cell = tuple(torch.empty_like(value) for value in self._initial_cell)
        frames = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        valid_lengths = torch.zeros(
            max_batch, dtype=torch.long, device=self.device
        )
        steps_remaining = torch.zeros(
            max_batch, dtype=torch.long, device=self.device
        )
        tokens_remaining = torch.zeros(
            max_batch, dtype=torch.long, device=self.device
        )
        batch_offsets = (
            torch.arange(max_batch, device=self.device, dtype=torch.long)
            * max_frames
        )
        duration_values = torch.tensor(
            config.durations, device=self.device, dtype=torch.long
        )

        # The recurrence is one shared causal device state. Only decisions are
        # double-buffered so the host can consume step N while step N+1 runs.
        def make_slot(slot_id: int) -> _TdtDecodeSlot:
            return _TdtDecodeSlot(
                slot_id=slot_id,
                compute_stream=compute_stream,
                encoded=encoded,
                frame_indices=frame_indices,
                active=active,
                decoder_hidden=decoder_hidden,
                hidden=hidden,
                cell=cell,
                decisions=CpuGpuBuffer(
                    2,
                    max_batch,
                    dtype=torch.int32,
                    device=self.device,
                    pin_memory=True,
                    with_numpy=False,
                    zero=False,
                ),
                frames=frames,
                valid_lengths=valid_lengths,
                steps_remaining=steps_remaining,
                tokens_remaining=tokens_remaining,
                batch_offsets=batch_offsets,
                duration_values=duration_values,
                read_done=make_event(self.device),
            )

        self.decode_slots: Sequence[_TdtDecodeSlot] = (
            make_slot(0),
            make_slot(1),
        )
        self._generated: GeneratedDecode
        self._graphs: DecodeGraphManager[_TdtDecodeSlot]

    def _advance(self, slot: _TdtDecodeSlot, batch: int) -> None:
        config = self.model.config
        active = slot.active[:batch]
        token = slot.decisions.gpu[0, :batch].to(torch.long)
        duration_index = slot.decisions.gpu[1, :batch].to(torch.long)
        duration = slot.duration_values[duration_index]
        duration = torch.where(
            (token == config.blank_token_id) & (duration == 0),
            1,
            duration,
        )
        duration = torch.where(active, duration, 0)
        emitted = active & (token != config.blank_token_id)
        slot.frames[:batch].add_(duration)
        slot.steps_remaining[:batch].sub_(active.to(torch.long))
        slot.tokens_remaining[:batch].sub_(emitted.to(torch.long))
        slot.active[:batch].copy_(
            (slot.frames[:batch] < slot.valid_lengths[:batch])
            & (slot.steps_remaining[:batch] > 0)
            & (slot.tokens_remaining[:batch] > 0)
        )
        slot.frame_indices[:batch].copy_(
            slot.batch_offsets[:batch]
            + slot.frames[:batch].clamp_max(slot.encoded.shape[1] - 1)
        )

    def _run_step(self, slot: _TdtDecodeSlot, batch: int) -> None:
        self._generated.static_launcher(slot, batch)()
        self._advance(slot, batch)

    @staticmethod
    def _zero_padding(
        slot: _TdtDecodeSlot, batch: int, graph_batch: int
    ) -> None:
        slot.active[batch:graph_batch].zero_()

    @staticmethod
    def _zero_for_capture(slot: _TdtDecodeSlot) -> None:
        slot.encoded.zero_()
        slot.frame_indices.zero_()
        slot.active.zero_()
        slot.decoder_hidden.zero_()
        for value in (*slot.hidden, *slot.cell):
            value.zero_()
        slot.decisions.gpu.zero_()
        slot.frames.zero_()
        slot.valid_lengths.zero_()
        slot.steps_remaining.zero_()
        slot.tokens_remaining.zero_()

    def _launch(self, slot: _TdtDecodeSlot, batch: int) -> None:
        with stream_context(self.compute_stream):
            self._graphs.run(slot, batch)
            slot.decisions.copy_to_cpu()
            slot.read_done.record(self.compute_stream)

    @staticmethod
    def _read(slot: _TdtDecodeSlot, batch: int) -> list[list[int]]:
        slot.read_done.synchronize()
        return slot.decisions.cpu[:, :batch].T.tolist()

    def generate(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None,
    ) -> TdtOutput:
        batch, width, _ = encoded.shape
        if not self.minimum_batch <= batch <= self.max_batch_size:
            raise ValueError("TDT generated decode requires a supported batch")
        control = self.decode_slots[0]
        if width > control.encoded.shape[1]:
            return self.model._generate_batch(
                encoded, valid, max_tokens=max_tokens
            )

        valid_tensor = valid.sum(-1)
        valid_lengths = valid_tensor.tolist()
        state = _TdtBatchState.create(
            self.model.config, valid_lengths, max_tokens
        )
        with stream_context(self.compute_stream):
            control.encoded[:batch, :width].copy_(encoded)
            control.decoder_hidden.copy_(self._initial_decoder_hidden)
            for current, initial in zip(
                (*control.hidden, *control.cell),
                (*self._initial_hidden, *self._initial_cell),
                strict=True,
            ):
                current.copy_(initial)
            control.frames[:batch].zero_()
            control.valid_lengths[:batch].copy_(valid_tensor)
            control.steps_remaining[:batch].copy_(
                valid_tensor * self.model.config.max_symbols_per_step
            )
            control.tokens_remaining[:batch].fill_(
                torch.iinfo(torch.long).max if max_tokens is None else max_tokens
            )
            control.active[:batch].copy_(
                (control.valid_lengths[:batch] > 0)
                & (control.steps_remaining[:batch] > 0)
                & (control.tokens_remaining[:batch] > 0)
            )
            control.frame_indices[:batch].copy_(control.batch_offsets[:batch])

        available = deque(self.decode_slots)
        pending: deque[_TdtDecodeSlot] = deque()
        active = state.active()
        while any(active):
            while len(pending) < 2:
                slot = available.popleft()
                self._launch(slot, batch)
                pending.append(slot)
            slot = pending.popleft()
            decisions = self._read(slot, batch)
            available.append(slot)
            state.commit(decisions, active)
            active = state.active()

        for slot in pending:
            slot.read_done.synchronize()
        with stream_context(self.compute_stream):
            return state.output(self.device)


__all__ = ["_TdtBatchGeneratedDecoder"]
