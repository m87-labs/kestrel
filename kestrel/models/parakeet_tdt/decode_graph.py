"""CUDA-graph replay for batched Parakeet TDT decoding."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch import Tensor

from kestrel.runtime.decode_graph import DecodeGraphManager

from .model import ParakeetTdt, TdtOutput


# How many steps the host stays ahead of the stopping signal. Each step's
# ``active`` flags go to pinned memory behind their own event, and the host
# reads them back this many steps later, so it is always enqueueing work the
# device has not reached. The lag costs the steps enqueued after the last row
# stopped; they write nothing, and two is a small share of the ~29 steps an
# AMI cohort takes.
_STOP_SIGNAL_LAG = 2

# The walk's state, beside the model state, one value per row.
_WALK = (
    "lengths",
    "steps_left",
    "tokens_left",
    "cursor",
    "step_tokens",
    "step_durations",
)


@dataclass(slots=True)
class _TdtDecodeSlot:
    """Every tensor one decode step reads or writes, at a fixed address."""

    encoded: Tensor
    frames: Tensor
    active: Tensor
    decoder_hidden: Tensor
    state: tuple[Tensor, Tensor]
    lengths: Tensor
    steps_left: Tensor
    tokens_left: Tensor
    cursor: Tensor
    step_tokens: Tensor
    step_durations: Tensor


class _TdtBatchGraphDecoder:
    """Replay one exact TDT step, with the loop's bookkeeping on the device."""

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
            **{
                name: torch.zeros(max_batch, dtype=torch.long, device=self.device)
                for name in _WALK
            },
        )
        self._batch_indices = torch.arange(max_batch, device=self.device)
        self._durations = torch.tensor(
            config.durations, dtype=torch.long, device=self.device
        )
        ring = _STOP_SIGNAL_LAG + 1
        self._stop_flags = torch.zeros(
            (ring, max_batch), dtype=torch.bool, device="cpu", pin_memory=True
        )
        self._stop_events = tuple(
            torch.cuda.Event(enable_timing=False) for _ in range(ring)
        )
        self._graphs = DecodeGraphManager[_TdtDecodeSlot](
            enabled=True,
            device=self.device,
            max_batch=max_batch,
            graph_capture_lock=threading.RLock(),
            compute_stream=compute_stream,
            run_forward=self._step,
            prepare_step=lambda _slot, _batch_size: None,
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
        self._advance(slot, batch_size, token_ids, duration_indices)

    @staticmethod
    def _walks_on(slot: _TdtDecodeSlot, batch_size: int) -> None:
        """A row walks on while it has a frame left and budget for a symbol."""

        torch.logical_and(
            slot.frames[:batch_size] < slot.lengths[:batch_size],
            (slot.steps_left[:batch_size] > 0)
            & (slot.tokens_left[:batch_size] > 0),
            out=slot.active[:batch_size],
        )

    def _advance(
        self,
        slot: _TdtDecodeSlot,
        batch_size: int,
        token_ids: Tensor,
        duration_indices: Tensor,
    ) -> None:
        """Commit this step's decision for every active row, on the device.

        Exactly the integer arithmetic the host used to run a row at a time:
        record the symbol and its duration, advance the frame by that
        duration, and count down the two budgets the caller owns -- symbols
        per frame, and ``max_tokens``. An inactive row takes the step as a
        no-op, so a row is active from step zero until its last and never
        again, which is what lets the caller write each step at a column it
        already knows.
        """

        blank = self.model.config.blank_token_id
        advanced = slot.active[:batch_size].long()
        durations = self._durations[duration_indices]
        # A blank that would not advance the frame advances it by one, or the
        # walk could never reach the end of the encoding.
        durations = durations + ((token_ids == blank) & (durations == 0)).long()
        slot.step_tokens[:batch_size].copy_(token_ids)
        slot.step_durations[:batch_size].copy_(durations)
        slot.frames[:batch_size].add_(durations * advanced)
        slot.cursor[:batch_size].add_(advanced)
        slot.steps_left[:batch_size].sub_(advanced)
        slot.tokens_left[:batch_size].sub_(advanced * (token_ids != blank))
        self._walks_on(slot, batch_size)

    @staticmethod
    def _zero_for_capture(slot: _TdtDecodeSlot) -> None:
        slot.encoded.zero_()
        slot.frames.zero_()
        slot.active.zero_()
        slot.decoder_hidden.zero_()
        for value in slot.state:
            value.zero_()
        for name in _WALK:
            getattr(slot, name).zero_()

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

        # The host's whole per-step job is the replay, two column writes, and
        # a 128-byte copy of the activity flags. What it used to be: push this
        # step's frame indices and flags up out of pageable memory, replay,
        # read the decisions back through a blocking copy, then walk 128 rows
        # of Python twice. A step's device work is a few microseconds; that
        # was two hundred of host.
        config = self.model.config
        slot = self.slot
        blank = config.blank_token_id
        lengths = valid.sum(-1)
        slot.encoded[:batch, :width].copy_(encoded)
        slot.decoder_hidden.copy_(self._initial_decoder_hidden)
        for current, initial in zip(slot.state, self._initial_state, strict=True):
            current.copy_(initial)
        slot.frames.zero_()
        # A row past ``batch`` has no length, so it can never walk: the graph
        # runs it, ``_walks_on`` leaves it inactive, and it advances nothing.
        # That is what the bucket padding needs; it needs nothing else.
        slot.lengths.zero_()
        slot.lengths[:batch].copy_(lengths)
        slot.steps_left[:batch].copy_(lengths * config.max_symbols_per_step)
        # "No budget" is a count no walk can exhaust: one token per step.
        slot.tokens_left[:batch].fill_(
            1 << 40 if max_tokens is None else max_tokens
        )
        slot.cursor[:batch].fill_(1)  # column zero holds the leading blank
        slot.active.zero_()
        self._walks_on(slot, batch)

        # A row walks from step zero until its last, so step ``index`` writes
        # column ``index + 1`` -- of every row, the stopped ones included,
        # whose columns past their own length the tail fill blanks at the end.
        # A row takes at most ``max_symbols_per_step`` steps per frame and the
        # host runs ``_STOP_SIGNAL_LAG`` steps past the last active one, so no
        # walk can write outside this.
        room = config.max_symbols_per_step * width + _STOP_SIGNAL_LAG + 1
        tokens_out = torch.full(
            (batch, room), blank, dtype=torch.long, device=self.device
        )
        durations_out = torch.zeros(
            (batch, room), dtype=torch.long, device=self.device
        )

        flags, events = self._stop_flags, self._stop_events
        ring = len(events)

        def enqueue(index: int) -> None:
            self._graphs.run(slot, batch)
            tokens_out[:, index + 1] = slot.step_tokens[:batch]
            durations_out[:, index + 1] = slot.step_durations[:batch]
            row = index % ring
            flags[row, :batch].copy_(slot.active[:batch], non_blocking=True)
            events[row].record()

        for index in range(_STOP_SIGNAL_LAG):
            enqueue(index)
        index = 0
        while True:
            events[index % ring].synchronize()
            if not bool(flags[index % ring, :batch].any()):
                break
            enqueue(index + _STOP_SIGNAL_LAG)
            index += 1

        cursor = slot.cursor[:batch].clone()  # the next cohort overwrites the slot
        emitted = int(cursor.max())
        past = torch.arange(emitted, device=self.device)[None] >= cursor[:, None]
        return TdtOutput(
            tokens_out[:, :emitted].masked_fill_(past, blank),
            durations_out[:, :emitted].masked_fill_(past, 0),
            cursor,
        )
