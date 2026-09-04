"""Generated one-decision execution for Parakeet TDT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec
from kestrel.utils import CpuGpuBuffer

from .model import ParakeetTdt, TdtOutput, TdtState, _TdtBatchDecodeState


@dataclass(slots=True)
class _TdtDecodeSlot:
    slot_id: int
    compute_stream: torch.cuda.Stream
    encoded: Tensor
    valid_lengths: Tensor
    frame_indices: Tensor
    active: Tensor
    decoder_hidden: Tensor
    hidden: tuple[Tensor, ...]
    cell: tuple[Tensor, ...]
    decisions: CpuGpuBuffer
    read_done: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class _TdtDecodeBindings:
    def is_eligible(self, runtime: Any) -> bool:
        config = runtime.model.config
        return (
            config.decoder_hidden_size == 640
            and config.vocab_size == 8193
            and config.blank_token_id == 8192
            and tuple(config.durations) == (0, 1, 2, 3, 4)
            and config.num_decoder_layers == 2
            and config.encoder.max_position_embeddings == 5000
        )

    def runtime_inputs(self, _runtime: Any) -> Mapping[str, Any]:
        return {}

    def slot_inputs(self, slot: _TdtDecodeSlot, capacity: int) -> Mapping[str, Any]:
        decisions = slot.decisions.gpu.view(2, -1)
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
            "valid_lengths": slot.valid_lengths[:capacity],
            "frame_indices": slot.frame_indices[:capacity],
            "frame_indices_next": slot.frame_indices[:capacity],
            "active": slot.active[:capacity],
            "active_next": slot.active[:capacity],
            "token": decisions[0, :capacity],
            "duration": decisions[1, :capacity],
            **state,
            **{f"{name}_next": value for name, value in state.items()},
        }

    @staticmethod
    def launch_extents(_slot: _TdtDecodeSlot, batch_size: int) -> Mapping[str, int]:
        return {"active_batch": int(batch_size)}


class _TdtBatchGeneratedDecoder:
    """Run one complete TDT decision per generated kernel launch."""

    # Tried an outer CUDA graph plus PyTorch tensor recurrence: it returned an
    # empty L4 transcript; graph-free recurrence slowed C1 75.4 -> 127.2 ms.
    # Keep recurrence inside the generated decision and pipeline only readback.

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
        encoded = torch.empty(
            (
                max_batch,
                config.encoder.max_position_embeddings,
                config.decoder_hidden_size,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        valid_lengths = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        frame_indices = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        active = torch.zeros(max_batch, dtype=torch.bool, device=self.device)
        decoder_hidden = torch.empty_like(self._initial_decoder_hidden)
        hidden = tuple(torch.empty_like(value) for value in self._initial_hidden)
        cell = tuple(torch.empty_like(value) for value in self._initial_cell)
        self._initial_frame_indices = (
            torch.arange(max_batch, dtype=torch.long, device=self.device)
            * config.encoder.max_position_embeddings
        )
        self._inactive = torch.zeros(1, dtype=torch.bool, pin_memory=True)
        self.decode_slots: Sequence[_TdtDecodeSlot] = tuple(
            _TdtDecodeSlot(
                slot_id=slot_id,
                compute_stream=compute_stream,
                encoded=encoded,
                valid_lengths=valid_lengths,
                frame_indices=frame_indices,
                active=active,
                decoder_hidden=decoder_hidden,
                hidden=hidden,
                cell=cell,
                decisions=CpuGpuBuffer(
                    2 * max_batch,
                    dtype=torch.int32,
                    device=self.device,
                    pin_memory=True,
                    zero=False,
                ),
                read_done=torch.cuda.Event(enable_timing=False),
            )
            for slot_id in range(2)
        )
        self._generated: GeneratedDecode
        self._launchers: dict[int, tuple[Callable[[], None], ...]] = {}

    def _launchers_for(self, batch: int) -> tuple[Callable[[], None], ...]:
        launchers = self._launchers.get(batch)
        if launchers is None:
            launchers = tuple(
                self._generated.static_launcher(slot, batch)
                for slot in self.decode_slots
            )
            self._launchers[batch] = launchers
        return launchers

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
        first_slot = self.decode_slots[0]
        if width > first_slot.encoded.shape[1]:
            return self.model._generate_batch(encoded, valid, max_tokens=max_tokens)

        state = self._run(encoded, valid, max_tokens=max_tokens)
        return state.output(self.device)

    def generate_windows(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None,
        start_frames: Sequence[int],
        frame_counts: Sequence[int | None],
        states: Sequence[TdtState | None],
    ) -> tuple[TdtOutput, ...]:
        """Decode a cohort of windows, retaining recurrent state on the GPU."""
        batch = encoded.shape[0]
        if not (len(start_frames) == len(frame_counts) == len(states) == batch):
            raise ValueError("TDT windows must match the encoded batch")
        if encoded.shape[1] > self.decode_slots[0].encoded.shape[1]:
            return tuple(
                self.model.generate_encoded(
                    encoded[row : row + 1],
                    valid[row : row + 1],
                    max_tokens=max_tokens,
                    start_frame=start_frames[row],
                    frame_count=frame_counts[row],
                    state=states[row],
                )
                for row in range(batch)
            )
        with torch.cuda.stream(self.compute_stream):
            state = self._run(
                encoded,
                valid,
                max_tokens=max_tokens,
                start_frames=start_frames,
                frame_counts=frame_counts,
                previous_states=states,
            )
            slot = self.decode_slots[0]
            # Returned rows retain immutable cohort snapshots, independent of
            # launch storage. Copy each state tensor once, not once per row.
            decoder_hidden = slot.decoder_hidden[:batch].clone()
            hidden = torch.stack([value[:batch] for value in slot.hidden], dim=0)
            cell = torch.stack([value[:batch] for value in slot.cell], dim=0)
            return tuple(
                TdtOutput(
                    sequences=torch.tensor([state.sequences[row]], device=self.device),
                    durations=torch.tensor([state.durations[row]], device=self.device),
                    lengths=torch.tensor(
                        [len(state.sequences[row])], device=self.device
                    ),
                    state=TdtState(
                        decoder_hidden[row : row + 1, None],
                        hidden[:, row : row + 1],
                        cell[:, row : row + 1],
                        max(0, state.frames[row] - state.valid_lengths[row]),
                    ),
                )
                for row in range(batch)
            )

    def _run(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None,
        start_frames: Sequence[int] | None = None,
        frame_counts: Sequence[int | None] | None = None,
        previous_states: Sequence[TdtState | None] | None = None,
    ) -> _TdtBatchDecodeState:
        batch, width, _ = encoded.shape
        if not self.minimum_batch <= batch <= self.max_batch_size:
            raise ValueError("TDT generated decode requires a supported batch")
        first_slot = self.decode_slots[0]

        with torch.cuda.stream(self.compute_stream):
            valid_device = valid.sum(-1).to(dtype=torch.long)
            valid_lengths = valid_device.tolist()
            starts = [0] * batch if start_frames is None else list(start_frames)
            counts = [None] * batch if frame_counts is None else list(frame_counts)
            previous = (
                [None] * batch if previous_states is None else list(previous_states)
            )
            ends = [
                length if count is None else min(length, start + count)
                for length, start, count in zip(
                    valid_lengths, starts, counts, strict=True
                )
            ]
            if any(
                not 0 <= start <= end for start, end in zip(starts, ends, strict=True)
            ):
                raise ValueError("TDT decode frames are outside the encoded audio")
            state = _TdtBatchDecodeState.create(self.model.config, ends, max_tokens)
            for row, (start, end, prior) in enumerate(
                zip(starts, ends, previous, strict=True)
            ):
                carry = 0 if prior is None else prior.carry
                state.frames[row] = start + carry
                state.durations[row][0] = min(carry, end - start)
                state.steps_remaining[row] = self.model.config.max_symbols_per_step * (
                    end - start
                )
            active = state.active()

            first_slot.encoded[:batch, :width].copy_(encoded)
            first_slot.valid_lengths[:batch].copy_(
                torch.tensor(ends, device=self.device)
            )
            first_slot.frame_indices[:batch].copy_(
                self._initial_frame_indices[:batch]
                + torch.tensor(state.frames, device=self.device).clamp_max(
                    first_slot.encoded.shape[1] - 1
                )
            )
            first_slot.active[:batch].copy_(torch.tensor(active, device=self.device))
            torch.cat(
                [
                    self._initial_decoder_hidden[row : row + 1]
                    if prior is None
                    else prior.decoder_hidden[:, 0]
                    for row, prior in enumerate(previous)
                ],
                out=first_slot.decoder_hidden[:batch],
            )
            for layer, (hidden, cell) in enumerate(
                zip(first_slot.hidden, first_slot.cell, strict=True)
            ):
                for destination, initial, field in (
                    (hidden, self._initial_hidden[layer], "hidden"),
                    (cell, self._initial_cell[layer], "cell"),
                ):
                    torch.cat(
                        [
                            initial[row : row + 1]
                            if prior is None
                            else getattr(prior, field)[layer]
                            for row, prior in enumerate(previous)
                        ],
                        out=destination[:batch],
                    )

            if not any(active):
                return state

            launchers = self._launchers_for(batch)
            pending: deque[_TdtDecodeSlot] = deque()

            def enqueue(slot: _TdtDecodeSlot) -> None:
                launchers[slot.slot_id]()
                slot.decisions.copy_to_cpu()
                slot.read_done.record(self.compute_stream)
                pending.append(slot)

            def fill_pipeline() -> None:
                # A pending decision can spend at most one token/step. Near a
                # host policy limit, don't enqueue a decision whose state might
                # later have to be rolled back at a streaming boundary.
                # Terminal requests discard the state and retain full lookahead.
                budget = 2
                if previous_states is not None:
                    budget = min(
                        min(steps, tokens if tokens is not None else steps)
                        for steps, tokens, enabled in zip(
                            state.steps_remaining,
                            state.tokens_remaining,
                            active,
                            strict=True,
                        )
                        if enabled
                    )
                for candidate in self.decode_slots:
                    if len(pending) >= min(2, budget):
                        break
                    if not any(item is candidate for item in pending):
                        enqueue(candidate)

            fill_pipeline()
            while pending:
                slot = pending.popleft()
                slot.read_done.synchronize()
                decisions = slot.decisions.cpu.view(2, self.max_batch_size)[
                    :, :batch
                ].T.tolist()
                newly_policy_stopped = state.commit(decisions, active)
                active = state.active()
                if not any(active):
                    break

                for row in newly_policy_stopped:
                    first_slot.active[row : row + 1].copy_(
                        self._inactive, non_blocking=True
                    )
                fill_pipeline()

            return state


__all__ = ["_TdtBatchGeneratedDecoder"]
