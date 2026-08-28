"""Generated one-decision execution for Parakeet TDT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec

from .model import ParakeetTdt, TdtOutput, _decode_batch


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
    decisions: Tensor


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
            "token": slot.decisions[0, :capacity],
            "duration": slot.decisions[1, :capacity],
            **state,
            **{f"{name}_next": value for name, value in state.items()},
        }

    @staticmethod
    def launch_extents(
        _slot: _TdtDecodeSlot, batch_size: int
    ) -> Mapping[str, int]:
        return {"active_batch": int(batch_size)}


class _TdtBatchGeneratedDecoder:
    """Run one complete TDT decision per generated kernel launch."""

    minimum_batch = 2

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
        self.slot = _TdtDecodeSlot(
            slot_id=0,
            compute_stream=compute_stream,
            encoded=torch.empty(
                (
                    max_batch,
                    config.encoder.max_position_embeddings,
                    config.decoder_hidden_size,
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            frame_indices=torch.zeros(
                max_batch, dtype=torch.long, device=self.device
            ),
            active=torch.zeros(max_batch, dtype=torch.bool, device=self.device),
            decoder_hidden=torch.empty_like(self._initial_decoder_hidden),
            hidden=tuple(torch.empty_like(value) for value in self._initial_hidden),
            cell=tuple(torch.empty_like(value) for value in self._initial_cell),
            decisions=torch.empty(
                (2, max_batch), dtype=torch.int32, device=self.device
            ),
        )
        self.decode_slots: Sequence[_TdtDecodeSlot] = (self.slot,)
        self._generated: GeneratedDecode

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
        if width > self.slot.encoded.shape[1]:
            return self.model._generate_batch(
                encoded, valid, max_tokens=max_tokens
            )

        valid_lengths = valid.sum(-1).tolist()
        slot = self.slot
        slot.encoded[:batch, :width].copy_(encoded)
        slot.decoder_hidden.copy_(self._initial_decoder_hidden)
        for current, initial in zip(
            (*slot.hidden, *slot.cell),
            (*self._initial_hidden, *self._initial_cell),
            strict=True,
        ):
            current.copy_(initial)
        max_frames = slot.encoded.shape[1]
        launch = self._generated.static_launcher(slot, batch)

        def step(frames: list[int], active: list[bool]) -> list[list[int]]:
            slot.frame_indices[:batch].copy_(torch.tensor(
                [row * max_frames + min(frame, max_frames - 1)
                 for row, frame in enumerate(frames)],
                device=self.device,
            ))
            slot.active[:batch].copy_(torch.tensor(active, device=self.device))
            launch()
            return slot.decisions[:, :batch].T.tolist()

        return _decode_batch(
            self.model.config,
            valid_lengths,
            max_tokens,
            self.device,
            step,
        )


__all__ = ["_TdtBatchGeneratedDecoder"]
