"""Serialized CUDA-graph replay of the native streaming codec."""

from dataclasses import fields
from threading import RLock

import torch

from .codec import IncrementalCodecState, Qwen3TTSIncrementalDecoder


def _state_tensors(states):
    for row, state in enumerate(states):
        for field in fields(state):
            mapping = getattr(state, field.name)
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    yield row, field.name, key, value


class _Replay:
    @torch.inference_mode()
    def __init__(self, decoder, batch, frames):
        parameter = next(decoder.decoder.parameters())
        self.codes = torch.zeros(
            (batch, decoder.decoder.config.num_quantizers, frames),
            dtype=torch.long,
            device=parameter.device,
        )
        states = [IncrementalCodecState() for _ in range(batch)]
        # Materialize fixed-shape histories/KV buffers before capture. Zeroed
        # histories plus context length zero represent a fresh request.
        decoder(self.codes, states)
        for row, group, key, value in _state_tensors(states):
            getattr(states[row], group)[key] = value.contiguous()
        self.inputs = tuple(_state_tensors(states))
        for _, _, _, value in self.inputs:
            value.zero_()
        self.positions_cpu = torch.zeros(
            (batch, frames), dtype=torch.long, pin_memory=True
        )
        self.lengths_cpu = torch.zeros(batch, dtype=torch.long, pin_memory=True)
        self.positions = self.positions_cpu.to(parameter.device)
        self.lengths = self.lengths_cpu.to(parameter.device)
        self.offsets = torch.arange(frames)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=torch.cuda.current_stream()):
            self.pcm = decoder(
                self.codes,
                states,
                position_ids=self.positions,
                context_lengths=self.lengths,
            ).squeeze(1)
            # Dense boundaries keep foreach state transfers batched. Strided
            # KV views cost 1.48 ms extra at H100 B8/F3 (6.63 vs 5.15 ms).
            for row, group, key, value in _state_tensors(states):
                getattr(states[row], group)[key] = value.contiguous()
        self.outputs = tuple(_state_tensors(states))
        self.staging_done = torch.cuda.Event()
        self.staging_done.record()

    @torch.inference_mode()
    def run(self, codes, states, retained_context):
        batch, frames = len(states), int(codes.shape[2])
        self.codes[:batch].copy_(codes)
        self.codes[batch:].zero_()
        destinations, sources, empty = [], [], []
        for row, group, key, destination in self.inputs:
            source = getattr(states[row], group).get(key) if row < batch else None
            if source is None:
                empty.append(destination)
            else:
                destinations.append(destination)
                sources.append(source)
        if destinations:
            torch._foreach_copy_(destinations, sources)
        if empty:
            torch._foreach_zero_(empty)
        # Do not overwrite pinned staging while its previous H2D is in flight.
        self.staging_done.synchronize()
        self.positions_cpu.zero_()
        self.lengths_cpu.zero_()
        for row, state in enumerate(states):
            self.positions_cpu[row].copy_(self.offsets + state.frame_position)
            self.lengths_cpu[row] = state.transformer_context_length
        self.positions.copy_(self.positions_cpu, non_blocking=True)
        self.lengths.copy_(self.lengths_cpu, non_blocking=True)
        self.staging_done.record()
        self.graph.replay()
        destinations, sources = [], []
        for row, group, key, value in self.outputs:
            if row < batch:
                # Request state must outlive this bucket and must not alias its
                # replay outputs when another request later occupies the row.
                mapping = getattr(states[row], group)
                if key not in mapping:
                    mapping[key] = torch.empty_like(value)
                destinations.append(mapping[key])
                sources.append(value)
        torch._foreach_copy_(destinations, sources)
        for state in states:
            state.frame_position += frames
            state.transformer_context_length = min(
                retained_context, state.transformer_context_length + frames
            )
        return self.pcm[:batch]


class TorchCodec:
    """Native codec tower with graphs for serving chunks and partial tails.

    Calls and output consumption are serialized on the supplied compute stream.
    Like the generated runtime, returned PCM is borrowed until the next call.
    """

    def __init__(self, codec, stream, capture_lock: RLock, max_batch_size):
        self.decoder = Qwen3TTSIncrementalDecoder(codec)
        self.stream = stream
        self.capture_lock = capture_lock
        self._graphs = {}
        self.stream.wait_stream(torch.cuda.current_stream())
        self.warmup(max_batch_size)

    def warmup(self, max_batch_size):
        # Serving ramps through 3/2/8/12 frames; final chunks may be any 1–12.
        # Eager partial tails took 15–17 ms versus 3–7 ms captured on H200.
        for frames in range(1, 13):
            batch = 1
            while batch < 2 * max_batch_size:
                self._replay(batch, frames)
                if batch >= max_batch_size:
                    break
                batch *= 2

    def _replay(self, batch, frames):
        key = batch, frames
        if key not in self._graphs:
            with self.capture_lock, torch.cuda.stream(self.stream):
                # Vendor convolution search is scoped to graph construction;
                # replay does not mutate global backend settings.
                with torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled,
                    benchmark=True,
                    deterministic=torch.backends.cudnn.deterministic,
                    allow_tf32=torch.backends.cudnn.allow_tf32,
                ):
                    self._graphs[key] = _Replay(self.decoder, batch, frames)
        return self._graphs[key]

    @torch.inference_mode()
    def run(self, codes, states):
        if codes.ndim != 3 or codes.shape[0] != len(states) or not states:
            raise ValueError("codec codes need one row per streaming state")
        with torch.cuda.stream(self.stream):
            frames = int(codes.shape[2])
            capacity = 1 << (len(states) - 1).bit_length()
            return self._replay(capacity, frames).run(
                codes, states, self.decoder.retained_context
            )
