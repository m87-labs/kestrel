"""Owned fixed-shape submission for accepted-prefix recurrence."""

from contextlib import contextmanager

from kestrel_kernels import get_runtime

from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph
from .qwen_model import _replay_recurrent_prefix


class Qwen35ReplayGraph:
    def __init__(self, runtime, block_size, num_layers):
        self._block_size = block_size
        self._num_layers = num_layers
        self._topologies = {}
        self._graphs = FixedShapeSinglePassGraph(
            enabled=True, device=runtime.device, stream=runtime._compute_stream,
            run_forward=self._forward, max_entries=block_size)

    def _forward(self, mixed, a, b, A_log, dt_bias, initial):
        count = initial.shape[0]
        if not 1 <= count <= self._num_layers or mixed.shape[1] % count:
            raise ValueError("replay graph requires bounded equal-length sequences")
        length = mixed.shape[1] // count
        if not 1 <= length < self._block_size:
            raise ValueError("replay graph requires a partial speculative prefix")
        key = (count, length)
        if key not in self._topologies:
            self._topologies[key] = get_runtime().gated_delta.bind_packed_prefill_topology(
                sequence_lengths=(length,) * count, device=mixed.device)
        cu, topology = self._topologies[key]
        final = _replay_recurrent_prefix(mixed, a, b, A_log, dt_bias, initial, cu, topology)
        return final, cu

    @contextmanager
    def launch(self, *inputs):
        with self._graphs.launch(*inputs) as (final, _cu):
            yield (final,)

    def shutdown(self):
        try:
            self._graphs.shutdown()
        finally:
            self._topologies.clear()
