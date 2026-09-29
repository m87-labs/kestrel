from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35 import spec_replay_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
def test_replay_graph_stages_parameters_and_releases_owned_entries(monkeypatch):
    device = torch.device("cuda")
    runtime = SimpleNamespace(device=device, _compute_stream=torch.cuda.Stream())
    seen = []

    def recurrence(mixed, a, b, A_log, dt_bias, initial, cu, topology):
        seen.append(tuple(initial.shape))
        return initial + A_log.sum() + dt_bias.sum() + mixed.sum() + a.sum() + b.sum()

    monkeypatch.setattr(spec_replay_graph, "_replay_recurrent_prefix", recurrence)
    graph = spec_replay_graph.Qwen35ReplayGraph(runtime, block_size=3, num_layers=2)
    try:
        for count, length, value in ((1, 1, 1), (1, 1, 2), (2, 1, 3),
                                     (1, 2, 4), (2, 2, 5), (1, 1, 6)):
            mixed = torch.full((1, count * length, 3), float(value), device=device)
            a = torch.zeros((1, count * length, 1), device=device)
            params = torch.full((count, 1), float(value), device=device)
            initial = torch.ones((count, 1, 1, 1), device=device)
            expected = initial + 2 * params.sum() + mixed.sum()
            with graph.launch(mixed, a, a, params, params, initial) as (final,):
                owned = final.clone()
            torch.testing.assert_close(owned, expected, atol=0, rtol=0)
        assert seen
        assert len(graph._graphs._entries) <= 3
    finally:
        graph.shutdown()
    assert not graph._topologies
