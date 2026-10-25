"""Retained replay bindings must follow tensor ownership, not tensor values."""
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.generated_verification import Qwen35GeneratedVerification


@pytest.mark.parametrize("replace_source", [False, True])
def test_retained_binding_requires_same_source_tensors(monkeypatch, replace_source):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    state = torch.ones(1)
    history = torch.ones(1) * 2
    source = SimpleNamespace(seq_length=10, layers=[SimpleNamespace(
        recurrent_states=state, conv_states=history)])
    target.pending, target.block, target.pending_rows = True, 4, {0}
    target.verified = [object()]
    target.sources, target.start_positions = [source], (10,)
    target.sequences, target.layers = 1, (0,)
    target.accepted = [SimpleNamespace()]
    target.stream = object()
    target.runtime = SimpleNamespace(device="cuda")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: target.stream)
    calls = []
    replay = SimpleNamespace(
        source_owners=((state,), (history,)),
        launch_bound=lambda count: calls.append(("bound", count)),
        launch=lambda states, histories, count: calls.append(("validate", count)))
    target.replays = [replay]
    if replace_source:
        source.layers[0].recurrent_states = state.clone()
    else:
        state.add_(10)
    session = SimpleNamespace()
    target.commit((session, target.verified[0], torch.zeros(1, 4, 2), [1, 2, 3, 4], 2))
    assert calls == [("validate" if replace_source else "bound", 2)]
    assert session.cache.seq_length == 12
    assert session.bonus == 2
    assert not target.pending
