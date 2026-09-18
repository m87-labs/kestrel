import pytest
import torch

from kestrel.models.qwen35.draft_workspace import DraftLayerKVWorkspace


@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_slot_mapping_separates_padding_and_transient_queries(slots):
    workspace = DraftLayerKVWorkspace(slots=slots, capacity=32, context_rows=4,
                                     query_rows=4, heads=2, head_dim=8, device="cpu")
    starts = [slot + 1 for slot in range(slots)]
    lengths = [slot % 5 for slot in range(slots)]
    mapping, used = workspace.append_inputs(starts, lengths)
    rows = mapping.reshape(slots, 8)
    assert mapping.unique().numel() == mapping.numel()
    for slot, (start, length) in enumerate(zip(starts, lengths)):
        local = rows[slot] - slot * workspace.storage_rows
        assert local[:length].tolist() == list(range(start, start + length))
        assert local[length:4].tolist() == list(range(32 + length, 36))
        assert local[4:].tolist() == list(range(start + length, start + length + 4))
        assert used[slot] == start + length + 4
    # Replaying with fewer newly accepted rows overwrites the old query suffix;
    # no committed prefix or shared padding destination is addressed.
    retry, _ = workspace.append_inputs(starts, [0] * slots)
    assert (retry.reshape(slots, 8)[:, 4:] >=
            torch.tensor(starts)[:, None] + torch.arange(slots)[:, None] * 36).all()


def test_append_rejects_entire_invalid_batch_before_writes():
    workspace = DraftLayerKVWorkspace(slots=2, capacity=8, context_rows=4,
                                     query_rows=4, heads=2, head_dim=8, device="cpu")
    workspace.keys.fill_(123)
    for starts, lengths in (([0, 5], [1, 0]), ([0, 0], [1, 5]), ([-1, 0], [0, 0])):
        with pytest.raises(ValueError):
            workspace.append_inputs(starts, lengths)
    assert (workspace.keys == 123).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_write_keeps_prefix_and_padding_private():
    workspace = DraftLayerKVWorkspace(slots=2, capacity=32, context_rows=4,
                                     query_rows=4, heads=2, head_dim=128, device="cuda")
    workspace.keys.fill_(123)
    workspace.values.fill_(124)
    source = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16)
    for lengths in ([3, 1], [0, 2]):
        mapping, used = workspace.append_inputs([5, 7], lengths)
        keys, values = workspace.write(source, source + 1, mapping)
        assert keys.data_ptr() == workspace.keys.data_ptr()
        for slot, (start, length) in enumerate(zip([5, 7], lengths)):
            assert (keys[slot, :start] == 123).all()
            expected = torch.cat((source[slot * 8:slot * 8 + length],
                                  source[slot * 8 + 4:slot * 8 + 8]))
            assert torch.equal(keys[slot, start:used[slot]], expected)
            assert torch.equal(values[slot, start:used[slot]], expected + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_graph_reuses_addresses_with_changed_live_lengths():
    from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph

    workspace = DraftLayerKVWorkspace(slots=2, capacity=32, context_rows=4,
                                     query_rows=4, heads=2, head_dim=128, device="cuda")
    workspace.keys.fill_(123)
    workspace.values.fill_(124)
    pointer = workspace.keys.data_ptr()
    graph = FixedShapeSinglePassGraph(enabled=True, device=torch.device("cuda:0"),
                                     stream=None, run_forward=workspace.write, max_entries=1)
    try:
        for lengths in ([3, 1], [0, 2], [4, 0]):
            source = torch.randn(2, 8, 2, 128, device="cuda", dtype=torch.bfloat16)
            for slot, length in enumerate(lengths):
                source[slot, length:4].fill_(float("nan"))
            mapping, used = workspace.append_inputs([5, 7], lengths)
            with graph.launch(source.flatten(0, 1), (source + 1).flatten(0, 1), mapping) as result:
                keys, values = result
                assert keys.data_ptr() == pointer
                for slot, (start, length) in enumerate(zip([5, 7], lengths)):
                    expected = torch.cat((source[slot, :length], source[slot, 4:]))
                    assert (keys[slot, :start] == 123).all()
                    assert torch.equal(keys[slot, start:used[slot]], expected)
                    assert torch.equal(values[slot, start:used[slot]], expected + 1)
        assert len(graph._entries) == 1
    finally:
        graph.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sliding", [False, True])
def test_stable_attention_layer_matches_eager_sessions(sliding):
    from types import SimpleNamespace
    from kestrel.models.qwen35.dflash import _Attention, _LayerContextBuffer
    from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph

    torch.manual_seed(241)
    config = SimpleNamespace(head_dim=128, hidden_size=128, num_attention_heads=4,
                             num_key_value_heads=1, rms_norm_eps=1e-6,
                             attention=lambda _: (sliding, 19 if sliding else None))
    layer = _Attention(config, 0).cuda().bfloat16().eval().requires_grad_(False)
    workspace = DraftLayerKVWorkspace(slots=2, capacity=64, context_rows=16,
                                     query_rows=16, heads=1, head_dim=128, device="cuda")
    workspace.keys.normal_()
    workspace.values.normal_()
    caches = [_LayerContextBuffer(workspace.keys[i:i + 1, :64].clone(),
                                  workspace.values[i:i + 1, :64].clone()) for i in range(2)]
    graph = FixedShapeSinglePassGraph(
        enabled=True, device=torch.device("cuda:0"), stream=None,
        run_forward=lambda h, c, cos, sin, mapping, used: (
            layer.forward_stable(h, c, cos, sin, workspace, mapping, used),),
        max_entries=1)
    try:
        for lengths in ([3, 1], [0, 16]):
            hidden = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
            context = torch.randn_like(hidden)
            cos = torch.ones(2, 32, 128, device="cuda", dtype=torch.bfloat16)
            sin = torch.zeros_like(cos)
            mapping, used = workspace.append_inputs([5, 7], lengths)
            with graph.launch(hidden, context, cos, sin, mapping, used) as (actual,):
                for slot, (start, length) in enumerate(zip([5, 7], lengths)):
                    expected = layer(
                        hidden[slot:slot + 1], context[slot:slot + 1, :length],
                        cos[slot:slot + 1, :length + 16], sin[slot:slot + 1, :length + 16],
                        cache=caches[slot], past_context=start, cache_capacity=64)
                    assert torch.equal(actual[slot:slot + 1], expected)
        assert len(graph._entries) == 1
    finally:
        graph.shutdown()


def _session_fixture():
    from kestrel.models.qwen35.dflash import (
        DFlashConfig, DFlashDraftModel, DFlashContextCache, _LayerContextBuffer)
    config = DFlashConfig(128, 256, 1, 4, 1, 128, 1e-6, 10000, 16, 0,
                         (0,), ("full_attention",), None)
    model = DFlashDraftModel(config).cuda().bfloat16().requires_grad_(False)
    caches = []
    for slot in range(2):
        cache = DFlashContextCache(64)
        cache.length = slot + 3
        keys = torch.full((1, 64, 1, 128), slot + 1, device="cuda", dtype=torch.bfloat16)
        cache.layers = [_LayerContextBuffer(keys, keys.clone())]
        caches.append(cache)
    return model, caches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_session_consumer_failure_does_not_commit(monkeypatch):
    from contextlib import contextmanager
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    model, caches = _session_fixture()
    session = DFlashDraftGraphSession(model, caches)
    starts = [cache.length for cache in caches]

    @contextmanager
    def fake_launch(*inputs):
        yield (inputs[0],)

    monkeypatch.setattr(session.graph, "launch", fake_launch)
    noise = [torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
    features = [value[:, :1] for value in noise]
    positions = [torch.arange(start, start + 17, device="cuda")[None] for start in starts]
    try:
        with pytest.raises(RuntimeError, match="consumer failure"):
            with session.launch(noise, features, positions):
                raise RuntimeError("consumer failure")
        assert session.failed
        with pytest.raises(RuntimeError, match="cannot rebind"):
            session.rebind(caches)
        assert [cache.length for cache in caches] == starts
        for slot, cache in enumerate(caches):
            assert (cache.layers[0].keys == slot + 1).all()
        with pytest.raises(RuntimeError, match="failed or retired"):
            with session.launch(noise, features, positions):
                pass
    finally:
        session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_session_retirement_reorder_and_external_advance():
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    model, caches = _session_fixture()
    session = DFlashDraftGraphSession(model, caches)
    caches[0].length += 1
    try:
        with pytest.raises(ValueError, match="advanced outside"):
            with session.launch([None] * 2, [None] * 2, [None] * 2):
                pass
    finally:
        session.shutdown()
    with pytest.raises(RuntimeError, match="retired"):
        with session.launch([], [], []):
            pass
    for owners in ([caches[1], caches[0]], [caches[1]]):
        replacement = DFlashDraftGraphSession(model, owners)
        try:
            for slot, owner in enumerate(owners):
                actual = replacement.workspaces[0].keys[slot]
                assert torch.equal(actual[:owner.length], owner.layers[0].keys[0, :owner.length])
                assert (actual[owner.length:] == 0).all()
        finally:
            replacement.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_rebind_keeps_one_capture_and_rejects_active_lease(monkeypatch):
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    model, caches = _session_fixture()
    session = DFlashDraftGraphSession(model, caches)
    captures = []
    original_capture = session.graph._capture
    def capture(inputs):
        captures.append(1)
        return original_capture(inputs)
    monkeypatch.setattr(session.graph, "_capture", capture)
    noise = [torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
    features = [value[:, :1] for value in noise]
    try:
        for iteration in range(2):
            if iteration:
                _, new_caches = _session_fixture()
                for slot, cache in enumerate(new_caches):
                    cache.layers[0].keys.fill_(slot + 7)
                    cache.layers[0].values.fill_(slot + 8)
                new_caches.reverse()
                session.rebind(new_caches)
                caches = new_caches
            starts = [cache.length for cache in caches]
            positions = [torch.arange(start, start + 17, device="cuda")[None] for start in starts]
            with session.launch(noise, features, positions):
                with pytest.raises(RuntimeError, match="cannot rebind"):
                    session.rebind(caches)
                for slot, start in enumerate(starts):
                    workspace = session.workspaces[0]
                    assert torch.equal(workspace.keys[slot, :start], caches[slot].layers[0].keys[0, :start])
                    assert (workspace.keys[slot, start + 17:64] == 0).all()
        assert len(captures) == 1
    finally:
        session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_staging_survives_delayed_copy_and_allocator_churn():
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    model, caches = _session_fixture()
    session = DFlashDraftGraphSession(model, caches)
    noise = [torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
    features = [value[:, :0] for value in noise]
    positions = [torch.arange(cache.length, cache.length + 16, device="cuda")[None]
                 for cache in caches]
    try:
        with session.launch(noise, features, positions) as output:
            expected = output.clone()
        session.stream.synchronize()
        for _ in range(4):
            with torch.cuda.stream(session.stream):
                torch.cuda._sleep(10000000)
            with session.launch(noise, features, positions) as output:
                actual = output.clone()
            # Staging tensors have left scope while their cross-stream copies
            # may still be pending. Churn matching and neighboring size classes.
            churn = [torch.empty((2, 32, 128), device="cuda", dtype=torch.bfloat16).fill_(321)
                     for _ in range(16)]
            session.stream.synchronize()
            assert torch.equal(actual, expected)
            del churn
    finally:
        session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_draft_commit_is_visible_to_ambient_eager_consumer():
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    model, caches = _session_fixture()
    session = DFlashDraftGraphSession(model, caches)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    starts = [cache.length for cache in caches]
    try:
        with torch.cuda.stream(consumer):
            noise = [torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
            features = [value[:, :1] for value in noise]
            positions = [torch.arange(start, start + 17, device="cuda")[None] for start in starts]
            with session.launch(noise, features, positions) as output:
                output[0, 0, 0].item()
                # Commit copies are submitted after this consumer returns.
                torch.cuda._sleep(10000000)
            for cache, start in zip(caches, starts):
                assert cache.length == start + 1
                assert (cache.layers[0].keys[:, start:start + 1] == 0).all()
                assert (cache.layers[0].values[:, start:start + 1] == 0).all()
            # Exercise the real eager reader after returning from the graph.
            eager_positions = [position + 1 for position in positions]
            result = model.forward_many(noise, features, eager_positions, context_caches=caches)
            assert all(torch.isfinite(value).all() for value in result)
    finally:
        session.shutdown()
