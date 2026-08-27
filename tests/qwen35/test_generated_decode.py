from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35 import generated_decode as qwen_generated
from kestrel.models.qwen35.cache import (
    Qwen35InferenceCache,
    Qwen35LinearStatePool,
)
from kestrel.models.qwen35.qwen_model import (
    Qwen3_5GatedDeltaNet,
    _PackedGatedDeltaPrefillWorkspaceCache,
)
from kestrel.runtime import generated_decode as runtime_generated
from kestrel.runtime.carried_state import (
    StatePhysicalForm,
    StateRepresentationRequirement,
)


def _linear_config():
    return SimpleNamespace(
        layer_types=("linear_attention", "full_attention", "linear_attention"),
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=4,
    )


def _state_pool(config=None):
    config = _linear_config() if config is None else config
    pool = Qwen35LinearStatePool(
        config=config,
        max_batch_slots=4,
        device=torch.device("cpu"),
        replay_capacity=2,
    )
    pool.initialize_from_config(config, dtype=torch.bfloat16)
    return pool


def _generated_form():
    return StatePhysicalForm(
        representation="materialized",
        storage_axis_order=("state_row", "value_head", "value", "key"),
        storage_dtype="bf16",
    )


def _inference_cache(config=None):
    config = _linear_config() if config is None else config
    return Qwen35InferenceCache(
        config=config,
        paged_kv=(None, object(), None),
        replay_capacity=2,
    )


def test_generated_state_pool_is_one_bf16_value_major_representation():
    pool = _state_pool()

    recurrent = pool.recurrent_tensors_for_form(_generated_form())

    assert len(recurrent) == 3
    assert recurrent[1] is None
    for layer_idx in (0, 2):
        state = recurrent[layer_idx]
        storage = pool.layers[layer_idx]
        assert state is not None and storage is not None
        assert state is storage.recurrent_states
        assert state.shape == (4, 2, 3, 2)
        assert state.dtype == torch.bfloat16
        assert state.is_contiguous()
        assert storage.replay_checkpoint_states is None
        assert storage.replay_k is None
        assert storage.replay_u is None
        assert storage.replay_g is None
        assert storage.replay_lengths is None

    with pytest.raises(RuntimeError, match="cannot switch to native replay"):
        pool.initialize_native_recurrent()


def test_native_state_pool_remains_mutually_exclusive_fp32_replay():
    pool = _state_pool()

    pool.initialize_native_recurrent()

    for layer_idx in (0, 2):
        storage = pool.layers[layer_idx]
        assert storage is not None and storage.recurrent_states is not None
        assert storage.recurrent_states.shape == (4, 2, 2, 3)
        assert storage.recurrent_states.dtype == torch.float32
        assert storage.replay_checkpoint_states is not None
        assert storage.replay_checkpoint_states.shape == (4, 2, 3, 2)
        assert storage.replay_checkpoint_states.dtype == torch.float32

    with pytest.raises(RuntimeError, match="cannot switch to generated decode"):
        pool.recurrent_tensors_for_form(_generated_form())


@pytest.mark.parametrize(
    "form",
    (
        StatePhysicalForm(
            "materialized",
            ("state_row", "value_head", "key", "value"),
            "bf16",
        ),
        StatePhysicalForm(
            "materialized",
            ("state_row", "value_head", "value", "key"),
            "fp32",
        ),
        StatePhysicalForm(
            "replay",
            ("state_row", "value_head", "value", "key"),
            "bf16",
        ),
    ),
)
def test_generated_state_pool_rejects_incompatible_physical_forms(form):
    with pytest.raises(ValueError, match="materialized BF16 value-major"):
        _state_pool().recurrent_tensors_for_form(form)


def test_generated_prefill_writes_pool_rows_directly_and_reset_is_row_scoped():
    config = _linear_config()
    pool = _state_pool(config)
    recurrent = pool.recurrent_tensors_for_form(_generated_form())
    cache = _inference_cache(config)
    pool.bind_generated_prefill_state(cache)
    indices = torch.tensor([3, 1], dtype=torch.long)

    for layer_idx in (0, 2):
        target = recurrent[layer_idx]
        layer = cache.layers[layer_idx]
        assert target is not None
        assert layer.recurrent_states is target
        target[3].fill_(layer_idx + 1)
        target[1].fill_(layer_idx + 2)
        layer.conv_states = torch.full(
            (2, 10, 4), layer_idx + 3, dtype=torch.bfloat16)
        layer.has_previous_state = True

    pool.capture_batch_from_cache(indices, cache, batch_size=2)

    for layer_idx in (0, 2):
        storage = pool.layers[layer_idx]
        assert storage is not None and storage.conv_states is not None
        assert torch.all(storage.conv_states[3] == layer_idx + 3)
        assert torch.all(storage.conv_states[1] == layer_idx + 3)
        assert storage.replay_checkpoint_states is None
    pool.clear(1)
    for layer_idx in (0, 2):
        state = recurrent[layer_idx]
        storage = pool.layers[layer_idx]
        assert state is not None and storage is not None
        assert torch.count_nonzero(state[1]) == 0
        assert torch.count_nonzero(storage.conv_states[1]) == 0
        assert torch.count_nonzero(state[3]) > 0


def test_indexed_prefill_passes_authoritative_bf16_pool_to_combined_kernel():
    config = SimpleNamespace(
        hidden_size=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",),
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(torch.bfloat16)
    assert not hasattr(module, "packed_prefill_prepare")
    assert not hasattr(module, "packed_recurrent_prefill")
    module.supports_packed_gdn = lambda *_args: True
    module.causal_conv1d_packed = lambda *, x, final_state, **_kwargs: x
    workspace = SimpleNamespace(
        out=torch.empty((1, 3, 2, 2), dtype=torch.bfloat16),
        can_serve=lambda *_args, **_kwargs: True,
    )
    module.allocate_packed_gdn_prefill_workspace = (
        lambda *_args, **_kwargs: workspace
    )
    captured = {}

    def combined(mixed_qkv, _a, _b, _A_log, _dt_bias, _cu, **kwargs):
        state = kwargs["final_state"]
        indices = kwargs["final_state_indices"]
        captured.update(
            mixed_qkv=mixed_qkv,
            state=state,
            indices=indices,
            workspace=kwargs["workspace"],
        )
        state[indices[0]].fill_(1)
        state[indices[1]].fill_(2)
        return torch.zeros_like(workspace.out), state

    module.packed_gated_delta_rule_prefill = combined
    pool = _state_pool(config)
    recurrent_states = pool.recurrent_tensors_for_form(_generated_form())
    cache = Qwen35InferenceCache(
        config=config,
        paged_kv=(None,),
        replay_capacity=2,
    )
    pool.bind_generated_prefill_state(cache)
    indices = torch.tensor([3, 1], dtype=torch.long)

    module(
        torch.zeros((1, 3, 4), dtype=torch.bfloat16),
        cache_params=cache,
        cu_seq_lens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        gdn_state_indices=indices,
    )

    state = recurrent_states[0]
    assert state is not None
    assert captured["state"] is state
    assert torch.equal(captured["indices"], indices)
    assert captured["workspace"] is workspace
    assert torch.all(state[3] == 1)
    assert torch.all(state[1] == 2)
    assert torch.count_nonzero(state[0]) == 0
    assert cache.layers[0].replay_checkpoint_states is None


def test_packed_gdn_prefill_workspace_cache_reuses_capacity():
    mixed_qkv = torch.empty((1, 3, 8), dtype=torch.bfloat16)
    a = torch.empty((1, 3, 2), dtype=torch.bfloat16)
    contract_checks = []

    def can_serve(candidate, *_args, **_kwargs):
        contract_checks.append(None)
        return candidate.shape[1] <= 4

    workspace = SimpleNamespace(
        can_serve=can_serve,
    )
    allocations = []

    def allocate(*_args, **_kwargs):
        allocations.append(None)
        return workspace

    cache = _PackedGatedDeltaPrefillWorkspaceCache()
    assert cache.get(mixed_qkv, a, head_dim=2, allocate=allocate) is workspace
    assert cache.get(mixed_qkv, a, head_dim=2, allocate=allocate) is workspace
    assert len(allocations) == 1
    assert contract_checks == []

    shorter_mixed_qkv = mixed_qkv[:, :2]
    shorter_a = a[:, :2]
    assert (
        cache.get(shorter_mixed_qkv, shorter_a, head_dim=2, allocate=allocate)
        is workspace
    )
    assert len(allocations) == 1
    assert len(contract_checks) == 1


def test_generated_decode_binds_rope_offsets_without_dropping_old_bundle_prep(
    monkeypatch,
):
    rope_deltas = object()
    rope_inv_freq = object()
    page_table = object()
    runtime = SimpleNamespace(
        max_batch_size=4,
        _decode_rope_deltas=rope_deltas,
        _gather_decode_rope_deltas=lambda *_args: None,
        _prepare_decode_position_ids=lambda *_args: None,
        _paged_kv=(),
        _linear_state_pool=object(),
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    rotary_emb=SimpleNamespace(inv_freq=rope_inv_freq)
                )
            )
        ),
        page_table=SimpleNamespace(page_table=page_table),
    )
    captured = {}

    def capture(_cls, bound_runtime, spec, *, required_batch_sizes=()):
        assert bound_runtime is runtime
        captured["spec"] = spec
        captured["required_batch_sizes"] = tuple(required_batch_sizes)
        return object()

    monkeypatch.setattr(
        qwen_generated.GeneratedDecode,
        "try_create",
        classmethod(capture),
    )

    result = qwen_generated.create_generated_decode(runtime)

    assert result is not None
    assert captured["required_batch_sizes"] == (1, 2, 3, 4)
    spec = captured["spec"]
    inputs = spec.bindings.runtime_inputs(runtime)
    assert inputs["page_table"] is page_table
    assert inputs["rope_delta_table"] is rope_deltas
    assert inputs["rope_inv_freq"] is rope_inv_freq
    # Old bundles still require the materialized position tensor.  New bundles
    # omit it from their ABI, so the generic preparation planner selects no
    # callbacks while this compatibility declaration remains inert.
    assert spec.not_ready_inputs == frozenset({"position_ids"})
    assert tuple(step.name for step in spec.preparations) == (
        "gather_rope_deltas",
        "prepare_position_ids",
    )

    def descriptor(*logical_names):
        return {
            "device_program": {
                "argument_plan": {
                    "arguments": [
                        {"name": name, "source": "external"}
                        for name in logical_names
                    ],
                },
                "physical_abi": {
                    "operands": [
                        {
                            "abi_name": name,
                            "logical_name": name,
                            "owner": "engine",
                        }
                        for name in logical_names
                    ],
                },
            },
        }

    new_plan = runtime_generated._preparation_plan(
        descriptor("rope_delta_table"),
        ready=set(inputs),
        preparations=spec.preparations,
    )
    assert new_plan == ()

    old_ready = set(inputs) | {"batch_idx", "input_pos", "position_ids"}
    old_ready -= spec.not_ready_inputs
    old_plan = runtime_generated._preparation_plan(
        descriptor("position_ids"),
        ready=old_ready,
        preparations=spec.preparations,
    )
    assert tuple(step.name for step in old_plan) == (
        "gather_rope_deltas",
        "prepare_position_ids",
    )


def test_generated_decode_reports_compiled_slot_capacity(monkeypatch):
    runtime = SimpleNamespace(
        max_batch_size=1,
        _decode_rope_deltas=object(),
        _gather_decode_rope_deltas=lambda *_args: None,
        _prepare_decode_position_ids=lambda *_args: None,
        _paged_kv=(),
        _linear_state_pool=object(),
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    rotary_emb=SimpleNamespace(inv_freq=object())
                )
            )
        ),
        page_table=SimpleNamespace(page_table=object()),
    )
    captured = {}

    def resolve(_cls, bound_runtime, spec, *, required_batch_sizes=()):
        captured["runtime"] = bound_runtime
        captured["spec"] = spec
        captured["required_batch_sizes"] = tuple(required_batch_sizes)
        return 8

    monkeypatch.setattr(
        qwen_generated.GeneratedDecode,
        "resolve_slot_capacity",
        classmethod(resolve),
    )

    assert qwen_generated.generated_decode_slot_capacity(runtime) == 8
    assert captured["runtime"] is runtime
    assert captured["required_batch_sizes"] == (1,)
    assert captured["spec"].label == "Qwen"


def test_generated_capacity_inputs_resolve_state_after_cached_field_snapshot(
    monkeypatch,
):
    config = _linear_config()
    pool = _state_pool(config)
    runtime = SimpleNamespace(
        max_batch_size=4,
        _decode_rope_deltas=object(),
        _gather_decode_rope_deltas=lambda *_args: None,
        _prepare_decode_position_ids=lambda *_args: None,
        _paged_kv=(),
        _linear_state_pool=pool,
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    rotary_emb=SimpleNamespace(inv_freq=object())
                )
            )
        ),
        page_table=SimpleNamespace(page_table=object()),
    )
    captured = {}

    def capture(_cls, _runtime, spec, *, required_batch_sizes=()):
        captured["spec"] = spec
        captured["required_batch_sizes"] = tuple(required_batch_sizes)
        return object()

    monkeypatch.setattr(
        qwen_generated.GeneratedDecode, "try_create", classmethod(capture))
    qwen_generated.create_generated_decode(runtime)
    assert captured["required_batch_sizes"] == (1, 2, 3, 4)
    requirement = StateRepresentationRequirement(
        "gdn_recurrent_state",
        "materialized",
        ("state_row", "value_head", "value", "key"),
        "bf16",
    )

    first = captured["spec"].capacity_inputs(4, (requirement,))
    second = captured["spec"].capacity_inputs(4, (requirement,))

    recurrent = first["gdn_recurrent_state"]
    assert second["gdn_recurrent_state"] is recurrent
    assert recurrent[1] is None
    for layer_idx in (0, 2):
        state = recurrent[layer_idx]
        assert state is pool.layers[layer_idx].recurrent_states
        assert state.dtype == torch.bfloat16
        assert state.shape == (4, 2, 3, 2)
