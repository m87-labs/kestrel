from types import SimpleNamespace

from kestrel.models.qwen35 import generated_decode as qwen_generated
from kestrel.runtime import generated_decode as runtime_generated


def test_generated_decode_binds_rope_offsets_without_dropping_old_bundle_prep(
    monkeypatch,
):
    rope_deltas = object()
    rope_inv_freq = object()
    page_table = object()
    runtime = SimpleNamespace(
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

    def capture(_cls, bound_runtime, spec):
        assert bound_runtime is runtime
        captured["spec"] = spec
        return object()

    monkeypatch.setattr(
        qwen_generated.GeneratedDecode,
        "try_create",
        classmethod(capture),
    )

    result = qwen_generated.create_generated_decode(runtime)

    assert result is not None
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
