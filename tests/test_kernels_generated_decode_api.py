import inspect

from kestrel_kernels import generated_decode, get_runtime


def test_pinned_kernels_exposes_generated_decode_weight_binding_api() -> None:
    resolve_parameters = inspect.signature(
        generated_decode.resolve_compatible_programs
    ).parameters
    assert {"device_sms", "weight_sources"} <= set(resolve_parameters), (
        "the pinned kestrel-kernels wheel predates generated-decode device and "
        "logical-weight binding"
    )

    materialize_parameters = inspect.signature(
        generated_decode.materialize_weights
    ).parameters
    assert {"engine_buffers", "weight_sources"} <= set(materialize_parameters), (
        "the pinned kestrel-kernels wheel predates generated-decode engine and "
        "logical-weight materialization"
    )


def test_pinned_kernels_exposes_packed_prefill_topology_factory() -> None:
    gated_delta = get_runtime().gated_delta
    assert callable(getattr(gated_delta, "bind_packed_prefill_topology", None)), (
        "the pinned kestrel-kernels wheel predates authoritative packed-prefill "
        "topology binding"
    )
