import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kestrel.runtime.compilation import (
    ScalarBufferCanonicalization,
    canonicalize_immutable_scalar_buffers,
    materialize_dynamic_batch_domain,
)
from kestrel.runtime.bounded_projection import (
    PackedBoundedProjections,
    apply_sibling_bounded_projections,
    bind_declared_packed_bounded_projections,
)


class _ClippedLinear(nn.Module):
    def __init__(self, lower: float, upper: float) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 5, bias=False)
        self.register_buffer("lower", torch.tensor(lower))
        self.register_buffer("upper", torch.tensor(upper))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.clamp(value, self.lower, self.upper))


class _RepeatedBounds(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ClippedLinear(-2.0, 2.0) for _ in range(count)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack([layer(value) for layer in self.layers])


class _BoundedProjection(nn.Module):
    def __init__(
        self,
        lower: float,
        upper: float,
        output_lower: float,
        output_upper: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 5, bias=False)
        self.register_buffer("input_min", torch.tensor(lower))
        self.register_buffer("input_max", torch.tensor(upper))
        self.register_buffer("output_min", torch.tensor(output_lower))
        self.register_buffer("output_max", torch.tensor(output_upper))

    def forward_bounded_input(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.linear(value),
            self.output_min,
            self.output_max,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.forward_bounded_input(
            torch.clamp(value, self.input_min, self.input_max)
        )


class _SiblingBoundedProjections(nn.Module):
    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                _BoundedProjection(lower, upper, -3.0 - index, 3.0 + index)
                for index, (lower, upper) in enumerate(bounds)
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            apply_sibling_bounded_projections(value, self.projections)
        )


class _PackedBoundedProjectionModel(nn.Module):
    def __init__(self, output_sizes: tuple[int, ...] = (2, 2, 2)) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.packed = PackedBoundedProjections(
            2,
            output_sizes,
            source_names=("query", "key", "value"),
        )


def _packed_projection_sources(
    output_sizes: tuple[int, ...] = (2, 2, 2),
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for index, (name, output_size) in enumerate(
        zip(("query", "key", "value"), output_sizes)
    ):
        prefix = f"attention.{name}."
        state_dict[prefix + "linear.weight"] = (
            torch.arange(output_size * 2, dtype=torch.float32)
            .reshape(output_size, 2)
            .add_(index * 10)
        )
        state_dict[prefix + "input_min"] = torch.tensor(-2.0)
        state_dict[prefix + "input_max"] = torch.tensor(2.0)
        state_dict[prefix + "output_min"] = torch.tensor(-4.0 - index)
        state_dict[prefix + "output_max"] = torch.tensor(4.0 + index)
    return state_dict


class _DynamicBatchCallable:
    def __init__(self) -> None:
        self.compiled_regimes: set[str] = set()
        self.compilation_count = 0
        self.calls: list[int] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        batch_size = int(value.shape[0])
        regime = "one" if batch_size == 1 else "symbolic_positive"
        if regime not in self.compiled_regimes:
            self.compiled_regimes.add(regime)
            self.compilation_count += 1
        self.calls.append(batch_size)
        return value + 1


def test_materialize_dynamic_batch_domain_covers_public_capacity() -> None:
    compiled = _DynamicBatchCallable()
    synchronize_calls = 0

    def synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1

    materialize_dynamic_batch_domain(
        compiled,
        max_batch_size=8,
        inputs_for_batch=lambda batch: (torch.zeros(batch, 3),),
        synchronize=synchronize,
    )
    compilations_before_public_call = compiled.compilation_count

    compiled(torch.zeros(8, 3))

    assert compiled.calls == [1, 8, 8]
    assert compiled.compilation_count == compilations_before_public_call == 2
    assert synchronize_calls == 1


def test_materialize_dynamic_batch_domain_runs_singleton_once() -> None:
    compiled = _DynamicBatchCallable()

    materialize_dynamic_batch_domain(
        compiled,
        max_batch_size=1,
        inputs_for_batch=lambda batch: (torch.zeros(batch, 3),),
    )
    assert compiled.calls == [1]


def test_materialize_dynamic_batch_domain_rejects_empty_domain() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be positive"):
        materialize_dynamic_batch_domain(
            _DynamicBatchCallable(),
            max_batch_size=0,
            inputs_for_batch=lambda batch: (torch.zeros(batch, 3),),
        )


def test_canonicalize_immutable_scalar_buffers_preserves_exact_output() -> None:
    torch.manual_seed(9)
    module = _RepeatedBounds(6).eval()
    value = torch.randn(11, 7)
    expected = module(value).clone()

    result = canonicalize_immutable_scalar_buffers(module)

    assert result == ScalarBufferCanonicalization(candidates=12, aliases=10)
    assert module.layers[0].lower is module.layers[-1].lower
    assert module.layers[0].upper is module.layers[-1].upper
    assert torch.equal(module(value), expected)
    assert canonicalize_immutable_scalar_buffers(module) == (
        ScalarBufferCanonicalization(candidates=12, aliases=0)
    )


def test_sibling_bounded_projections_share_exact_input_transform() -> None:
    torch.manual_seed(11)
    module = _SiblingBoundedProjections([(-2.0, 2.0)] * 3).eval()
    value = torch.randn(13, 7)
    expected = torch.stack([projection(value) for projection in module.projections])

    canonicalize_immutable_scalar_buffers(module)
    actual = module(value)

    assert torch.equal(actual, expected)
    assert module.projections[0].input_min is module.projections[-1].input_min
    assert module.projections[0].input_max is module.projections[-1].input_max


def test_sibling_bounded_projections_preserve_heterogeneous_bounds() -> None:
    torch.manual_seed(12)
    module = _SiblingBoundedProjections(
        [(-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0)]
    ).eval()
    value = torch.randn(13, 7)
    expected = torch.stack([projection(value) for projection in module.projections])

    canonicalize_immutable_scalar_buffers(module)

    assert torch.equal(module(value), expected)


def test_sibling_bounded_projection_compile_graph_has_one_input_clamp() -> None:
    module = _SiblingBoundedProjections([(-2.0, 2.0)] * 3).eval()
    canonicalize_immutable_scalar_buffers(module)
    graphs: list[torch.fx.GraphModule] = []

    def capture_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(module, backend=capture_backend, fullgraph=True)
    value = torch.randn(13, 7)
    expected = module(value)

    assert torch.equal(compiled(value), expected)
    assert torch.equal(compiled(value), expected)
    assert len(graphs) == 1
    clamp_nodes = [
        node
        for node in graphs[0].graph.nodes
        if node.op == "call_function" and node.target is torch.clamp
    ]
    assert len(clamp_nodes) == 4


def test_sibling_bounded_projection_does_not_compare_uncanonicalized_values() -> None:
    module = _SiblingBoundedProjections([(-2.0, 2.0)] * 3).eval()
    assert module.projections[0].input_min is not module.projections[-1].input_min
    assert module.projections[0].input_max is not module.projections[-1].input_max
    graphs: list[torch.fx.GraphModule] = []

    def capture_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(module, backend=capture_backend, fullgraph=True)
    value = torch.randn(13, 7)

    assert torch.equal(compiled(value), module(value))
    assert len(graphs) == 1
    clamp_nodes = [
        node
        for node in graphs[0].graph.nodes
        if node.op == "call_function" and node.target is torch.clamp
    ]
    assert len(clamp_nodes) == 6


def test_packed_bounded_projection_binding_owns_one_parameter() -> None:
    output_sizes = (2, 3, 4)
    module = _PackedBoundedProjectionModel(output_sizes).eval()
    state_dict = _packed_projection_sources(output_sizes)
    source_weights = [
        state_dict[f"attention.{name}.linear.weight"].clone()
        for name in ("query", "key", "value")
    ]
    source_parameter_bytes = sum(
        weight.numel() * weight.element_size()
        for weight in source_weights
    )

    stats = bind_declared_packed_bounded_projections(module, state_dict)

    packed_key = "attention.packed.linear.weight"
    assert stats.groups == 1
    assert stats.source_parameter_bytes == source_parameter_bytes
    assert stats.packed_parameter_bytes == source_parameter_bytes
    assert stats.source_bound_bytes == 12 * torch.tensor(0.0).element_size()
    assert stats.packed_bound_bytes == (
        2 + 2 * sum(output_sizes)
    ) * torch.tensor(0.0).element_size()
    assert all(
        not key.startswith(
            ("attention.query.", "attention.key.", "attention.value.")
        )
        for key in state_dict
    )
    torch.testing.assert_close(
        state_dict[packed_key],
        torch.cat(source_weights, dim=0),
    )
    assert state_dict[packed_key].is_contiguous()
    assert torch.equal(
        state_dict["attention.packed.output_min"],
        torch.tensor([-4.0] * 2 + [-5.0] * 3 + [-6.0] * 4),
    )
    assert torch.equal(
        state_dict["attention.packed.output_max"],
        torch.tensor([4.0] * 2 + [5.0] * 3 + [6.0] * 4),
    )

    module.load_state_dict(state_dict, strict=False)
    parameter_names = [name for name, _ in module.named_parameters()]
    assert parameter_names == ["attention.packed.linear.weight"]
    assert module.attention.packed.linear.weight.numel() == sum(
        weight.numel() for weight in source_weights
    )

    outputs = module.attention.packed(torch.ones(3, 2))
    expected = torch.clamp(
        F.linear(
            torch.ones(3, 2).clamp(-2.0, 2.0),
            torch.cat(source_weights, dim=0),
        ),
        state_dict["attention.packed.output_min"],
        state_dict["attention.packed.output_max"],
    )
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected)
    storage_pointers = {
        output.untyped_storage().data_ptr()
        for output in outputs
    }
    assert len(storage_pointers) == 1


def test_packed_bounded_projection_binding_without_bounds() -> None:
    module = nn.Module()
    module.packed = PackedBoundedProjections(
        2,
        (2, 2, 2),
        source_names=("query", "key", "value"),
        source_weight_leaf="weight",
        use_bounds=False,
    )
    state_dict = {
        f"{name}.weight": torch.full((2, 2), float(index))
        for index, name in enumerate(("query", "key", "value"), start=1)
    }

    stats = bind_declared_packed_bounded_projections(module, state_dict)

    assert stats.groups == 1
    assert stats.source_bound_bytes == 0
    assert stats.packed_bound_bytes == 0
    assert set(state_dict) == {"packed.linear.weight"}
    module.load_state_dict(state_dict)
    outputs = module.packed(torch.ones(1, 2))
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        torch.tensor([[2.0, 2.0, 4.0, 4.0, 6.0, 6.0]]),
    )


def test_packed_bounded_projection_refuses_different_input_bounds() -> None:
    module = _PackedBoundedProjectionModel().eval()
    state_dict = _packed_projection_sources()
    state_dict["attention.key.input_max"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match="different input_max"):
        bind_declared_packed_bounded_projections(module, state_dict)


def test_packed_bounded_projection_compares_input_bounds_bitwise() -> None:
    module = _PackedBoundedProjectionModel().eval()
    state_dict = _packed_projection_sources()
    state_dict["attention.query.input_min"] = torch.tensor(-0.0)
    state_dict["attention.key.input_min"] = torch.tensor(0.0)
    state_dict["attention.value.input_min"] = torch.tensor(-0.0)

    with pytest.raises(ValueError, match="different input_min"):
        bind_declared_packed_bounded_projections(module, state_dict)


def test_packed_bounded_projection_refuses_noncontiguous_weights() -> None:
    module = _PackedBoundedProjectionModel().eval()
    state_dict = _packed_projection_sources()
    state_dict["attention.key.linear.weight"] = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
    )
    assert not state_dict["attention.key.linear.weight"].is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        bind_declared_packed_bounded_projections(module, state_dict)


@pytest.mark.parametrize("seed", range(8))
def test_canonicalize_random_scalar_trees_by_exact_value(seed: int) -> None:
    rng = random.Random(seed)
    values = [rng.choice((-3.0, -0.0, 0.0, 1.5)) for _ in range(24)]
    module = nn.Module()
    module.children_by_value = nn.ModuleList()
    for index, value in enumerate(values):
        child = nn.Module()
        child.register_buffer(f"value_{index}", torch.tensor(value))
        module.children_by_value.append(child)
    module.eval()

    canonicalize_immutable_scalar_buffers(module)

    buffers = list(module.buffers())
    for lhs_index, lhs in enumerate(buffers):
        for rhs_index, rhs in enumerate(buffers):
            same_bits = torch.equal(
                lhs.reshape(1).view(torch.uint8), rhs.reshape(1).view(torch.uint8)
            )
            assert (lhs is rhs) == same_bits, (lhs_index, rhs_index)


def test_canonicalization_keeps_dtype_signed_zero_and_vectors_distinct() -> None:
    module = nn.Module()
    module.register_buffer("positive_zero", torch.tensor(0.0, dtype=torch.float32))
    module.register_buffer("negative_zero", torch.tensor(-0.0, dtype=torch.float32))
    module.register_buffer("half_zero", torch.tensor(0.0, dtype=torch.float16))
    module.register_buffer("vector", torch.tensor([0.0], dtype=torch.float32))
    module.eval()

    result = canonicalize_immutable_scalar_buffers(module)

    assert result == ScalarBufferCanonicalization(candidates=3, aliases=0)
    assert module.positive_zero is not module.negative_zero
    assert module.positive_zero is not module.half_zero
    assert module.positive_zero is not module.vector


def test_canonicalization_preserves_nonpersistent_buffer_registration() -> None:
    module = nn.Module()
    module.register_buffer("persistent", torch.tensor(4.0))
    module.register_buffer("temporary", torch.tensor(4.0), persistent=False)
    module.eval()

    canonicalize_immutable_scalar_buffers(module)

    assert module.persistent is module.temporary
    assert "persistent" in module.state_dict()
    assert "temporary" not in module.state_dict()


def test_canonicalization_rejects_training_modules() -> None:
    module = _RepeatedBounds(2)

    with pytest.raises(ValueError, match="requires eval mode"):
        canonicalize_immutable_scalar_buffers(module)


def test_canonicalization_rejects_gradient_buffers() -> None:
    module = nn.Module()
    module.register_buffer("value", torch.tensor(1.0, requires_grad=True))
    module.eval()

    with pytest.raises(ValueError, match="must not require gradients"):
        canonicalize_immutable_scalar_buffers(module)
