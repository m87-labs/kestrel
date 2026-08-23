from types import SimpleNamespace

import pytest
import torch

from kestrel.models.gemma4.runtime import Gemma4Runtime
from kestrel.models.moondream.runtime import MoondreamRuntime
from kestrel.models.qwen35 import generated_decode as qwen_generated
from kestrel.models.qwen35.runtime import Qwen35Runtime
from kestrel.runtime.generated_decode import GeneratedDecode


class _Generated:
    state_requirements_by_capacity = {}
    state_buffers = ()

    def __init__(self, *, supported: bool) -> None:
        self.supported = supported
        self.runs: list[int] = []

    def supports(self, _batch_size: int) -> bool:
        return self.supported

    def run(self, _slot, batch_size: int) -> None:
        self.runs.append(batch_size)


def test_qwen_native_does_not_construct_generated_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "native"
    calls = []
    runtime._linear_state_pool = SimpleNamespace(
        initialize_native_recurrent=lambda: calls.append("native"))
    monkeypatch.setattr(
        qwen_generated,
        "create_generated_decode",
        lambda *_args, **_kwargs: pytest.fail(
            "native mode must not construct generated decode"
        ),
    )

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is None
    assert calls == ["native"]


def test_gemma_native_does_not_construct_generated_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.gemma4 import generated_decode as gemma_generated

    runtime = object.__new__(Gemma4Runtime)
    runtime.decode_path = "native"
    monkeypatch.setattr(
        gemma_generated,
        "create_generated_decode",
        lambda *_args, **_kwargs: pytest.fail(
            "native mode must not construct generated decode"
        ),
    )

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is None


def test_qwen_required_generated_unavailable_refuses_before_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "generated"
    graph_capture = SimpleNamespace(
        ensure_ready=lambda _slots: pytest.fail(
            "required generated failure must precede graph capture"
        )
    )
    runtime._decode_graphs = graph_capture
    calls: list[bool] = []

    def unavailable(_runtime, *, required: bool = False):
        calls.append(required)
        raise RuntimeError("no compatible generated program")

    monkeypatch.setattr(qwen_generated, "create_generated_decode", unavailable)

    with pytest.raises(RuntimeError, match="no compatible generated program"):
        runtime._initialize_generated_decode()

    assert calls == [True]
    assert runtime.generated_decode is None


def test_qwen_selected_generated_never_captures_native_decode_graphs() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime._use_cuda_graphs = True
    runtime._decode_slots = (object(),)
    runtime._decode_graphs = SimpleNamespace(
        ensure_ready=lambda _slots: pytest.fail(
            "generated state must not prepare native decode graphs"))

    def select_generated():
        runtime.generated_decode = _Generated(supported=True)

    runtime._initialize_generated_decode = select_generated
    runtime._initialize_decode_execution()


def test_qwen_native_selection_captures_decode_graphs() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime._use_cuda_graphs = True
    runtime._decode_slots = (object(),)
    captures = []
    runtime._decode_graphs = SimpleNamespace(
        ensure_ready=lambda slots: captures.append(slots))

    def select_native():
        runtime.generated_decode = None

    runtime._initialize_generated_decode = select_native
    runtime._initialize_decode_execution()

    assert captures == [runtime._decode_slots]


def test_generated_requirement_unavailable_refuses_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[object] = []
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: ()),
    )
    monkeypatch.setattr(
        GeneratedDecode,
        "__init__",
        lambda self, *_args, **_kwargs: constructions.append(self),
    )

    with pytest.raises(
        RuntimeError,
        match="Qwen requires a compatible bundled generated-decode program",
    ):
        GeneratedDecode.require(
            object(),
            SimpleNamespace(label="Qwen"),
            batch_sizes=range(1, 5),
        )

    assert constructions == []


def test_generated_requirement_covers_complete_batch_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(
            lambda _cls, _runtime, _spec: (SimpleNamespace(capacity=4),)
        ),
    )
    monkeypatch.setattr(
        GeneratedDecode,
        "supports",
        lambda _self, batch_size: batch_size != 3,
    )

    with pytest.raises(
        RuntimeError,
        match=r"does not cover active batch sizes \[3\]",
    ):
        GeneratedDecode.require(
            object(),
            SimpleNamespace(label="Qwen"),
            batch_sizes=range(1, 5),
        )


def test_qwen_required_generated_never_falls_back_to_native() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "generated"
    runtime.generated_decode = _Generated(supported=False)
    runtime._decode_graphs = SimpleNamespace(
        run=lambda *_args: pytest.fail("generated mode must not run native decode")
    )

    with pytest.raises(
        RuntimeError,
        match="selected generated Qwen decode does not cover active batch size 3",
    ):
        runtime.decode_with_slot(object(), 3)


def test_qwen_selected_generated_failure_never_falls_back_to_native() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "generated"
    generated = _Generated(supported=True)

    def fail_generated(_slot, _batch_size: int) -> None:
        raise RuntimeError("generated launch failed")

    generated.run = fail_generated
    runtime.generated_decode = generated
    runtime._decode_graphs = SimpleNamespace(
        run=lambda *_args: pytest.fail("generated mode must not run native decode")
    )

    with pytest.raises(RuntimeError, match="generated launch failed"):
        runtime.decode_with_slot(object(), 2)


def test_qwen_auto_selected_generated_state_refuses_native_fallback() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "auto"
    runtime.generated_decode = _Generated(supported=False)
    runtime._decode_graphs = SimpleNamespace(
        run=lambda *_args: pytest.fail("generated state must not run native decode")
    )

    with pytest.raises(
        RuntimeError,
        match="selected generated Qwen decode does not cover active batch size 3",
    ):
        runtime.decode_with_slot(object(), 3)


def test_qwen_auto_without_generated_program_selects_native_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "auto"
    calls = []
    runtime._linear_state_pool = SimpleNamespace(
        initialize_native_recurrent=lambda: calls.append("native"))
    monkeypatch.setattr(
        qwen_generated, "create_generated_decode", lambda *_args, **_kwargs: None)

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is None
    assert calls == ["native"]


def test_qwen_auto_probe_cannot_allocate_generated_then_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.qwen35.cache import Qwen35LinearStatePool
    from kestrel.runtime.carried_state import StatePhysicalForm

    config = SimpleNamespace(
        layer_types=("linear_attention",),
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=4,
    )
    pool = Qwen35LinearStatePool(
        config=config,
        max_batch_slots=4,
        device=torch.device("cpu"),
        replay_capacity=2,
    )
    pool.initialize_from_config(config, dtype=torch.bfloat16)
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "auto"
    runtime._linear_state_pool = pool

    def touches_generated_state(*_args, **_kwargs):
        pool.recurrent_tensors_for_form(StatePhysicalForm(
            "materialized",
            ("state_row", "value_head", "value", "key"),
            "bf16",
        ))
        return None

    monkeypatch.setattr(
        qwen_generated, "create_generated_decode", touches_generated_state)

    with pytest.raises(RuntimeError, match="cannot switch to native replay"):
        runtime._initialize_generated_decode()

    storage = pool.layers[0]
    assert storage is not None and storage.recurrent_states is not None
    assert storage.recurrent_states.dtype == torch.bfloat16
    assert storage.replay_checkpoint_states is None


def test_gemma_required_generated_never_falls_back_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.gemma4 import runtime as gemma_runtime

    runtime = object.__new__(Gemma4Runtime)
    runtime.decode_path = "generated"
    runtime.generated_decode = _Generated(supported=False)
    runtime._run_native_decode = lambda *_args: pytest.fail(
        "generated mode must not run native decode"
    )
    monkeypatch.setattr(
        gemma_runtime.torch.cuda,
        "stream",
        lambda _stream: pytest.fail(
            "uncovered generated batches must refuse before stream work"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="required generated Gemma decode does not cover active batch size 2",
    ):
        runtime.decode_with_slot(SimpleNamespace(compute_stream=object()), 2)


def test_moondream_explicit_decode_path_refuses_before_runtime_work() -> None:
    cfg = SimpleNamespace(
        decode_path="native",
        resolved_device=lambda: pytest.fail(
            "unsupported policy must fail before runtime work"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Moondream runtimes currently support only decode_path='auto'",
    ):
        MoondreamRuntime(
            cfg,
            kv_pool=object(),
            compute_stream=None,
        )
