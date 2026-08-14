from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(
        qwen_generated,
        "create_generated_decode",
        lambda *_args, **_kwargs: pytest.fail(
            "native mode must not construct generated decode"
        ),
    )

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is None
    assert runtime._decode_state_coordinator is None


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
            capacity=4,
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
            capacity=4,
        )


def test_qwen_required_generated_never_falls_back_to_native() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "generated"
    runtime.generated_decode = _Generated(supported=False)
    runtime._decode_state_coordinator = None
    runtime._decode_graphs = SimpleNamespace(
        run=lambda *_args: pytest.fail("generated mode must not run native decode")
    )

    with pytest.raises(
        RuntimeError,
        match="required generated Qwen decode does not cover active batch size 3",
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
    runtime._decode_state_coordinator = None
    runtime._decode_graphs = SimpleNamespace(
        run=lambda *_args: pytest.fail("generated mode must not run native decode")
    )

    with pytest.raises(RuntimeError, match="generated launch failed"):
        runtime.decode_with_slot(object(), 2)


def test_qwen_auto_preserves_native_fallback() -> None:
    native_batches: list[int] = []
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "auto"
    runtime.generated_decode = _Generated(supported=False)
    runtime._decode_state_coordinator = None
    runtime._decode_graphs = SimpleNamespace(
        run=lambda _slot, batch_size: native_batches.append(batch_size)
    )

    runtime.decode_with_slot(object(), 3)

    assert native_batches == [3]


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


def test_moondream_native_does_not_construct_generated_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.moondream import generated_decode as md_generated

    runtime = object.__new__(MoondreamRuntime)
    runtime.decode_path = "native"
    monkeypatch.setattr(
        md_generated,
        "create_generated_decode",
        lambda *_args, **_kwargs: pytest.fail(
            "native mode must not construct generated decode"
        ),
    )

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is None


def test_moondream_required_generated_unavailable_refuses_before_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.moondream import generated_decode as md_generated

    runtime = object.__new__(MoondreamRuntime)
    runtime.decode_path = "generated"
    runtime._decode_graphs = SimpleNamespace(
        ensure_ready=lambda _slots: pytest.fail(
            "required generated failure must precede graph capture"
        )
    )
    calls: list[bool] = []

    def unavailable(_runtime, *, required: bool = False):
        calls.append(required)
        raise RuntimeError("no compatible generated program")

    monkeypatch.setattr(md_generated, "create_generated_decode", unavailable)

    with pytest.raises(RuntimeError, match="no compatible generated program"):
        runtime._initialize_generated_decode()

    assert calls == [True]
    assert runtime.generated_decode is None
