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


def test_qwen_native_path_is_rejected_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models import registry

    monkeypatch.setattr(registry, "get_spec", lambda _model: object())
    cfg = SimpleNamespace(
        model="qwen-test",
        decode_path="native",
        resolved_device=lambda: torch.device("cpu"),
        resolved_dtype=lambda: torch.bfloat16,
    )

    with pytest.raises(
        ValueError,
        match="requires generated decode; native decode was removed",
    ):
        Qwen35Runtime(cfg, kv_pool=object())


def test_qwen_initialization_requires_generated_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.decode_path = "auto"
    generated = _Generated(supported=True)
    calls: list[bool] = []

    def create(_runtime, *, required: bool = False):
        calls.append(required)
        return generated

    monkeypatch.setattr(
        qwen_generated,
        "create_generated_decode",
        create,
    )

    runtime._initialize_generated_decode()

    assert runtime.generated_decode is generated
    assert calls == [True]


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

    with pytest.raises(RuntimeError, match="generated launch failed"):
        runtime.decode_with_slot(object(), 2)


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
