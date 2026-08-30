from __future__ import annotations

import asyncio
import hashlib
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Callable

import numpy as np

import pytest

from kestrel.engine import (
    EngineMetrics,
    EngineResult,
    EngineStream,
    InferenceEngine,
    _AdmissionCoordinator,
    _AutoregressiveRequest,
)
from kestrel.models.moondream.runtime import TextToken
from kestrel.runtime import ExecutionShape
from kestrel.scheduler import GeneratedPrefix, StreamUpdate
from kestrel.skills import (
    DecodeStep,
    PreparedSkillPrompt,
    SkillFinalizeResult,
    SkillRegistry,
    SkillSpec,
    SkillState,
)


def test_default_prepared_skill_prompt_preserves_tokens_context_and_budget() -> None:
    context = object()
    skill = _PrefixSkill()

    prepared = skill.prepare_prompt(object(), context, 7)

    assert prepared == PreparedSkillPrompt(
        request_context=context,
        tokens=(),
        max_new_tokens=7,
    )


def test_engine_stream_close_stops_a_waiter_and_settles() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EngineResult] = loop.create_future()
        cancelled = threading.Event()
        queue = asyncio.Queue()
        stream = EngineStream(1, queue, future, cancel=cancelled.set)
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        assert cancelled.is_set()
        queue.put_nowait(StreamUpdate(1, TextToken(2), "x", 0))
        result = EngineResult(
            request_id=1,
            tokens=[],
            finish_reason="cancelled",
            metrics=EngineMetrics(0, 0, 0.0, 0.0, 0.0),
            output={},
        )
        future.set_result(result)

        await closing
        with pytest.raises(StopAsyncIteration):
            await waiting
        assert await stream.result() is result

    asyncio.run(run())


def _make_request(
    *,
    request_id: int = 1,
    image: np.ndarray | bytes | None = None,
    encoder_input: object | None = None,
) -> _AutoregressiveRequest:
    return _AutoregressiveRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[object(), object()],
        image=image,
        image_hash=None,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        submitted_at=0.0,
        future=None,
        stream_queue=None,
        skill=SimpleNamespace(),
        request_context=object(),
        encoder_input=encoder_input,
        adapter=None,
    )


class _PrefixSkillState(SkillState):
    def __init__(
        self,
        spec: SkillSpec,
        request: object,
        stop_token_ids: tuple[int, ...],
    ) -> None:
        super().__init__(spec, request)
        self.positions: list[int] = []
        self._stop_token_ids = stop_token_ids

    def consume_step(self, runtime: object, step: DecodeStep) -> None:
        self.positions.append(step.position)
        self.append_token(step.token)

    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        return SkillFinalizeResult(text="", tokens=list(self.tokens), output={})

    def stop_token_ids(self, runtime: object) -> tuple[int, ...] | None:
        return self._stop_token_ids or None


class _PrefixSkill(SkillSpec):
    def __init__(self, stop_token_ids: tuple[int, ...] = ()) -> None:
        super().__init__(name="prefix")
        self._stop_token_ids = stop_token_ids

    def build_prompt_tokens(
        self, runtime: object, request_context: object
    ) -> list[object]:
        return []

    def create_state(
        self,
        runtime: object,
        request: object,
        request_context: object,
    ) -> _PrefixSkillState:
        return _PrefixSkillState(self, request, self._stop_token_ids)


class _FakeImagePreprocessor:
    """Stand-in for a runtime's image-preprocessing executor.

    The engine now dispatches image preprocessing through
    ``runtime.preprocess_image_async``; tests attach one of these to a
    ``_FakeRuntime`` to control the future returned per call.
    """

    def __init__(self, futures: list[Future[object]] | None = None) -> None:
        self.futures = futures or [Future()]
        self.submissions: list[object] = []

    def submit(self, value: object) -> Future[object]:
        self.submissions.append(value)
        return self.futures.pop(0)


class _FakeRuntime:
    def __init__(
        self,
        *,
        prefix_cache: object | None,
        prefix_hit: bool,
        image_preprocessor: _FakeImagePreprocessor | None = None,
        encoder_input_preprocessor: _FakeImagePreprocessor | None = None,
    ) -> None:
        self.prefix_cache = prefix_cache
        self.config = SimpleNamespace(vision=object())
        self._prefix_hit = prefix_hit
        self.cache_checks: list[tuple[list[object], bytes | None, str | None]] = []
        self._image_preprocessor = image_preprocessor or _FakeImagePreprocessor()
        self._encoder_input_preprocessor = (
            encoder_input_preprocessor or _FakeImagePreprocessor()
        )

    def check_prefix_cache(
        self, tokens_list: list[object], image_hash: bytes | None, adapter: str | None
    ) -> bool:
        self.cache_checks.append((tokens_list, image_hash, adapter))
        return self._prefix_hit

    def preprocess_image_async(self, image: np.ndarray | bytes) -> Future[object]:
        return self._image_preprocessor.submit(image)

    def preprocess_encoder_input_async(
        self, encoder_input: object
    ) -> Future[object]:
        return self._encoder_input_preprocessor.submit(encoder_input)

    def shutdown(self) -> None:  # pragma: no cover
        pass


def _record_failure(
    failures: list[tuple[_AutoregressiveRequest, BaseException]]
) -> Callable[[_AutoregressiveRequest, BaseException], None]:
    def _fail(req: _AutoregressiveRequest, exc: BaseException) -> None:
        failures.append((req, exc))

    return _fail


def test_admission_coordinator_immediately_admits_text_only_request() -> None:
    preprocessor = _FakeImagePreprocessor()
    runtime = _FakeRuntime(
        prefix_cache=None, prefix_hit=False, image_preprocessor=preprocessor,
    )
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        wake_event=threading.Event(),
        fail_request=_record_failure(failures),
    )

    ready = coordinator.submit(_make_request(image=None))

    assert ready is not None
    assert ready.crops is None
    assert ready.prefix_cache_hit is False
    assert not coordinator.has_pending()
    assert preprocessor.submissions == []
    assert failures == []


def test_admission_coordinator_prepares_encoder_input_and_drops_raw_payload() -> None:
    encoder_input = object()
    encoder_input_future: Future[object] = Future()
    encoder_input_preprocessor = _FakeImagePreprocessor([encoder_input_future])
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(
            prefix_cache=None,
            prefix_hit=False,
            encoder_input_preprocessor=encoder_input_preprocessor,
        ),
        wake_event=threading.Event(),
        fail_request=lambda *_: None,
    )
    req = _make_request(encoder_input=encoder_input)

    assert coordinator.submit(req) is None
    encoder_input_future.set_result("prepared-encoder-input")
    ready = coordinator.take_ready()

    assert ready is not None
    assert ready.encoder_input == "prepared-encoder-input"
    assert ready.req is req
    assert req.encoder_input is None
    assert encoder_input_preprocessor.submissions == [encoder_input]


def test_encoder_input_disables_image_prefix_reuse() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    encoder_input = object()
    crop_future: Future[object] = Future()
    encoder_input_future: Future[object] = Future()
    image_preprocessor = _FakeImagePreprocessor([crop_future])
    encoder_input_preprocessor = _FakeImagePreprocessor([encoder_input_future])
    runtime = _FakeRuntime(
        prefix_cache=object(),
        prefix_hit=True,
        image_preprocessor=image_preprocessor,
        encoder_input_preprocessor=encoder_input_preprocessor,
    )
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        wake_event=threading.Event(),
        fail_request=lambda *_: None,
    )

    assert coordinator.submit(
        _make_request(image=image, encoder_input=encoder_input)
    ) is None
    assert runtime.cache_checks == []
    assert image_preprocessor.submissions == [image]
    assert encoder_input_preprocessor.submissions == [encoder_input]

    crop_future.set_result("crops")
    assert coordinator.take_ready() is None
    encoder_input_future.set_result("prepared-encoder-input")
    ready = coordinator.take_ready()
    assert ready is not None
    assert ready.crops == "crops"
    assert ready.encoder_input == "prepared-encoder-input"


def test_encoder_input_failure_cancels_concurrent_image_preprocessing() -> None:
    crop_future: Future[object] = Future()
    encoder_input_future: Future[object] = Future()
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(
            prefix_cache=None,
            prefix_hit=False,
            image_preprocessor=_FakeImagePreprocessor([crop_future]),
            encoder_input_preprocessor=_FakeImagePreprocessor(
                [encoder_input_future]
            ),
        ),
        wake_event=threading.Event(),
        fail_request=_record_failure(failures),
    )
    req = _make_request(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        encoder_input=object(),
    )

    assert coordinator.submit(req) is None
    encoder_input_future.set_exception(ValueError("bad audio"))
    assert coordinator.take_ready() is None

    assert crop_future.cancelled()
    assert failures == [(req, encoder_input_future.exception())]
    assert not coordinator.has_pending()


def test_admission_close_cancels_pending_preprocessing() -> None:
    crop_future: Future[object] = Future()
    encoder_future: Future[object] = Future()
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(
            prefix_cache=None,
            prefix_hit=False,
            image_preprocessor=_FakeImagePreprocessor([crop_future]),
            encoder_input_preprocessor=_FakeImagePreprocessor([encoder_future]),
        ),
        wake_event=threading.Event(),
        fail_request=lambda *_: None,
    )
    req = _make_request(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        encoder_input=object(),
    )
    req.cancel_event = threading.Event()

    assert coordinator.submit(req) is None
    req.cancel_event.set()

    assert coordinator.pop_cancelled() == [req]
    assert crop_future.cancelled()
    assert encoder_future.cancelled()
    assert not coordinator.has_pending()


def test_admission_coordinator_skips_crop_work_on_prefix_hit() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessor = _FakeImagePreprocessor()
    runtime = _FakeRuntime(
        prefix_cache=object(), prefix_hit=True, image_preprocessor=preprocessor,
    )
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        wake_event=threading.Event(),
        fail_request=_record_failure(failures),
    )
    req = _make_request(image=image)

    ready = coordinator.submit(req)

    assert ready is not None
    assert ready.crops is None
    assert ready.prefix_cache_hit is True
    assert req.image_hash == hashlib.sha256(image.tobytes()).digest()
    assert len(runtime.cache_checks) == 1
    assert preprocessor.submissions == []
    assert failures == []


def test_admission_coordinator_checks_prefix_cache_with_generated_prefix() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    preprocessor = _FakeImagePreprocessor()
    runtime = _FakeRuntime(
        prefix_cache=object(), prefix_hit=True, image_preprocessor=preprocessor,
    )
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        wake_event=threading.Event(),
        fail_request=_record_failure(failures),
    )
    req = _make_request(image=image)
    req.prompt_tokens = [TextToken(1)]
    req.generated_prefix = GeneratedPrefix(tokens=(TextToken(10), TextToken(11)))

    ready = coordinator.submit(req)

    assert ready is not None
    assert len(runtime.cache_checks) == 1
    assert runtime.cache_checks[0][0] == [TextToken(1), TextToken(10), TextToken(11)]
    assert preprocessor.submissions == []
    assert failures == []


def test_admission_coordinator_promotes_completed_crops() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    crop_future: Future[object] = Future()
    preprocessor = _FakeImagePreprocessor([crop_future])
    wake_event = threading.Event()
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(
            prefix_cache=object(), prefix_hit=False,
            image_preprocessor=preprocessor,
        ),
        wake_event=wake_event,
        fail_request=_record_failure(failures),
    )
    req = _make_request(image=image)

    ready = coordinator.submit(req)
    crop_future.set_result("crops")

    assert ready is None
    assert coordinator.has_pending()
    assert wake_event.is_set()

    promoted = coordinator.take_ready()

    assert promoted is not None
    assert promoted.req is req
    assert promoted.crops == "crops"
    assert promoted.prefix_cache_hit is False
    assert not coordinator.has_pending()
    assert len(preprocessor.submissions) == 1
    assert failures == []


def test_admission_coordinator_skips_failed_crop_and_keeps_promoting() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    failed_future: Future[object] = Future()
    ready_future: Future[object] = Future()
    preprocessor = _FakeImagePreprocessor([failed_future, ready_future])
    failures: list[tuple[_AutoregressiveRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(
            prefix_cache=None, prefix_hit=False,
            image_preprocessor=preprocessor,
        ),
        wake_event=threading.Event(),
        fail_request=_record_failure(failures),
    )
    failed_req = _make_request(request_id=1, image=image)
    ready_req = _make_request(request_id=2, image=image)

    coordinator.submit(failed_req)
    coordinator.submit(ready_req)
    failed_future.set_exception(RuntimeError("crop failed"))
    ready_future.set_result("crops")

    promoted = coordinator.take_ready()

    assert promoted is not None
    assert promoted.req is ready_req
    assert promoted.crops == "crops"
    assert len(failures) == 1
    assert failures[0][0] is failed_req
    assert str(failures[0][1]) == "crop failed"


def test_transcribe_only_runtime_skips_legacy_query_warmup() -> None:
    engine = object.__new__(InferenceEngine)
    engine._default_model = "fake"
    engine._runtimes = {
        "fake": SimpleNamespace(execution_shape=ExecutionShape.AUTOREGRESSIVE),
    }
    engine._skills_override = SkillRegistry([SkillSpec("transcribe")])

    async def query(**_kwargs: object) -> None:
        raise AssertionError("query warmup must not run")

    engine.query = query  # type: ignore[method-assign]
    asyncio.run(engine._warmup_query_pipeline())


def test_extract_private_logprobs_setting() -> None:
    engine = object.__new__(InferenceEngine)

    assert engine._extract_logprobs(None) is None
    assert engine._extract_logprobs({}) is None
    assert engine._extract_logprobs({"_logprobs": None}) is None
    assert engine._extract_logprobs({"_logprobs": True}) is True
    assert engine._extract_logprobs({"_logprobs": False}) is False
    with pytest.raises(TypeError, match="settings._logprobs"):
        engine._extract_logprobs({"_logprobs": 1})


def test_extract_private_generated_prefix_setting() -> None:
    engine = object.__new__(InferenceEngine)

    assert engine._extract_generated_prefix(None) == GeneratedPrefix()
    assert engine._extract_generated_prefix({}) == GeneratedPrefix()

    prefix = engine._extract_generated_prefix(
        {
            "_generated_prefix": {
                "tokens": [TextToken(10), TextToken(11)],
                "logprobs": [-1, -2.5],
            }
        }
    )

    assert prefix.tokens == (TextToken(10), TextToken(11))
    assert prefix.logprobs == (-1.0, -2.5)

    prefix = engine._normalize_generated_prefix(
        GeneratedPrefix(tokens=(TextToken(10),), logprobs=("-1",)),
        "_generated_prefix",
    )
    assert prefix.tokens == (TextToken(10),)
    assert prefix.logprobs == (-1.0,)

    with pytest.raises(TypeError, match="settings._generated_prefix"):
        engine._extract_generated_prefix({"_generated_prefix": []})
    with pytest.raises(TypeError, match="unsupported key"):
        engine._extract_generated_prefix(
            {"_generated_prefix": {"tokens": [], "text": "answer"}}
        )
    with pytest.raises(ValueError, match="tokens is required"):
        engine._extract_generated_prefix({"_generated_prefix": {}})
    with pytest.raises(TypeError, match="generated Token"):
        engine._extract_generated_prefix({"_generated_prefix": {"tokens": [10]}})
    with pytest.raises(TypeError, match="generated Token"):
        engine._normalize_generated_prefix(
            GeneratedPrefix(tokens=(10,)),
            "_generated_prefix",
        )
    with pytest.raises(ValueError, match="same length"):
        engine._extract_generated_prefix(
            {"_generated_prefix": {"tokens": [TextToken(10)], "logprobs": []}}
        )


def test_validate_generated_prefix_rejects_unsupported_requests() -> None:
    engine = object.__new__(InferenceEngine)
    engine._default_model = "fake"
    engine._runtimes = {
        "fake": SimpleNamespace(
            eos_token_ids=(2, 3),
            execution_shape=ExecutionShape.AUTOREGRESSIVE,
        ),
    }

    prefix = GeneratedPrefix(tokens=(TextToken(10),))

    with pytest.raises(ValueError, match="streaming"):
        engine._validate_generated_prefix_for_request(
            prefix,
            max_new_tokens=4,
            return_logprobs=None,
            streaming=True,
        )
    with pytest.raises(ValueError, match="shorter"):
        engine._validate_generated_prefix_for_request(
            prefix,
            max_new_tokens=1,
            return_logprobs=None,
            streaming=False,
        )
    with pytest.raises(ValueError, match="logprobs is required"):
        engine._validate_generated_prefix_for_request(
            prefix,
            max_new_tokens=4,
            return_logprobs=True,
            streaming=False,
        )
    with pytest.raises(ValueError, match="must not contain EOS"):
        engine._validate_generated_prefix_for_request(
            GeneratedPrefix(tokens=(TextToken(3),), logprobs=(-0.1,)),
            max_new_tokens=4,
            return_logprobs=True,
            streaming=False,
        )


def test_build_generation_request_consumes_generated_prefix() -> None:
    engine = object.__new__(InferenceEngine)
    req = _make_request()
    req.prompt_tokens = [TextToken(1)]
    req.max_new_tokens = 4
    req.skill = _PrefixSkill()
    req.generated_prefix = GeneratedPrefix(
        tokens=(TextToken(10), TextToken(11)),
        logprobs=(-0.1, -0.2),
    )
    runtime = SimpleNamespace(
        max_seq_length=32, image_prefix_length=0, spec=None
    )

    generation_req, skill_state = engine._build_generation_request(runtime, req, None)

    assert generation_req.prompt_tokens == [TextToken(1)]
    assert generation_req.prefill_tokens == [TextToken(1), TextToken(10), TextToken(11)]
    assert generation_req.remaining_new_tokens == 2
    assert list(skill_state.tokens) == [TextToken(10), TextToken(11)]
    assert skill_state.positions == [0, 1]


def test_build_generation_request_rejects_skill_stop_in_generated_prefix() -> None:
    engine = object.__new__(InferenceEngine)
    req = _make_request()
    req.prompt_tokens = [TextToken(1)]
    req.max_new_tokens = 4
    req.skill = _PrefixSkill(stop_token_ids=(11,))
    req.generated_prefix = GeneratedPrefix(tokens=(TextToken(10), TextToken(11)))
    runtime = SimpleNamespace(
        max_seq_length=32, image_prefix_length=0, spec=None
    )

    with pytest.raises(ValueError, match="must not contain stop tokens"):
        engine._build_generation_request(runtime, req, None)


def test_build_generation_request_uses_runtime_image_kv_length() -> None:
    engine = object.__new__(InferenceEngine)
    req = _make_request(image=b"image")
    req.skill = _PrefixSkill()
    runtime = SimpleNamespace(
        max_seq_length=32,
        spec=None,
        image_kv_length=lambda prompt_tokens, image, image_crops: 5,
    )

    generation_req, _ = engine._build_generation_request(runtime, req, object())

    assert generation_req.image_length == 5
    assert generation_req.target_length == 15


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_build_generation_request_rejects_invalid_image_kv_length(invalid) -> None:
    engine = object.__new__(InferenceEngine)
    req = _make_request(image=b"image")
    runtime = SimpleNamespace(
        spec=None,
        image_kv_length=lambda prompt_tokens, image, image_crops: invalid,
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        engine._build_generation_request(runtime, req, object())


def test_build_generation_request_rejects_encoder_input_with_spec_decode() -> None:
    engine = object.__new__(InferenceEngine)
    req = _make_request()
    runtime = SimpleNamespace(spec=object())

    with pytest.raises(ValueError, match="require non-speculative"):
        engine._build_generation_request(
            runtime,
            req,
            None,
            encoder_input=object(),
        )


def test_extract_private_suppress_next_token_ids_setting() -> None:
    engine = object.__new__(InferenceEngine)

    assert engine._extract_suppress_next_token_ids(None) is None
    assert engine._extract_suppress_next_token_ids({}) is None
    assert engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": None}) is None
    assert engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": []}) is None
    assert engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": [3, 5, 3]}) == (3, 5)

    with pytest.raises(TypeError, match="settings._suppress_next_token_ids"):
        engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": 3})
    with pytest.raises(TypeError, match="settings._suppress_next_token_ids"):
        engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": ["3"]})
    with pytest.raises(TypeError, match="settings._suppress_next_token_ids"):
        engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": [True]})
    with pytest.raises(ValueError, match="settings._suppress_next_token_ids"):
        engine._extract_suppress_next_token_ids({"_suppress_next_token_ids": [-1]})


def test_validate_suppress_next_token_ids_rejects_bad_request_scope() -> None:
    engine = object.__new__(InferenceEngine)
    runtime = SimpleNamespace(vocab_size=4)

    def skill_state(
        *,
        allowed: list[int] | None = None,
        suppressed: list[int] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            allowed_token_ids=lambda runtime: allowed,
            suppressed_token_ids=lambda runtime: suppressed,
        )

    valid_request = SimpleNamespace(suppress_next_token_ids=(1,))
    engine._validate_suppress_next_token_ids(
        runtime,
        valid_request,
        skill_state(allowed=[0, 1], suppressed=None),
    )

    with pytest.raises(ValueError, match="vocab size"):
        engine._validate_suppress_next_token_ids(
            runtime,
            SimpleNamespace(suppress_next_token_ids=(4,)),
            skill_state(),
        )

    with pytest.raises(ValueError, match="removed every allowed next token"):
        engine._validate_suppress_next_token_ids(
            runtime,
            SimpleNamespace(suppress_next_token_ids=(0, 1)),
            skill_state(allowed=[0, 1]),
        )

    with pytest.raises(ValueError, match="removed every next token"):
        engine._validate_suppress_next_token_ids(
            runtime,
            SimpleNamespace(suppress_next_token_ids=(0, 1)),
            skill_state(suppressed=[2, 3]),
        )
