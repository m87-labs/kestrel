from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Callable

import numpy as np

from kestrel.engine import _AdmissionCoordinator, _PendingRequest


def _make_request(
    *, request_id: int = 1, image: np.ndarray | bytes | None = None
) -> _PendingRequest:
    return _PendingRequest(
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
        adapter=None,
    )


class _FakeRuntime:
    def __init__(self, *, prefix_cache: object | None, prefix_hit: bool) -> None:
        self.prefix_cache = prefix_cache
        self.config = SimpleNamespace(vision=object())
        self._prefix_hit = prefix_hit
        self.cache_checks: list[tuple[list[object], bytes | None, str | None]] = []

    def check_prefix_cache(
        self, tokens_list: list[object], image_hash: bytes | None, adapter: str | None
    ) -> bool:
        self.cache_checks.append((tokens_list, image_hash, adapter))
        return self._prefix_hit


class _FakeImagePreprocessor:
    def __init__(self, futures: list[Future[object]] | None = None) -> None:
        self.futures = futures or [Future()]
        self.submissions: list[tuple[np.ndarray | bytes, object]] = []

    def submit(
        self, image: np.ndarray | bytes, vision_config: object
    ) -> Future[object]:
        self.submissions.append((image, vision_config))
        return self.futures.pop(0)


def _record_failure(
    failures: list[tuple[_PendingRequest, BaseException]]
) -> Callable[[_PendingRequest, BaseException], None]:
    def _fail(req: _PendingRequest, exc: BaseException) -> None:
        failures.append((req, exc))

    return _fail


def test_admission_coordinator_immediately_admits_text_only_request() -> None:
    runtime = _FakeRuntime(prefix_cache=None, prefix_hit=False)
    preprocessor = _FakeImagePreprocessor()
    failures: list[tuple[_PendingRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        image_preprocessor=preprocessor,
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


def test_admission_coordinator_skips_crop_work_on_prefix_hit() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    runtime = _FakeRuntime(prefix_cache=object(), prefix_hit=True)
    preprocessor = _FakeImagePreprocessor()
    failures: list[tuple[_PendingRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=runtime,
        image_preprocessor=preprocessor,
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


def test_admission_coordinator_promotes_completed_crops() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    crop_future: Future[object] = Future()
    preprocessor = _FakeImagePreprocessor([crop_future])
    wake_event = threading.Event()
    failures: list[tuple[_PendingRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(prefix_cache=object(), prefix_hit=False),
        image_preprocessor=preprocessor,
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
    failures: list[tuple[_PendingRequest, BaseException]] = []
    coordinator = _AdmissionCoordinator(
        runtime=_FakeRuntime(prefix_cache=None, prefix_hit=False),
        image_preprocessor=preprocessor,
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
