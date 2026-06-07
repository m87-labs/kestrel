from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import torch

from kestrel import photon
from kestrel.photon import PhotonReporter


class _FakeAsyncClient:
    responses: list[httpx.Response] = []
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, *, timeout: float, headers: dict[str, str] | None = None) -> None:
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.posts: list[dict[str, Any]] = []
        self.closed = False
        self.instances.append(self)

    async def post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
        self.posts.append({
            "url": url,
            "json": json,
            "headers": dict(self.headers),
        })
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
) -> type[_FakeAsyncClient]:
    _FakeAsyncClient.responses = list(responses)
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(photon.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


def _runtime_cfg() -> SimpleNamespace:
    return SimpleNamespace(service_name="test_service", model="moondream2")


def test_validate_without_api_key_uploads_anonymous_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(
        monkeypatch,
        [httpx.Response(200, json={"accepted": True, "standing": "anonymous"})],
    )

    async def run() -> bool:
        reporter = PhotonReporter(
            _runtime_cfg(),
            torch.device("cpu"),
            api_key=None,
            api_base_url="https://api.moondream.ai",
        )
        reporter.record_success(finetune=None, input_tokens=3, output_tokens=4)
        auth_available = await reporter.validate_api_key()
        await reporter.shutdown()
        return auth_available

    assert asyncio.run(run()) is False

    client = fake_client.instances[0]
    assert client.closed is True
    assert len(client.posts) == 1
    assert "X-Moondream-Auth" not in client.posts[0]["headers"]
    report = client.posts[0]["json"]["reports"][0]
    assert report["request_count"] == 1
    assert report["input_tokens"] == 3
    assert report["output_tokens"] == 4


def test_invalid_api_key_retries_pending_telemetry_anonymously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(
        monkeypatch,
        [
            httpx.Response(401, json={"error": "Invalid API key"}),
            httpx.Response(200, json={"accepted": True, "standing": "anonymous"}),
        ],
    )

    async def run() -> bool:
        reporter = PhotonReporter(
            _runtime_cfg(),
            torch.device("cpu"),
            api_key="bad-key",
            api_base_url="https://api.moondream.ai",
        )
        reporter.record_success(finetune="ft_123@7", input_tokens=5, output_tokens=6)
        auth_available = await reporter.validate_api_key()
        await reporter.shutdown()
        return auth_available

    assert asyncio.run(run()) is False

    client = fake_client.instances[0]
    assert len(client.posts) == 2
    assert client.posts[0]["headers"]["X-Moondream-Auth"] == "bad-key"
    assert "X-Moondream-Auth" not in client.posts[1]["headers"]

    first_reports = client.posts[0]["json"]["reports"]
    retry_reports = client.posts[1]["json"]["reports"]
    assert len(first_reports) == 1
    assert len(retry_reports) == 1
    assert retry_reports[0]["report_id"] == first_reports[0]["report_id"]
    assert retry_reports[0]["finetune"] == "ft_123@7"


def test_active_api_key_keeps_authenticated_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(
        monkeypatch,
        [httpx.Response(200, json={"accepted": True, "standing": "active"})],
    )

    async def run() -> bool:
        reporter = PhotonReporter(
            _runtime_cfg(),
            torch.device("cpu"),
            api_key="valid-key",
            api_base_url="https://api.moondream.ai",
        )
        auth_available = await reporter.validate_api_key()
        await reporter.shutdown()
        return auth_available

    assert asyncio.run(run()) is True

    client = fake_client.instances[0]
    assert len(client.posts) == 1
    assert client.posts[0]["headers"]["X-Moondream-Auth"] == "valid-key"


def test_revoked_key_disables_authenticated_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(
        monkeypatch,
        [httpx.Response(200, json={"accepted": True, "standing": "key_revoked"})],
    )

    async def run() -> bool:
        reporter = PhotonReporter(
            _runtime_cfg(),
            torch.device("cpu"),
            api_key="revoked-key",
            api_base_url="https://api.moondream.ai",
        )
        auth_available = await reporter.validate_api_key()
        await reporter.shutdown()
        return auth_available

    assert asyncio.run(run()) is False

    client = fake_client.instances[0]
    assert len(client.posts) == 1
    assert client.headers["X-Moondream-Auth"] == "revoked-key"
