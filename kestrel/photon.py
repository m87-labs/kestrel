"""Photon telemetry reporting and API key handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import socket
import threading
import uuid
from typing import Any, Optional

import httpx
import torch

from kestrel.config import RuntimeConfig


_LOGGER = logging.getLogger(__name__)
_DOCS_URL = "https://moondream.ai/docs"
_TELEMETRY_FLUSH_INTERVAL_SECONDS = 60
_MAX_PENDING_REPORTS = 100
_MAX_REPORTS_PER_UPLOAD = 10


@dataclass(slots=True)
class _TelemetryWindow:
    started_at: datetime


@dataclass(slots=True)
class _TelemetryBucket:
    request_count: int = 0
    error_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class PhotonReporter:
    """Tracks local usage rollups and reports them to the Photon API."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        runtime_device: torch.device,
        *,
        api_key: Optional[str],
        api_base_url: str,
    ) -> None:
        self._api_key = self._normalize_api_key(api_key)
        self._api_base_url = api_base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=10.0,
            headers={"X-Moondream-Auth": self._api_key} if self._api_key else None,
        )
        self._service_name = runtime_cfg.service_name
        self._model = runtime_cfg.model
        self._hostname = socket.gethostname()
        self._instance_id = str(uuid.uuid4())
        self._active_gpu = self._describe_active_gpu(runtime_device)
        self._window_lock = threading.Lock()
        self._window = _TelemetryWindow(started_at=self._utc_now())
        self._window_buckets: dict[str, _TelemetryBucket] = {}
        self._pending_reports: list[dict[str, Any]] = []
        self._stop_event = asyncio.Event()
        self._telemetry_task: Optional[asyncio.Task[None]] = None

    async def validate_api_key(self) -> bool:
        """Flush startup telemetry and return whether finetunes can use auth.

        Missing or invalid keys never block base inference.
        """
        if self._api_key is not None and not self._is_api_key_header_safe(self._api_key):
            self._disable_auth_header()
            _LOGGER.warning(
                "MOONDREAM_API_KEY has non-ASCII or whitespace characters that "
                "make it unusable as an HTTP header. Finetune inference is "
                "disabled. Check for stray whitespace or pasted control "
                "characters. See %s",
                _DOCS_URL,
            )
            await self._flush_window()
            return False

        had_api_key = self._api_key is not None
        standing = await self._flush_window()

        if standing == "active":
            return True

        if not had_api_key:
            return False

        if standing is None:
            _LOGGER.warning(
                "Could not reach the Moondream API. The Photon inference "
                "engine will start, but finetune inference is disabled while "
                "the API is unreachable.",
            )
            return False

        if standing == "invalid_api_key":
            _LOGGER.warning(
                "MOONDREAM_API_KEY is not valid. Finetune inference is "
                "disabled. Check that the key is correct. See %s",
                _DOCS_URL,
            )
            await self._flush_window(rotate=False)
            return False

        _LOGGER.warning(
            "MOONDREAM_API_KEY is not active (standing=%s). Finetune "
            "inference is disabled.",
            standing,
        )
        return False

    def start(self) -> None:
        if self._telemetry_task is None:
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._telemetry_task is not None:
            await self._telemetry_task
            self._telemetry_task = None
        await self._client.aclose()

    def record_success(
        self,
        *,
        finetune: Optional[str],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._window_lock:
            bucket = self._window_buckets.setdefault(
                self._normalize_finetune(finetune),
                _TelemetryBucket(),
            )
            bucket.request_count += 1
            bucket.input_tokens += max(int(input_tokens), 0)
            bucket.output_tokens += max(int(output_tokens), 0)

    def record_error(self, *, finetune: Optional[str]) -> None:
        with self._window_lock:
            bucket = self._window_buckets.setdefault(
                self._normalize_finetune(finetune),
                _TelemetryBucket(),
            )
            bucket.request_count += 1
            bucket.error_count += 1

    async def _telemetry_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=_TELEMETRY_FLUSH_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass

            await self._flush_window()
            if self._stop_event.is_set():
                break

    async def _flush_window(self, *, rotate: bool = True) -> Optional[str]:
        """Flush pending telemetry reports and return account standing.

        Returns the standing string from the telemetry response:
        ``"active"``, ``"key_revoked"``, ``"billing_paused"``,
        ``"inactive"``, ``"anonymous"``, ``"error"``, or ``None`` if the
        flush could not be completed.
        """
        reports = self._rotate_window() if rotate else []

        self._pending_reports.extend(reports)
        if len(self._pending_reports) > _MAX_PENDING_REPORTS:
            self._pending_reports = self._pending_reports[-_MAX_PENDING_REPORTS:]

        standing: Optional[str] = None

        while self._pending_reports:
            batch = self._pending_reports[:_MAX_REPORTS_PER_UPLOAD]
            try:
                response = await self._client.post(
                    f"{self._api_base_url}/v1/photon/telemetry",
                    json={"reports": batch},
                )
            except httpx.HTTPError:
                return None

            if response.status_code == 401:
                if self._api_key is not None:
                    self._disable_auth_header()
                    return "invalid_api_key"
                return None

            if response.status_code >= 400:
                return None

            del self._pending_reports[:len(batch)]

            # Read standing from the last successful response.
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    standing = payload.get("standing")
            except ValueError:
                pass

            if standing is None:
                standing = "active"

        return standing

    def _disable_auth_header(self) -> None:
        self._api_key = None
        try:
            del self._client.headers["X-Moondream-Auth"]
        except KeyError:
            pass

    def _rotate_window(self) -> list[dict[str, Any]]:
        now = self._utc_now()
        with self._window_lock:
            current = self._window
            current_buckets = self._window_buckets
            self._window = _TelemetryWindow(started_at=now)
            self._window_buckets = {}

        if not current_buckets:
            current_buckets = {"": _TelemetryBucket()}

        reports: list[dict[str, Any]] = []
        for finetune, bucket in current_buckets.items():
            report: dict[str, Any] = {
                "report_id": str(uuid.uuid4()),
                "instance_id": self._instance_id,
                "service_name": self._service_name,
                "model": self._model,
                "hostname": self._hostname,
                "window_start": self._isoformat(current.started_at),
                "window_end": self._isoformat(now),
                "request_count": bucket.request_count,
                "error_count": bucket.error_count,
                "input_tokens": bucket.input_tokens,
                "output_tokens": bucket.output_tokens,
                "active_gpu": self._active_gpu,
            }
            if finetune:
                report["finetune"] = finetune
            reports.append(report)

        return reports

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _isoformat(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _normalize_finetune(finetune: Optional[str]) -> str:
        normalized = (finetune or "").strip()
        return normalized

    @staticmethod
    def _normalize_api_key(api_key: Optional[str]) -> Optional[str]:
        normalized = api_key.strip() if api_key else ""
        return normalized or None

    @staticmethod
    def _is_api_key_header_safe(api_key: str) -> bool:
        return api_key.isascii() and all(
            not c.isspace() and c.isprintable() for c in api_key
        )

    @staticmethod
    def _describe_active_gpu(runtime_device: torch.device) -> dict[str, Any]:
        if runtime_device.type != "cuda" or not torch.cuda.is_available():
            return {
                "name": runtime_device.type,
                "device": str(runtime_device),
                "total_memory_bytes": 0,
            }

        device_index = torch.cuda.current_device() if runtime_device.index is None else runtime_device.index
        props = torch.cuda.get_device_properties(device_index)
        return {
            "name": props.name,
            "device": str(runtime_device),
            "total_memory_bytes": int(props.total_memory),
        }
