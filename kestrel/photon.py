"""Photon telemetry reporting and API key validation."""

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
_PRICING_URL = "https://moondream.ai/pricing"
_DOCS_URL = "https://moondream.ai/docs"
_TELEMETRY_FLUSH_INTERVAL_SECONDS = 60
_TELEMETRY_WARNING_INTERVAL_SECONDS = 600
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
        engine: Any,
    ) -> None:
        self._engine = engine
        self._api_key = api_key
        self._api_base_url = api_base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=10.0,
            headers={"X-Moondream-Auth": api_key} if api_key else None,
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
        self._last_telemetry_warning_at = 0.0

    async def validate_api_key(self) -> None:
        """Validate the API key by sending an initial telemetry flush.

        Raises RuntimeError if the key is missing, invalid, or the
        account is not in good standing.  Warns and returns if the
        Moondream API is unreachable (to avoid blocking startup during
        API outages).
        """
        if not self._api_key:
            raise RuntimeError(
                "MOONDREAM_API_KEY is not set. Set this environment variable "
                "to start the Photon inference engine. See %s" % _DOCS_URL
            )

        standing = await self._flush_window()

        if standing is None:
            _LOGGER.warning(
                "Could not reach the Moondream API. The Photon inference "
                "engine will start, but usage will not be reported and "
                "finetunes will not be available while the API is "
                "unreachable.",
            )
            return

        if standing == "invalid_api_key":
            raise RuntimeError(
                "MOONDREAM_API_KEY is not valid. Check that the key is "
                "correct and try again. See %s" % _DOCS_URL
            )

        if standing != "active":
            raise RuntimeError(
                "Your account does not have enough credits to start the "
                "Photon inference engine. Add credits at %s" % _PRICING_URL
            )

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

    async def _flush_window(self) -> Optional[str]:
        """Flush pending telemetry reports and return account standing.

        Returns the standing string from the telemetry response:
        ``"active"``, ``"key_revoked"``, ``"billing_paused"``,
        ``"inactive"``, ``"error"``, or ``None`` if the flush could not
        be completed.
        """
        reports = self._rotate_window()

        if not self._api_key:
            return None

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
            except httpx.HTTPError as exc:
                _LOGGER.warning("Photon usage reporting failed: %s", exc)
                return None

            if response.status_code == 401:
                return "invalid_api_key"

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

        if standing == "key_revoked":
            _LOGGER.error(
                "MOONDREAM_API_KEY has been revoked. The Photon inference "
                "engine will stop accepting new requests.",
            )
            # Schedule engine shutdown as a separate task to avoid
            # deadlock (engine.shutdown awaits this telemetry task).
            asyncio.create_task(self._engine.shutdown())

        elif standing == "invalid_api_key":
            _LOGGER.error(
                "MOONDREAM_API_KEY is no longer valid. The Photon inference "
                "engine will stop accepting new requests.",
            )
            asyncio.create_task(self._engine.shutdown())

        elif standing is not None and standing != "active":
            self._maybe_log_telemetry_warning(
                "Your account has run out of credits. Inference will "
                "continue for this process, but new processes will fail "
                "to start. Add credits at %s" % _PRICING_URL,
                None,
            )

        return standing

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

    def _maybe_log_telemetry_warning(
        self,
        message: str,
        error: Optional[BaseException],
    ) -> None:
        now = asyncio.get_running_loop().time()
        if now - self._last_telemetry_warning_at < _TELEMETRY_WARNING_INTERVAL_SECONDS:
            return

        self._last_telemetry_warning_at = now
        if error is None:
            _LOGGER.warning(message)
            return
        _LOGGER.warning("%s %s", message, error)

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
