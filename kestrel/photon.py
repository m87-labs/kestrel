"""Photon license verification and telemetry reporting."""

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
_DEFAULT_PRICING_URL = "https://moondream.ai/pricing"
_TELEMETRY_FLUSH_INTERVAL_SECONDS = 60
_LICENSE_REFRESH_INTERVAL_SECONDS = 600
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


@dataclass(slots=True)
class _LicenseSnapshot:
    state: str
    active_instances: int = 0
    instance_limit: int = 0
    pricing_url: str = _DEFAULT_PRICING_URL


class PhotonReporter:
    """Tracks local usage rollups and commercial-license standing."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        runtime_device: torch.device,
        *,
        api_key: Optional[str],
        api_base_url: str,
    ) -> None:
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
        self._license_task: Optional[asyncio.Task[None]] = None
        self._license_state: Optional[str] = None
        self._last_telemetry_warning_at = 0.0

    async def initial_license_check(self) -> None:
        """Run the first license check. Raises SystemExit for fatal states."""
        await self._refresh_license(log=False)
        if self._license_state == "missing_api_key":
            _LOGGER.error(
                "MOONDREAM_API_KEY is not set. A valid API key is required to "
                "start the server. See %s",
                _DEFAULT_PRICING_URL,
            )
            raise SystemExit(1)
        if self._license_state == "invalid_api_key":
            _LOGGER.error(
                "The provided MOONDREAM_API_KEY is invalid. A valid API key is "
                "required to start the server. See %s",
                _DEFAULT_PRICING_URL,
            )
            raise SystemExit(1)

    def start(self) -> None:
        if self._telemetry_task is None:
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())
        if self._license_task is None:
            self._license_task = asyncio.create_task(self._license_loop())

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._telemetry_task is not None:
            await self._telemetry_task
            self._telemetry_task = None
        if self._license_task is not None:
            await self._license_task
            self._license_task = None
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

    async def _license_loop(self) -> None:
        while True:
            await self._refresh_license()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=_LICENSE_REFRESH_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                continue
            if self._stop_event.is_set():
                break

    async def _refresh_license(self, *, log: bool = True) -> None:
        if not self._api_key:
            self._publish_license_snapshot(
                _LicenseSnapshot(state="missing_api_key"), log=log,
            )
            return

        try:
            response = await self._client.post(
                f"{self._api_base_url}/v1/photon/license",
                json={
                    "instance_id": self._instance_id,
                    "service_name": self._service_name,
                },
            )
        except httpx.HTTPError as exc:
            self._publish_license_snapshot(
                _LicenseSnapshot(state="verification_error"), log=log,
            )
            _LOGGER.debug("Photon license request failed", exc_info=exc)
            return

        if response.status_code == 401:
            self._pending_reports.clear()
            self._publish_license_snapshot(
                _LicenseSnapshot(state="invalid_api_key"), log=log,
            )
            return

        if response.status_code >= 400:
            self._publish_license_snapshot(
                _LicenseSnapshot(state="verification_error"), log=log,
            )
            return

        try:
            payload = response.json()
        except ValueError:
            self._publish_license_snapshot(
                _LicenseSnapshot(state="verification_error"), log=log,
            )
            return

        if not isinstance(payload, dict):
            self._publish_license_snapshot(
                _LicenseSnapshot(state="verification_error"), log=log,
            )
            return

        standing_status = payload.get("standing_status")
        license_status = payload.get("license_status")
        try:
            active_instances = int(payload.get("active_instances", 0))
            instance_limit = int(payload.get("instance_limit", 0))
        except (TypeError, ValueError):
            self._publish_license_snapshot(
                _LicenseSnapshot(state="verification_error"), log=log,
            )
            return
        pricing_url = str(payload.get("pricing_url", _DEFAULT_PRICING_URL))

        if standing_status == "verification_error" or license_status == "verification_error":
            state = "verification_error"
        elif license_status == "licensed":
            state = "licensed"
        elif license_status == "over_limit":
            state = "over_limit"
        else:
            state = "inactive"

        self._publish_license_snapshot(
            _LicenseSnapshot(
                state=state,
                active_instances=active_instances,
                instance_limit=instance_limit,
                pricing_url=pricing_url,
            ),
            log=log,
        )

    async def _flush_window(self) -> None:
        reports = self._rotate_window()

        if not self._api_key or self._license_state in {"missing_api_key", "invalid_api_key"}:
            return

        self._pending_reports.extend(reports)
        if len(self._pending_reports) > _MAX_PENDING_REPORTS:
            self._pending_reports = self._pending_reports[-_MAX_PENDING_REPORTS:]

        while self._pending_reports:
            batch = self._pending_reports[:_MAX_REPORTS_PER_UPLOAD]
            try:
                response = await self._client.post(
                    f"{self._api_base_url}/v1/photon/telemetry",
                    json={"reports": batch},
                )
            except httpx.HTTPError as exc:
                self._maybe_log_telemetry_warning(
                    "Photon telemetry upload failed. Local inference will continue.",
                    exc,
                )
                return

            if response.status_code == 401:
                self._pending_reports.clear()
                self._publish_license_snapshot(
                    _LicenseSnapshot(state="invalid_api_key"),
                )
                return

            if response.status_code >= 400:
                self._maybe_log_telemetry_warning(
                    "Photon telemetry upload failed. Local inference will continue.",
                    None,
                )
                return

            del self._pending_reports[:len(batch)]

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

    def _publish_license_snapshot(
        self, snapshot: _LicenseSnapshot, *, log: bool = True,
    ) -> None:
        previous_state = self._license_state
        self._license_state = snapshot.state

        if not log:
            return

        if snapshot.state == "licensed":
            if previous_state and previous_state != "licensed":
                _LOGGER.info("Photon license verification succeeded.")
            return

        if snapshot.state == "missing_api_key":
            _LOGGER.warning(
                "Photon license could not be verified because MOONDREAM_API_KEY is not set. "
                "Inference will continue, but only noncommercial and evaluation use is "
                "permitted without a valid license. See %s",
                snapshot.pricing_url,
            )
            return

        if snapshot.state == "invalid_api_key":
            _LOGGER.warning(
                "Photon license could not be verified because the provided API key is invalid. "
                "Inference will continue, but only noncommercial and evaluation use is "
                "permitted without a valid license. See %s",
                snapshot.pricing_url,
            )
            return

        if snapshot.state == "verification_error":
            _LOGGER.warning(
                "Photon license could not be verified due to a connectivity or server issue. "
                "Inference will continue, but commercial use requires a valid license. "
                "See %s",
                snapshot.pricing_url,
            )
            return

        if snapshot.state == "over_limit":
            _LOGGER.warning(
                "Photon active instance count exceeds the licensed limit "
                "(%s active / %s allowed). Please reduce active instances or "
                "upgrade your plan. See %s",
                snapshot.active_instances,
                snapshot.instance_limit,
                snapshot.pricing_url,
            )
            return

        _LOGGER.warning(
            "Photon commercial license is not active for this API key or account. "
            "Inference will continue, but only noncommercial and evaluation use is "
            "permitted without a valid license. See %s",
            snapshot.pricing_url,
        )

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
