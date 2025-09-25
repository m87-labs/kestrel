#!/usr/bin/env python
"""Load test the Kestrel HTTP server with incremental TPS ramping."""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx


@dataclass(frozen=True)
class StageConfig:
    target_rps: float
    duration_s: float
    concurrency_limit: int


@dataclass
class RequestRecord:
    start_time: float
    end_time: float
    latency_s: float
    status_code: Optional[int]
    ok: bool
    error: Optional[str]
    server_metrics: Dict[str, Any]

    @property
    def ttft_s(self) -> Optional[float]:
        value = self.server_metrics.get("ttft_s")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def decode_latency_s(self) -> Optional[float]:
        value = self.server_metrics.get("decode_latency_s")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def processing_latency_s(self) -> Optional[float]:
        value = self.server_metrics.get("processing_latency_s")
        return float(value) if isinstance(value, (int, float)) else None


@dataclass
class StageResult:
    config: StageConfig
    records: List[RequestRecord]
    overload_reasons: List[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for rec in self.records if rec.ok)

    @property
    def error_count(self) -> int:
        return len(self.records) - self.success_count

    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def actual_duration_s(self) -> float:
        if not self.records:
            return 0.0
        start = min(rec.start_time for rec in self.records)
        end = max(rec.end_time for rec in self.records)
        return max(end - start, 1e-6)

    @property
    def throughput_rps(self) -> float:
        duration = self.actual_duration_s
        if duration <= 0:
            return 0.0
        return self.success_count / duration

    @property
    def error_rate(self) -> float:
        if not self.records:
            return 0.0
        return self.error_count / len(self.records)

    @property
    def overloaded(self) -> bool:
        return bool(self.overload_reasons)


DEFAULT_PROMPTS = [
    "Describe the image.",
    "What stands out the most?",
    "Summarize the scene in one sentence.",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Kestrel HTTP server")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/query",
        help="HTTP endpoint to exercise",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=False,
        help="Optional image file to send as base64 data URL",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="File containing prompts (one per line). Defaults to built-in prompts.",
    )
    parser.add_argument(
        "--stage-duration",
        type=float,
        default=20.0,
        help="Duration in seconds to run each load stage",
    )
    parser.add_argument(
        "--start-rps",
        type=float,
        default=1.0,
        help="Initial target requests per second",
    )
    parser.add_argument(
        "--rps-step",
        type=float,
        default=1.0,
        help="Increment in target RPS after each successful stage",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=64.0,
        help="Upper bound on target RPS",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Maximum number of in-flight requests",
    )
    parser.add_argument(
        "--concurrency-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to target RPS when deriving concurrency per stage",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout per HTTP request in seconds",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.05,
        help="Maximum tolerated error rate before marking a stage overloaded",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=2000.0,
        help="P95 latency threshold (ms) for overload detection",
    )
    parser.add_argument(
        "--ttft-threshold-ms",
        type=float,
        default=1500.0,
        help="P95 TTFT threshold (ms) for overload detection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results",
    )
    parser.add_argument(
        "--settings-temperature",
        type=float,
        default=0.0,
        help="Temperature field to send in request settings",
    )
    parser.add_argument(
        "--settings-top-p",
        type=float,
        default=1.0,
        help="Top-p field to send in request settings",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens request override",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Number of warmup requests before measuring",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-stage JSON details",
    )
    args = parser.parse_args(argv)
    if args.stage_duration <= 0:
        raise SystemExit("stage-duration must be positive")
    if args.max_concurrency < 1:
        raise SystemExit("max-concurrency must be at least 1")
    if not (0.0 <= args.error_threshold < 1.0):
        raise SystemExit("error-threshold must be between 0 and 1")
    if args.latency_threshold_ms <= 0:
        raise SystemExit("latency-threshold-ms must be positive")
    if args.ttft_threshold_ms <= 0:
        raise SystemExit("ttft-threshold-ms must be positive")
    if args.settings_top_p <= 0 or args.settings_top_p > 1:
        raise SystemExit("settings-top-p must be in (0, 1]")
    if args.settings_temperature < 0:
        raise SystemExit("settings-temperature must be >= 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise SystemExit("max-new-tokens must be positive when provided")
    return args


def load_prompts(path: Optional[Path]) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS.copy()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise SystemExit("Prompt file is empty")
    return prompts


def load_image_base64(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    data = path.read_bytes()
    mime = infer_mime_type(path)
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{payload}"


def infer_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    clamped = max(0.0, min(100.0, pct))
    rank = (clamped / 100.0) * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[int(rank)]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_latency(records: Iterable[RequestRecord]) -> Dict[str, Optional[float]]:
    latencies = sorted(rec.latency_s for rec in records)
    return {
        "p50": percentile(latencies, 50),
        "p90": percentile(latencies, 90),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": statistics.mean(latencies) if latencies else None,
    }


def summarize_server_metric(records: Iterable[RequestRecord], key: str) -> Dict[str, Optional[float]]:
    values = sorted(rec.server_metrics.get(key) for rec in records if isinstance(rec.server_metrics.get(key), (int, float)))
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "mean": None}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def aggregate_token_stats(
    records: Iterable[RequestRecord],
    *,
    duration_s: float,
) -> Dict[str, float]:
    prompt_total = 0.0
    decode_total = 0.0
    success = 0
    for rec in records:
        prompt_val = rec.server_metrics.get("prompt_tokens")
        decode_val = rec.server_metrics.get("decode_tokens")
        if isinstance(prompt_val, (int, float)):
            prompt_total += float(prompt_val)
        if isinstance(decode_val, (int, float)):
            decode_total += float(decode_val)
        if rec.ok:
            success += 1

    duration = max(duration_s, 1e-6)
    return {
        "input_total": prompt_total,
        "input_per_s": prompt_total / duration,
        "input_per_request": prompt_total / success if success else 0.0,
        "output_total": decode_total,
        "output_per_s": decode_total / duration,
        "output_per_request": decode_total / success if success else 0.0,
    }


def build_stage_configs(args: argparse.Namespace) -> List[StageConfig]:
    if args.start_rps <= 0:
        raise SystemExit("start-rps must be positive")
    if args.rps_step <= 0:
        raise SystemExit("rps-step must be positive")
    if args.max_rps < args.start_rps:
        raise SystemExit("max-rps must be >= start-rps")
    configs: List[StageConfig] = []
    target = args.start_rps
    while target <= args.max_rps + 1e-6:
        concurrency = max(1, min(
            args.max_concurrency,
            math.ceil(target * max(args.concurrency_scale, 1e-3)),
        ))
        configs.append(StageConfig(target_rps=target, duration_s=args.stage_duration, concurrency_limit=concurrency))
        target += args.rps_step
    return configs


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> RequestRecord:
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        latency = time.perf_counter() - start
        if response.status_code == 200:
            body = response.json()
            metrics = body.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            record = RequestRecord(
                start_time=start,
                end_time=time.perf_counter(),
                latency_s=latency,
                status_code=response.status_code,
                ok=True,
                error=None,
                server_metrics=metrics,
            )
        else:
            record = RequestRecord(
                start_time=start,
                end_time=time.perf_counter(),
                latency_s=latency,
                status_code=response.status_code,
                ok=False,
                error=response.text.strip() or f"HTTP {response.status_code}",
                server_metrics={},
            )
    except Exception as exc:  # pragma: no cover - network failures
        latency = time.perf_counter() - start
        record = RequestRecord(
            start_time=start,
            end_time=time.perf_counter(),
            latency_s=latency,
            status_code=None,
            ok=False,
            error=str(exc),
            server_metrics={},
        )
    return record


async def run_stage(
    stage: StageConfig,
    client: httpx.AsyncClient,
    url: str,
    payloads: Sequence[Dict[str, Any]],
    timeout: float,
) -> List[RequestRecord]:
    semaphore = asyncio.Semaphore(stage.concurrency_limit)
    records: List[RequestRecord] = []
    tasks: List[asyncio.Task[RequestRecord]] = []
    if not payloads:
        return []
    payload_count = len(payloads)
    payload_index = 0

    stage_start = time.perf_counter()
    stage_end = stage_start + stage.duration_s
    scheduled = 0

    async def launch_request(payload: Dict[str, Any]) -> None:
        async with semaphore:
            record = await send_request(client, url, payload, timeout)
            records.append(record)

    while True:
        now = time.perf_counter()
        if now >= stage_end:
            break
        target_next = stage_start + scheduled / stage.target_rps
        if now < target_next:
            await asyncio.sleep(min(target_next - now, 0.01))
            continue
        payload_template = payloads[payload_index]
        payload_index = (payload_index + 1) % payload_count
        payload = copy.deepcopy(payload_template)
        scheduled += 1
        tasks.append(asyncio.create_task(launch_request(payload)))

    if tasks:
        await asyncio.gather(*tasks)

    return records


def compute_overload(
    result: StageResult,
    latency_threshold_s: float,
    ttft_threshold_s: float,
    error_threshold: float,
) -> List[str]:
    reasons: List[str] = []
    total = result.total_requests
    if total == 0:
        return ["no responses"]
    if result.error_rate > error_threshold:
        reasons.append(f"error_rate {result.error_rate:.3f} > {error_threshold:.3f}")
    latencies = sorted(rec.latency_s for rec in result.records)
    p95_latency = percentile(latencies, 95)
    if p95_latency is not None and p95_latency > latency_threshold_s:
        reasons.append(
            f"latency_p95 {p95_latency*1000:.1f}ms > {latency_threshold_s*1000:.1f}ms"
        )
    ttft_values = sorted(
        rec.ttft_s for rec in result.records if isinstance(rec.ttft_s, (int, float))
    )
    if ttft_values:
        p95_ttft = percentile(ttft_values, 95)
        if p95_ttft is not None and p95_ttft > ttft_threshold_s:
            reasons.append(
                f"ttft_p95 {p95_ttft*1000:.1f}ms > {ttft_threshold_s*1000:.1f}ms"
            )
    return reasons


async def warmup(
    client: httpx.AsyncClient,
    url: str,
    payloads: Sequence[Dict[str, Any]],
    timeout: float,
    count: int,
) -> None:
    if not payloads:
        return
    payload_count = len(payloads)
    for i in range(count):
        template = payloads[i % payload_count]
        payload = copy.deepcopy(template)
        await send_request(client, url, payload, timeout)


def build_payload_templates(
    prompts: Sequence[str],
    image_data_url: Optional[str],
    temperature: float,
    top_p: float,
    max_new_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for prompt in prompts:
        payload: Dict[str, Any] = {
            "question": prompt,
            "settings": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if image_data_url is not None:
            payload["image_url"] = image_data_url
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        templates.append(payload)
    return templates


def render_stage_summary(result: StageResult) -> Dict[str, Any]:
    success = result.success_count
    total = result.total_requests
    latency_summary = summarize_latency(result.records)
    ttft_summary = summarize_server_metric(result.records, "ttft_s")
    decode_summary = summarize_server_metric(result.records, "decode_latency_s")
    processing_summary = summarize_server_metric(result.records, "processing_latency_s")
    token_summary = aggregate_token_stats(result.records, duration_s=result.actual_duration_s)
    return {
        "target_rps": result.config.target_rps,
        "concurrency_limit": result.config.concurrency_limit,
        "planned_duration_s": result.config.duration_s,
        "actual_duration_s": result.actual_duration_s,
        "requests_total": total,
        "requests_success": success,
        "requests_error": total - success,
        "error_rate": result.error_rate,
        "throughput_rps": result.throughput_rps,
        "latency": latency_summary,
        "ttft": ttft_summary,
        "decode_latency": decode_summary,
        "processing_latency": processing_summary,
        "tokens": token_summary,
        "overloaded": result.overloaded,
        "overload_reasons": result.overload_reasons,
    }


def print_stage_line(summary: Dict[str, Any]) -> None:
    latency = summary["latency"]
    ttft = summary["ttft"]
    tokens = summary["tokens"]
    print(
        f"RPS target={summary['target_rps']:.1f} obs={summary['throughput_rps']:.2f} | "
        f"concurrency={summary['concurrency_limit']} | "
        f"succ={summary['requests_success']}/{summary['requests_total']} | "
        f"err={summary['error_rate']*100:.1f}% | "
        f"lat-p95={latency['p95']*1000 if latency['p95'] is not None else float('nan'):.1f}ms | "
        f"ttft-p95={ttft['p95']*1000 if ttft['p95'] is not None else float('nan'):.1f}ms | "
        f"tok-in/s={tokens['input_per_s']:.1f} | tok-out/s={tokens['output_per_s']:.1f} | "
        f"overloaded={summary['overloaded']}"
    )
    if summary["overloaded"]:
        print("  reasons:", "; ".join(summary["overload_reasons"]))


async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    prompts = load_prompts(args.prompts)
    image_data_url = load_image_base64(args.image)
    payload_templates = build_payload_templates(
        prompts,
        image_data_url,
        args.settings_temperature,
        args.settings_top_p,
        args.max_new_tokens,
    )
    if not payload_templates:
        raise SystemExit("No payloads prepared")

    configs = build_stage_configs(args)
    latency_threshold_s = args.latency_threshold_ms / 1000.0
    ttft_threshold_s = args.ttft_threshold_ms / 1000.0

    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        if args.warmup_requests > 0:
            await warmup(
                client,
                args.url,
                payload_templates,
                args.request_timeout,
                args.warmup_requests,
            )

        for config in configs:
            stage_records = await run_stage(
                config,
                client,
                args.url,
                payload_templates,
                args.request_timeout,
            )
            stage_result = StageResult(config=config, records=stage_records)
            stage_result.overload_reasons = compute_overload(
                stage_result,
                latency_threshold_s,
                ttft_threshold_s,
                args.error_threshold,
            )
            summary = render_stage_summary(stage_result)
            print_stage_line(summary)
            if args.verbose:
                print(json.dumps(summary, indent=2))
            results.append(summary)
            if summary["overloaded"]:
                break

    payload = {
        "url": args.url,
        "image": str(args.image) if args.image else None,
        "start_rps": args.start_rps,
        "rps_step": args.rps_step,
        "max_rps": args.max_rps,
        "stage_duration_s": args.stage_duration,
        "error_threshold": args.error_threshold,
        "latency_threshold_ms": args.latency_threshold_ms,
        "ttft_threshold_ms": args.ttft_threshold_ms,
        "results": results,
    }

    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        payload = asyncio.run(main_async(args))
    except KeyboardInterrupt:  # pragma: no cover - user abort
        print("Interrupted", file=sys.stderr)
        return

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
