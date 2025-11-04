#!/usr/bin/env python3
"""Profile the token sampling kernel used during batched decoding.

Examples
--------
  uv run python scripts/profile_sampling.py --device cuda --dtype bfloat16 \
      --batch-size 64 --vocab-size 65536 --iterations 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kestrel.scheduler.sampling import sample_tokens

# Minimal mapping from CLI dtype strings to actual torch dtypes.
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile kestrel.scheduler.sampling.sample_tokens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument("--batch-size", type=int, default=4, help="request batch size")
    parser.add_argument("--vocab-size", type=int, default=51200, help="vocabulary size")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=sorted(DTYPE_MAP),
        default="bfloat16",
        help="dtype for logits/temperatures/top_p tensors",
    )
    parser.add_argument("--device", type=str, default=default_device, help="device for tensors")
    parser.add_argument("--iterations", type=int, default=200, help="profiled iterations")
    parser.add_argument("--warmup-iters", type=int, default=20, help="warmup iterations to amortize startup")
    parser.add_argument("--temperature", type=float, default=0.2, help="base sampling temperature")
    parser.add_argument("--temperature-std", type=float, default=0.05, help="std-dev for per-request temperature jitter")
    parser.add_argument("--top-p", type=float, default=0.9, help="base nucleus sampling probability")
    parser.add_argument("--top-p-min", type=float, default=0.6, help="minimum top-p when sampling with jitter")
    parser.add_argument("--seed", type=int, default=0, help="torch manual seed")
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="regenerate logits and sampling parameters on every iteration",
    )
    parser.add_argument(
        "--export-trace",
        type=Path,
        help="optional Chrome trace export path (requires torch.profiler)",
    )
    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="skip torch.profiler and emit simple wall-clock timings instead",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="capture tensor shapes inside the profiler trace",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="capture memory events inside the profiler trace",
    )

    return parser.parse_args()


def make_inputs(
    *,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    temperature: float,
    temperature_std: float,
    top_p: float,
    top_p_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)

    if temperature_std > 0.0:
        noise = torch.randn(batch_size, device=device, dtype=dtype) * temperature_std
    else:
        noise = torch.zeros(batch_size, device=device, dtype=dtype)
    temperatures = torch.clamp(
        torch.full((batch_size,), temperature, device=device, dtype=dtype) + noise,
        min=0.0,
    )

    if top_p_min > top_p:
        raise ValueError("--top-p-min must be <= --top-p")
    if top_p_min <= 0.0:
        raise ValueError("--top-p-min must be > 0")
    top_ps = torch.rand(batch_size, device=device, dtype=dtype)
    top_ps = top_ps * (top_p - top_p_min) + top_p_min

    return logits, temperatures, top_ps


def run_sample(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    logits, temperatures, top_ps = inputs
    return sample_tokens(logits, temperatures, top_ps)


def ensure_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def wall_clock_profile(
    inputs_iter: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    iterations: int,
    warmup_iters: int,
    device: torch.device,
) -> None:
    # Warmup loops to populate caches and trigger kernel compilation.
    for _ in range(warmup_iters):
        _ = run_sample(next(inputs_iter))
        ensure_cuda_sync(device)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = run_sample(next(inputs_iter))
        ensure_cuda_sync(device)
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    avg_ms = total_ms / max(iterations, 1)
    print(f"Wall-clock total: {total_ms:.3f} ms over {iterations} iterations")
    print(f"Wall-clock average: {avg_ms:.3f} ms per call")


def profiler_profile(
    inputs_iter: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    iterations: int,
    warmup_iters: int,
    device: torch.device,
    record_shapes: bool,
    profile_memory: bool,
    export_trace: Path | None,
) -> None:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError as exc:  # pragma: no cover - torch ships profiler by default
        print(f"torch.profiler is unavailable: {exc}", file=sys.stderr)
        print("Falling back to wall-clock timings.\n", file=sys.stderr)
        wall_clock_profile(
            inputs_iter,
            iterations=iterations,
            warmup_iters=warmup_iters,
            device=device,
        )
        return

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    for _ in range(warmup_iters):
        _ = run_sample(next(inputs_iter))
        ensure_cuda_sync(device)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
    ) as prof:
        for _ in range(iterations):
            with record_function("sample_tokens"):
                _ = run_sample(next(inputs_iter))
            ensure_cuda_sync(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    if export_trace:
        export_trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(export_trace))
        print(f"Profiler trace exported to {export_trace}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        dtype = DTYPE_MAP[args.dtype]
    except KeyError as exc:
        raise SystemExit(f"Unsupported dtype {args.dtype!r}") from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() returned False")

    def inputs_generator() -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        base_inputs = make_inputs(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            dtype=dtype,
            device=device,
            temperature=args.temperature,
            temperature_std=args.temperature_std,
            top_p=args.top_p,
            top_p_min=args.top_p_min,
        )

        if not args.refresh_inputs:
            while True:
                yield base_inputs

        while True:
            yield make_inputs(
                batch_size=args.batch_size,
                vocab_size=args.vocab_size,
                dtype=dtype,
                device=device,
                temperature=args.temperature,
                temperature_std=args.temperature_std,
                top_p=args.top_p,
                top_p_min=args.top_p_min,
            )

    iterator = inputs_generator()

    if args.no_profiler:
        wall_clock_profile(
            iterator,
            iterations=args.iterations,
            warmup_iters=args.warmup_iters,
            device=device,
        )
    else:
        profiler_profile(
            iterator,
            iterations=args.iterations,
            warmup_iters=args.warmup_iters,
            device=device,
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            export_trace=args.export_trace,
        )


if __name__ == "__main__":
    main()
