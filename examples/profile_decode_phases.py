#!/usr/bin/env python3
"""Profile decode-time attention and MoE cost at a fixed KV length."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import torch

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.models import MoondreamTextRuntime
from kestrel.moondream import text as text_mod
from kestrel.moondream.layers import (
    LayerNormWeights,
    LinearWeights,
    MLPWeights,
    layer_norm,
    mlp as dense_mlp,
    moe_mlp,
)


@dataclasses.dataclass
class PhaseSample:
    cpu_ms: float
    gpu_ms: float


class PhaseTimer:
    def __init__(self) -> None:
        self._records: dict[str, List[PhaseSample]] = {}

    @contextlib.contextmanager
    def record(self, name: str):
        cuda = torch.cuda.is_available()
        start_event = end_event = None
        if cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_start = perf_counter()
        try:
            yield
        finally:
            cpu_end = perf_counter()
            cpu_ms = (cpu_end - cpu_start) * 1000.0
            gpu_ms = float("nan")
            if cuda and start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                gpu_ms = start_event.elapsed_time(end_event)
            self._records.setdefault(name, []).append(PhaseSample(cpu_ms=cpu_ms, gpu_ms=gpu_ms))

    def merge(self, other: "PhaseTimer") -> None:
        for name, samples in other._records.items():
            self._records.setdefault(name, []).extend(samples)

    def items(self):
        return self._records.items()


_current_timer: PhaseTimer | None = None


def _phase(name: str):
    global _current_timer
    timer = _current_timer
    if timer is None:
        return contextlib.nullcontext()
    return timer.record(name)


# Monkey patches ----------------------------------------------------------------

_original_attn = text_mod.attn
_original_moe_mlp = moe_mlp
_original_mlp = dense_mlp


def _attn_instrumented(*args, **kwargs):
    with _phase("decode.attn.total"):
        return _original_attn(*args, **kwargs)


def _moe_mlp_instrumented(x, mlp_module, experts_per_token, *, mode="decode"):
    if mode == "decode":
        with _phase("decode.moe.total"):
            return _original_moe_mlp(x, mlp_module, experts_per_token, mode=mode)
    return _original_moe_mlp(x, mlp_module, experts_per_token, mode=mode)


def _mlp_instrumented(x, w, lora=None):
    with _phase("decode.dense_mlp.total"):
        return _original_mlp(x, w, lora=lora)


text_mod.attn = _attn_instrumented  # type: ignore[assignment]
text_mod.moe_mlp = _moe_mlp_instrumented  # type: ignore[assignment]
text_mod.mlp = _mlp_instrumented  # type: ignore[assignment]


@dataclasses.dataclass
class SummaryRow:
    name: str
    total_gpu_ms: float
    total_cpu_ms: float
    per_step_gpu_ms: float


def summarise(records: Dict[str, List[PhaseSample]], steps: int) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    for name, samples in records.items():
        gpu_vals = [s.gpu_ms for s in samples if not math.isnan(s.gpu_ms)]
        cpu_vals = [s.cpu_ms for s in samples]
        total_gpu = sum(gpu_vals)
        total_cpu = sum(cpu_vals)
        per_step = total_gpu / steps if steps else float("nan")
        rows.append(SummaryRow(name=name, total_gpu_ms=total_gpu, total_cpu_ms=total_cpu, per_step_gpu_ms=per_step))
    rows.sort(key=lambda r: r.total_gpu_ms, reverse=True)
    return rows


def build_runtime(args: argparse.Namespace) -> MoondreamTextRuntime:
    paths = ModelPaths(weights=args.weights)
    cfg = RuntimeConfig(
        model_paths=paths,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        max_batch_size=args.max_batch_size,
        max_seq_length=args.max_seq_length,
        enable_compile=False,
        enable_cuda_graphs=False,
    )
    return MoondreamTextRuntime(cfg)


def make_prompt_tokens(runtime: MoondreamTextRuntime, length: int) -> torch.Tensor:
    vocab = runtime.model.text.wte.shape[0]
    tokens = torch.randint(0, vocab, (1, length), device=runtime.device, dtype=torch.long)
    return tokens


def profile_decode(args: argparse.Namespace) -> Dict[str, List[PhaseSample]]:
    runtime = build_runtime(args)
    prompt_tokens = make_prompt_tokens(runtime, args.prompt_length)

    state, _ = runtime.start_sequence(prompt_tokens=prompt_tokens)

    dummy_token = torch.zeros((1, 1), dtype=torch.long, device=runtime.device)

    global _current_timer
    timer = PhaseTimer()
    _current_timer = timer

    # warm-up without recording
    for _ in range(args.warmup):
        runtime.decode(state, dummy_token)

    _current_timer = timer
    for _ in range(args.steps):
        runtime.decode(state, dummy_token)

    _current_timer = None

    runtime.release_sequence(state)
    return timer._records


def format_ms(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Profile decode attention vs MoE")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-length", type=int, default=823)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    records = profile_decode(args)
    summary = summarise(records, args.steps)

    header = ["phase", "gpu_ms_total", "gpu_ms_per_step", "cpu_ms_total"]
    print(", ".join(header))
    for row in summary:
        print(
            f"{row.name}, {format_ms(row.total_gpu_ms)}, {format_ms(row.per_step_gpu_ms)}, {format_ms(row.total_cpu_ms)}"
        )

    if args.output:
        payload = {
            "prompt_length": args.prompt_length,
            "steps": args.steps,
            "summary": [
                {
                    "phase": row.name,
                    "gpu_ms_total": row.total_gpu_ms,
                    "gpu_ms_per_step": row.per_step_gpu_ms,
                    "cpu_ms_total": row.total_cpu_ms,
                }
                for row in summary
            ],
        }
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
