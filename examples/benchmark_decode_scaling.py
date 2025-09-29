#!/usr/bin/env python
"""Measure decode throughput scaling without the HTTP server."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch

try:  # optional dependency for image support
    import pyvips  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional path
    pyvips = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.models import MoondreamTextRuntime, SequenceState


@dataclass
class PromptSpec:
    text: str
    tokens: torch.Tensor  # stored on CPU for reuse
    image: Optional["pyvips.Image"] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", type=Path, required=True, help="Path to text weights")
    parser.add_argument("--config", type=Path, help="Optional model config JSON")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda", help="Torch device to benchmark on")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Computation dtype",
    )
    parser.add_argument("--max-batch-size", type=int, default=16, help="Runtime max batch size")
    parser.add_argument("--page-size", type=int, default=128, help="KV cache page size")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum total tokens (prompt + decode)",
    )
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph capture for decode",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Batch sizes to benchmark (space separated)",
    )
    parser.add_argument("--decode-steps", type=int, default=64, help="Decode iterations per trial")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4,
        help="Decode iterations to run before the timed region",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per batch size")

    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text (repeatable). Defaults to a generic caption if omitted.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional file containing prompts (one per line)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image (pyvips) to attach to every request",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write raw trial metrics to this JSON file",
    )

    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _load_image(path: Optional[Path]) -> Optional["pyvips.Image"]:
    if path is None:
        return None
    if pyvips is None:
        raise RuntimeError("pyvips is required for image support; install pyvips and retry")
    try:
        return pyvips.Image.new_from_file(str(path), access="sequential")
    except Exception as exc:  # pragma: no cover - file IO
        raise RuntimeError(f"Failed to load image at {path}: {exc}") from exc


def _load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts_file:
        text = args.prompts_file.read_text(encoding="utf-8")
        prompts.extend(line.strip() for line in text.splitlines() if line.strip())
    if args.prompt:
        prompts.extend(p.strip() for p in args.prompt if p.strip())
    if not prompts:
        prompts = ["Describe the image in detail."]
    return prompts


def _prepare_payloads(
    runtime: MoondreamTextRuntime,
    prompts: Sequence[str],
    image: Optional["pyvips.Image"],
) -> List[PromptSpec]:
    payloads: List[PromptSpec] = []
    for text in prompts:
        tokens = runtime.build_prompt_tokens(text).to("cpu")
        payloads.append(PromptSpec(text=text, tokens=tokens, image=image))
    return payloads


def _cycle_payloads(payloads: Sequence[PromptSpec], count: int) -> List[PromptSpec]:
    if not payloads:
        raise RuntimeError("No prompts available to build batches")
    repeated: List[PromptSpec] = []
    for idx in range(count):
        repeated.append(payloads[idx % len(payloads)])
    return repeated


def _argmax_tokens(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return torch.argmax(logits[:, -1, :], dim=-1)
    if logits.ndim == 2:
        return torch.argmax(logits, dim=-1)
    if logits.ndim == 1:
        return torch.argmax(logits.unsqueeze(0), dim=-1)
    raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _run_trial(
    runtime: MoondreamTextRuntime,
    payloads: Sequence[PromptSpec],
    batch_size: int,
    decode_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(runtime.device)

    sequences: List[tuple[PromptSpec, SequenceState]] = []
    initial_tokens: List[torch.Tensor] = []
    prefill_start = time.perf_counter()
    logits: Optional[torch.Tensor] = None
    for idx in range(batch_size):
        payload = payloads[idx]
        state, logits = runtime.start_sequence(
            prompt_tokens=payload.tokens.clone(),
            image=payload.image,
            max_new_tokens=decode_steps + warmup_steps,
        )
        sequences.append((payload, state))
        token = _argmax_tokens(logits).to(runtime.device)
        initial_tokens.append(token.view(-1))
    _maybe_sync(runtime.device)
    prefill_time = time.perf_counter() - prefill_start

    if not initial_tokens:
        raise RuntimeError("No sequences prepared for decode trial")
    next_tokens = torch.stack(initial_tokens, dim=0).to(runtime.device)

    def _decode_once(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = token_ids.view(len(sequences)).to(runtime.device)
        logits_out = runtime.decode_batch([state for _, state in sequences], token_ids)
        for _, seq_state in sequences:
            seq_state.advance()
        return logits_out, _argmax_tokens(logits_out)

    if warmup_steps > 0:
        for _ in range(warmup_steps):
            logits, next_tokens = _decode_once(next_tokens)

    _maybe_sync(runtime.device)
    decode_start = time.perf_counter()
    for _ in range(decode_steps):
        logits, next_tokens = _decode_once(next_tokens.to(runtime.device))
    _maybe_sync(runtime.device)
    decode_time = time.perf_counter() - decode_start

    for _, state in sequences:
        runtime.release_sequence(state)

    total_decode_tokens = batch_size * decode_steps
    tokens_per_s = total_decode_tokens / decode_time if decode_time > 0 else float("nan")

    metrics = {
        "prefill_time_s": prefill_time,
        "prefill_time_per_request_s": prefill_time / batch_size if batch_size else 0.0,
        "decode_time_s": decode_time,
        "decode_tokens": total_decode_tokens,
        "decode_toks_per_s": tokens_per_s,
    }

    if runtime.device.type == "cuda" and torch.cuda.is_available():
        metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)

    return metrics


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.dtype)

    model_paths = ModelPaths(
        weights=args.weights,
        config_json=args.config,
        tokenizer=args.tokenizer,
    )
    runtime_cfg = RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=dtype,
        max_batch_size=args.max_batch_size,
        page_size=args.page_size,
        max_seq_length=args.max_seq_length,
        enable_compile=not args.disable_compile,
        enable_cuda_graphs=not args.disable_cuda_graphs,
    )

    runtime = MoondreamTextRuntime(runtime_cfg)
    effective_max_batch = runtime.max_batch_size - 1  # batch index 0 reserved by paged cache
    print(
        f"Loaded runtime on {runtime.device} (dtype {runtime.dtype}), runtime max_batch_size={runtime.max_batch_size}"
    )
    if effective_max_batch <= 0:
        raise RuntimeError(
            "Runtime max_batch_size must be greater than 1 to admit any requests"
        )

    image = _load_image(args.image)
    prompt_texts = _load_prompts(args)
    payloads = _prepare_payloads(runtime, prompt_texts, image)
    max_batch = max(args.batch_sizes)
    if max_batch > effective_max_batch:
        raise ValueError(
            f"Requested batch size {max_batch} exceeds runtime capacity ({effective_max_batch})"
        )

    results: list[dict[str, object]] = []

    for batch_size in args.batch_sizes:
        batch_payloads = _cycle_payloads(payloads, batch_size)
        trial_metrics: list[dict[str, float]] = []
        print(f"\nBatch size {batch_size}")
        for trial in range(args.trials):
            metrics = _run_trial(
                runtime,
                batch_payloads,
                batch_size=batch_size,
                decode_steps=args.decode_steps,
                warmup_steps=args.warmup_steps,
            )
            trial_metrics.append(metrics)
            print(
                f"  Trial {trial+1}/{args.trials}: decode {metrics['decode_time_s']:.3f}s "
                f"({metrics['decode_toks_per_s']:.1f} tok/s), prefill {metrics['prefill_time_s']:.3f}s"
            )
        avg_decode = sum(m["decode_toks_per_s"] for m in trial_metrics) / args.trials
        avg_prefill = sum(m["prefill_time_per_request_s"] for m in trial_metrics) / args.trials
        results.append(
            {
                "batch_size": batch_size,
                "trials": trial_metrics,
                "avg_decode_toks_per_s": avg_decode,
                "avg_prefill_time_per_request_s": avg_prefill,
            }
        )
        print(
            f"  -> mean decode throughput {avg_decode:.1f} tok/s | "
            f"mean prefill {avg_prefill*1000:.2f} ms/request"
        )

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
