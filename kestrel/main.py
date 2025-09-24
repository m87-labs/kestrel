"""Command-line entrypoint for Kestrel demos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.models import MoondreamTextRuntime
from kestrel.scheduler import GenerationScheduler


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = value.lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype '{value}'. Choose from {', '.join(sorted(mapping))}."
        )
    return mapping[key]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moondream scheduler demo")
    subparsers = parser.add_subparsers(dest="command")

    schedule = subparsers.add_parser("schedule", help="Run batched text generation")
    schedule.add_argument("prompts", nargs="+", help="Prompts to generate responses for")
    schedule.add_argument("--weights", type=Path, required=True, help="Path to text weights file")
    schedule.add_argument("--config", type=Path, help="Optional model config JSON")
    schedule.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    schedule.add_argument("--device", default="cuda", help="Torch device to run on")
    schedule.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="Computation dtype")
    schedule.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to sample per request")
    schedule.add_argument("--max-batch-size", type=int, default=4, help="Max sequences per decode step")
    schedule.add_argument("--page-size", type=int, default=128, help="KV cache page size")
    schedule.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum total sequence length (prompt + generation)",
    )
    return parser


def _handle_schedule(args: argparse.Namespace) -> None:
    model_paths = ModelPaths(
        weights=args.weights,
        config_json=args.config,
        tokenizer=args.tokenizer,
    )
    runtime_cfg = RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        page_size=args.page_size,
        max_seq_length=args.max_seq_length,
    )

    runtime = MoondreamTextRuntime(runtime_cfg)
    scheduler = GenerationScheduler(runtime)
    for prompt in args.prompts:
        scheduler.submit(prompt, max_new_tokens=args.max_new_tokens)

    results = scheduler.run()
    for result in results:
        print(f"[{result.request_id}] {result.finish_reason}: {result.text}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "schedule":
        _handle_schedule(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
