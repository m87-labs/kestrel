"""Command-line entrypoint for Kestrel demos."""

from __future__ import annotations

import argparse
import asyncio
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine


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


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, required=True, help="Path to text weights file")
    parser.add_argument("--config", type=Path, help="Optional model config JSON")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda", help="Torch device to run on")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="Computation dtype")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Max sequences per decode step")
    parser.add_argument("--page-size", type=int, default=128, help="KV cache page size")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=131072,
        help="Maximum total sequence length (prompt + generation)",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for eligible runtime paths",
    )
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph capture for batched decode",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moondream scheduler demo")
    subparsers = parser.add_subparsers(dest="command")

    schedule = subparsers.add_parser("schedule", help="Run batched text generation")
    schedule.add_argument("prompts", nargs="+", help="Prompts to generate responses for")
    _add_runtime_args(schedule)
    schedule.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to sample per request")
    schedule.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=20.0,
        help="Micro-batching timeout (in milliseconds) before dispatching queued requests",
    )
    schedule.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Softmax temperature; 0 selects greedy decoding",
    )
    schedule.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling mass (0 < p <= 1)",
    )
    schedule.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated",
    )

    serve = subparsers.add_parser("serve", help="Run the HTTP inference server")
    _add_runtime_args(serve)
    serve.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=20.0,
        help="Micro-batching timeout (in milliseconds) before dispatching queued requests",
    )
    serve.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=64,
        help="Default max tokens to generate when a request does not specify it",
    )
    serve.add_argument(
        "--default-temperature",
        type=float,
        default=0.0,
        help="Default sampling temperature when a request omits it",
    )
    serve.add_argument(
        "--default-top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling mass when a request omits it",
    )
    serve.add_argument("--host", default="0.0.0.0", help="Host address to bind the HTTP server")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server")
    serve.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level for the HTTP server",
    )

    return parser


def _create_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    model_paths = ModelPaths(
        weights=args.weights,
        config_json=args.config,
        tokenizer=args.tokenizer,
    )
    return RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        page_size=args.page_size,
        max_seq_length=args.max_seq_length,
        enable_compile=not args.disable_compile,
        enable_cuda_graphs=not args.disable_cuda_graphs,
    )


async def _handle_schedule(args: argparse.Namespace) -> None:
    if args.temperature < 0.0:
        raise SystemExit("temperature must be non-negative")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise SystemExit("top-p must be in the range (0, 1]")

    runtime_cfg = _create_runtime_config(args)

    engine = await InferenceEngine.create(
        runtime_cfg,
        batch_timeout_s=args.batch_timeout_ms / 1000.0,
    )
    try:
        if args.stream:
            streams = []
            for prompt in args.prompts:
                stream = await engine.submit_streaming(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                streams.append(stream)

            async def consume(stream):
                async for update in stream:
                    print(f"[{stream.request_id}] +{update.text}", flush=True)
                return await stream.result()

            results = await asyncio.gather(*(consume(stream) for stream in streams))
        else:
            submissions = [
                engine.submit(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for prompt in args.prompts
            ]
            results = await asyncio.gather(*submissions)
    finally:
        await engine.shutdown()

    for result in results:
        metrics = result.metrics
        print(
            f"[{result.request_id}] {result.finish_reason}: {result.text} "
            f"(processing={metrics.processing_latency_s:.3f}s, ttft={metrics.ttft_s:.3f}s, "
            f"decode={metrics.decode_latency_s:.3f}s, decode_tokens={metrics.decode_tokens})"
        )


def _handle_serve(args: argparse.Namespace) -> None:
    if args.default_temperature < 0.0:
        raise SystemExit("default-temperature must be non-negative")
    if args.default_top_p <= 0.0 or args.default_top_p > 1.0:
        raise SystemExit("default-top-p must be in the range (0, 1]")
    if args.default_max_new_tokens <= 0:
        raise SystemExit("default-max-new-tokens must be positive")
    if args.port <= 0 or args.port > 65535:
        raise SystemExit("port must be between 1 and 65535")

    runtime_cfg = _create_runtime_config(args)
    batch_timeout = args.batch_timeout_ms / 1000.0

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "uvicorn is required for server mode. Install it with 'pip install uvicorn'."
        ) from exc

    try:
        from kestrel.server import create_app
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Unable to import server module: {exc}") from exc

    app_factory = partial(
        create_app,
        runtime_cfg=runtime_cfg,
        batch_timeout_s=batch_timeout,
        default_max_new_tokens=args.default_max_new_tokens,
        default_temperature=args.default_temperature,
        default_top_p=args.default_top_p,
    )

    uvicorn.run(
        app_factory,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        factory=True,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "schedule":
        asyncio.run(_handle_schedule(args))
    elif args.command == "serve":
        _handle_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
