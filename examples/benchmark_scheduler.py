"""Measure Moondream scheduler throughput using synthetic prompt traffic."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Sequence

import pyvips
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.moondream.runtime import MoondreamRuntime
from kestrel.skills import QueryRequest, QuerySettings, QuerySkill


@dataclass
class PromptPayload:
    """Cached prompt metadata used for repeated benchmark rounds."""

    text: str
    tokens: torch.Tensor  # stored on CPU for reuse
    length: int  # number of tokens including BOS/prefix/suffix
    image: Optional[pyvips.Image] = None
    request: QueryRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", type=Path, required=True, help="Path to text weights file")
    parser.add_argument("--config", type=Path, help="Optional model config JSON override")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda", help="Torch device to benchmark on")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Inference dtype for the runtime",
    )
    parser.add_argument("--max-batch-size", type=int, default=16, help="Runtime max batch size")
    parser.add_argument("--page-size", type=int, default=128, help="KV cache page size")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum total sequence length (prompt + decode)",
    )

    parser.add_argument("--num-prompts", type=int, default=16, help="Number of requests per round")
    parser.add_argument("--min-input-tokens", type=int, default=64, help="Minimum prompt tokens (excluding template)")
    parser.add_argument("--max-input-tokens", type=int, default=512, help="Maximum prompt tokens (excluding template)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate per request")
    parser.add_argument("--rounds", type=int, default=3, help="Benchmark repetitions")
    parser.add_argument("--seed", type=int, default=2025, help="Seed for synthetic prompt generation")
    parser.add_argument(
        "--prompt-template",
        default="Explain the implications of recent advances in artificial intelligence.",
        help="Base sentence repeated to synthesize prompts",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        type=Path,
        default=[],
        help="Path to an image file to attach to prompts (repeatable).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images (jpg/png/webp) to attach; files are cycled if fewer than prompts.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph capture",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Optional path to dump aggregate metrics as JSON",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _synthetic_prompts(
    runtime: MoondreamRuntime,
    *,
    count: int,
    min_tokens: int,
    max_tokens: int,
    template: str,
    seed: int,
    max_new_tokens: int,
    images: Optional[Sequence[pyvips.Image]] = None,
) -> List[PromptPayload]:
    tokenizer = runtime.tokenizer
    cfg = runtime.config.tokenizer
    prefix = cfg.templates["query"]["prefix"] or []
    suffix = cfg.templates["query"]["suffix"] or []
    overhead = 1 + len(prefix) + len(suffix)
    max_allowed_content = runtime.max_seq_length - max_new_tokens - overhead
    if max_allowed_content <= 0:
        raise ValueError("max_seq_length is too small for the requested generation budget")

    rng = random.Random(seed)
    snippets = [
        template,
        "Discuss the long-term impact of renewable energy adoption in urban centers.",
        "Summarize the key takeaways from the latest climate science reports.",
        "Outline considerations for deploying machine learning in healthcare settings.",
        "Describe the trade-offs between latency and accuracy in edge deployments.",
    ]

    payloads: List[PromptPayload] = []
    image_iter = cycle(images) if images else None
    skill = QuerySkill()
    for idx in range(count):
        target = rng.randint(min_tokens, max_tokens)
        target = min(target, max_allowed_content)
        base = f"Request {idx}: {rng.choice(snippets)}"
        token_ids = tokenizer.encode(base).ids
        while len(token_ids) < target:
            base += " " + rng.choice(snippets)
            token_ids = tokenizer.encode(base).ids
        if len(token_ids) > max_allowed_content:
            token_ids = token_ids[:max_allowed_content]
            base = tokenizer.decode(token_ids)

        current_image = next(image_iter) if image_iter is not None else None
        request = QueryRequest(
            question=base,
            image=current_image,
            reasoning=False,
            stream=False,
            settings=QuerySettings(temperature=0.0, top_p=1.0),
        )
        prompt_tokens = skill.build_prompt_tokens(
            runtime,
            base,
            image=current_image,
        ).to("cpu")
        payloads.append(
            PromptPayload(
                text=base,
                tokens=prompt_tokens,
                length=prompt_tokens.shape[1],
                image=current_image,
                request=request,
            )
        )

    return payloads


def _random_order(prompts: List[PromptPayload], seed: int) -> List[PromptPayload]:
    ordering = list(prompts)
    random.Random(seed).shuffle(ordering)
    return ordering


async def _run_round(
    engine: InferenceEngine,
    prompts: List[PromptPayload],
    max_new_tokens: int,
) -> tuple[float, List[int], List[int], List[float], List[float], List[float]]:
    start = time.perf_counter()
    tasks = [
        engine.submit(
            payload.text,
            max_new_tokens=max_new_tokens,
            prompt_tokens=payload.tokens.clone(),
            image=payload.image,
            temperature=payload.request.settings.temperature,
            top_p=payload.request.settings.top_p,
            skill="query",
            skill_context=payload.request,
        )
        for payload in prompts
    ]
    results = await asyncio.gather(*tasks)
    if engine.runtime.device.type == "cuda":
        torch.cuda.synchronize(engine.runtime.device)
    elapsed = time.perf_counter() - start
    decode_lengths = [res.metrics.decode_tokens for res in results]
    prompt_lengths = [res.metrics.prompt_tokens for res in results]
    processing_latencies = [res.metrics.processing_latency_s for res in results]
    ttfts = [res.metrics.ttft_s for res in results]
    decode_latencies = [res.metrics.decode_latency_s for res in results]
    return elapsed, decode_lengths, prompt_lengths, processing_latencies, ttfts, decode_latencies


async def _async_main(args: argparse.Namespace) -> None:
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

    engine = await InferenceEngine.create(runtime_cfg)
    runtime = engine.runtime

    images: List[pyvips.Image] = []
    image_paths: List[Path] = []
    if args.image_dir:
        if not args.image_dir.exists():
            raise FileNotFoundError(f"Image directory {args.image_dir} does not exist")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend(sorted(args.image_dir.glob(ext)))
    if args.images:
        image_paths.extend(args.images)
    if image_paths:
        for path in image_paths:
            try:
                vips_image = pyvips.Image.new_from_file(str(path), access="sequential")
                images.append(vips_image)
            except Exception as exc:
                raise RuntimeError(f"Failed to load image at {path}: {exc}") from exc
        if not images:
            raise RuntimeError("No valid images loaded for benchmarking")

    prompt_payloads = _synthetic_prompts(
        runtime,
        count=args.num_prompts,
        min_tokens=args.min_input_tokens,
        max_tokens=args.max_input_tokens,
        template=args.prompt_template,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        images=images,
    )

    prompt_token_total = sum(p.length for p in prompt_payloads)
    print(f"Loaded runtime on {runtime.device}, cached {len(prompt_payloads)} prompts")
    print(
        f"Average prompt length: {prompt_token_total / len(prompt_payloads):.1f} tokens (including template)"
    )
    if images:
        print(f"Attached {len(images)} unique image(s); cycling if fewer than prompts")

    runs: List[dict] = []
    try:
        for round_idx in range(args.rounds):
            ordering = _random_order(prompt_payloads, args.seed + round_idx)
            (
                elapsed,
                decode_lengths,
                prompt_lengths,
                processing_latencies,
                ttfts,
                decode_latencies,
            ) = await _run_round(engine, ordering, args.max_new_tokens)

            decode_tokens = sum(decode_lengths)
            prompt_tokens = sum(prompt_lengths)
            total_tokens = prompt_tokens + decode_tokens
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float("nan")
            decode_throughput = decode_tokens / elapsed if elapsed > 0 else float("nan")
            prefill_throughput = prompt_tokens / elapsed if elapsed > 0 else float("nan")

            run_metrics = {
                "round": round_idx,
                "wall_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "decode_tokens": decode_tokens,
                "total_tokens": total_tokens,
                "throughput_toks_per_s": tokens_per_sec,
                "decode_toks_per_s": decode_throughput,
                "prefill_toks_per_s": prefill_throughput,
                "avg_processing_latency_s": (
                    sum(processing_latencies) / len(processing_latencies)
                    if processing_latencies
                    else 0.0
                ),
                "avg_ttft_s": (
                    sum(ttfts) / len(ttfts) if ttfts else 0.0
                ),
                "avg_decode_latency_s": (
                    sum(decode_latencies) / len(decode_latencies)
                    if decode_latencies
                    else 0.0
                ),
            }
            runs.append(run_metrics)

            print(
                f"Round {round_idx+1}/{args.rounds}: {elapsed:.2f}s | "
                f"prefill {prompt_tokens} tok ({prefill_throughput:.1f} tok/s), "
                f"decode {decode_tokens} tok ({decode_throughput:.1f} tok/s)"
            )
    finally:
        await engine.shutdown()

    throughput_values = [run["throughput_toks_per_s"] for run in runs]
    processing_latency_values = [run["avg_processing_latency_s"] for run in runs]
    ttft_values = [run["avg_ttft_s"] for run in runs]
    decode_latency_values = [run["avg_decode_latency_s"] for run in runs]
    total_prefill_tokens = sum(run["prompt_tokens"] for run in runs)
    total_decode_tokens = sum(run["decode_tokens"] for run in runs)
    total_wall = sum(run["wall_seconds"] for run in runs)

    prefill_tok_per_s = total_prefill_tokens / total_wall if total_wall > 0 else 0.0
    decode_tok_per_s = total_decode_tokens / total_wall if total_wall > 0 else 0.0
    avg_prefill_tokens = (
        total_prefill_tokens / (len(prompt_payloads) * args.rounds)
        if prompt_payloads and args.rounds > 0
        else 0.0
    )
    avg_decode_tokens = (
        total_decode_tokens / (len(prompt_payloads) * args.rounds)
        if prompt_payloads and args.rounds > 0
        else 0.0
    )

    aggregate = {
        "mean_throughput_tok_per_s": statistics.mean(throughput_values) if throughput_values else 0.0,
        "stdev_throughput_tok_per_s": statistics.pstdev(throughput_values) if len(throughput_values) > 1 else 0.0,
        "mean_processing_latency_s": statistics.mean(processing_latency_values)
        if processing_latency_values
        else 0.0,
        "mean_ttft_s": statistics.mean(ttft_values) if ttft_values else 0.0,
        "mean_decode_latency_s": statistics.mean(decode_latency_values)
        if decode_latency_values
        else 0.0,
        "prefill_tok_per_s": prefill_tok_per_s,
        "decode_tok_per_s": decode_tok_per_s,
        "avg_prefill_tokens": avg_prefill_tokens,
        "avg_decode_tokens": avg_decode_tokens,
        "runs": runs,
    }

    print("-- Summary --")
    print(
        f"Throughput mean: {aggregate['mean_throughput_tok_per_s']:.1f} tok/s"
        f" (stdev {aggregate['stdev_throughput_tok_per_s']:.1f})"
    )
    print(
        f"Avg processing latency: {aggregate['mean_processing_latency_s']*1000:.1f} ms "
        f"(TTFT {aggregate['mean_ttft_s']*1000:.1f} ms, decode {aggregate['mean_decode_latency_s']*1000:.1f} ms)"
    )
    print(
        f"Average prefill tokens: {aggregate['avg_prefill_tokens']:.1f} | "
        f"Average decode tokens: {aggregate['avg_decode_tokens']:.1f}"
    )
    print(
        f"Prefill throughput: {aggregate['prefill_tok_per_s']:.1f} tok/s | "
        f"Decode throughput: {aggregate['decode_tok_per_s']:.1f} tok/s"
    )

    if args.export_json:
        args.export_json.write_text(json.dumps(aggregate, indent=2))
        print(f"Wrote metrics to {args.export_json}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
