import argparse
import asyncio
import collections
import contextlib
import io
import json
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset, Image as HFImage
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "`datasets` is required for the ChartQA evaluation. "
        "Install optional extras via `pip install kestrel[eval]` or "
        "`uv run --extra eval ...` before running this script."
    ) from exc

import kestrel_native

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "`tqdm` is required for ChartQA evaluation progress reporting. "
        "Install optional extras via `pip install kestrel[eval]` or "
        "`uv run --extra eval ...` before running this script."
    ) from exc

import numpy as np
import torch

# Ensure repo root (containing the ``kestrel`` package) is importable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kestrel.config import RuntimeConfig
from kestrel.engine import EngineMetrics, InferenceEngine


# Prompt templates and generation limits derived from vixtral-train.
PREFIX = (
    "Analyze the chart carefully, consider both visual features and data values, "
    "and provide a precise answer without any additional explanation or formatting. "
)
POT_PREFIX = "Write a Python program to answer the following question: "
COT_MAX_TOKENS = 512
DEFAULT_MAX_TOKENS = 10
POT_MAX_TOKENS = 200


@dataclass(slots=True)
class EvalConfig:
    weights: Path
    max_batch_size: int
    dataset_split: str
    limit: Optional[int]
    use_pot: bool
    cot_samples: int
    temperature: float
    debug: bool
    enable_prefix_cache: bool
    dump_jsonl: Optional[Path] = None
    concurrency: Optional[int] = None


def decode_image(image_data: Dict[str, Any]) -> np.ndarray:
    """Decode image bytes using native decoder."""
    result = kestrel_native.decode_image(image_data["bytes"])
    if result is None:
        raise ValueError("Unsupported image format")
    return result


def relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> bool:
    """Return True if ``prediction`` matches ``target`` within tolerance."""

    def _to_float(text: str) -> Optional[float]:
        stripped = text.strip()
        try:
            if stripped.endswith("%"):
                return float(stripped.rstrip("%")) / 100.0
            return float(stripped)
        except ValueError:
            return None

    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float is not None:
        if target_float == 0:
            return prediction_float == 0
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change

    return prediction.strip().lower() == target.strip().lower()


def parse_maybe_list(value: Any) -> Any:
    """Attempt to parse strings that look like lists."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return value

    inner = text[1:-1].strip()
    if not inner:
        return []

    elements: List[Any] = []
    for item in inner.split(","):
        token = item.strip()
        if not token:
            continue
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            elements.append(token[1:-1])
            continue
        try:
            elements.append(int(token))
            continue
        except ValueError:
            pass
        try:
            elements.append(float(token))
            continue
        except ValueError:
            pass
        elements.append(token)
    return elements


def strip_trailing_percent(text: str) -> str:
    text = text.strip()
    if text.endswith("%"):
        return text[:-1].strip()
    return text


def execute_program_source(source: str, timeout: float = 10.0) -> str:
    """Execute model-generated Python source and capture stdout."""
    if not source.strip():
        return ""

    cleaned = source.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python") :].lstrip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].lstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    indented_body = textwrap.indent(cleaned, " " * 8)
    wrapped = textwrap.dedent(
        f"""
        import contextlib
        import io
        import sys
        import traceback


        def _main():
        {indented_body if indented_body.strip() else "        return"}


        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                _main()
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Error during execution: {{exc}}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        sys.stdout.write(buffer.getvalue())
        """
    )

    try:
        process = subprocess.run(
            ["python", "-c", wrapped],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "Execution Timeout"

    if process.returncode != 0:
        stderr = process.stderr.strip()
        return f"Execution Error: {stderr}" if stderr else "Execution Error"

    return process.stdout.strip()


async def create_engine(cfg: EvalConfig) -> InferenceEngine:
    runtime_cfg = RuntimeConfig(
        model_path=cfg.weights.expanduser(),
        max_batch_size=cfg.max_batch_size,
        enable_prefix_cache=cfg.enable_prefix_cache,
    )
    return await InferenceEngine.create(runtime_cfg)


def build_prompt_and_settings(
    question: str,
    *,
    use_pot: bool,
    cot_samples: int,
    base_temperature: float,
) -> Tuple[str, bool, int, float]:
    """Return prompt, reasoning flag, max_tokens, temperature."""
    temperature = max(base_temperature, 0.0)
    if cot_samples > 0:
        prompt = question.strip()
        reasoning = True
        max_tokens = COT_MAX_TOKENS
        if cot_samples > 1 and temperature == 0.0:
            # Preserve legacy behaviour of injecting small stochasticity for CoT@N
            temperature = 1.0
    elif use_pot:
        prompt = f"{POT_PREFIX}{question.strip()}"
        reasoning = False
        max_tokens = POT_MAX_TOKENS
    else:
        prompt = f"{PREFIX}{question.strip()}"
        reasoning = False
        max_tokens = DEFAULT_MAX_TOKENS
    return prompt, reasoning, max_tokens, temperature


def evaluate_prediction(
    answer: str,
    prediction: str,
) -> Tuple[bool, List[Any], List[Any]]:
    parsed_answer = parse_maybe_list(answer)
    parsed_prediction = parse_maybe_list(prediction)

    if (
        isinstance(parsed_answer, Sequence)
        and not isinstance(parsed_answer, (str, bytes))
        and isinstance(parsed_prediction, Sequence)
        and not isinstance(parsed_prediction, (str, bytes))
        and len(parsed_answer) == len(parsed_prediction)
    ):
        answer_list = list(parsed_answer)
        prediction_list = list(parsed_prediction)
    else:
        answer_list = [answer]
        prediction_list = [prediction]

    is_correct = all(
        relaxed_correctness(str(expected), str(actual))
        for expected, actual in zip(answer_list, prediction_list)
    )
    return is_correct, answer_list, prediction_list


async def query_engine(
    engine: InferenceEngine,
    *,
    image: np.ndarray,
    prompt: str,
    reasoning: bool,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]], EngineMetrics]:
    response = await engine.query(
        image=image,
        question=prompt,
        reasoning=reasoning,
        settings={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    output = response.output
    answer = str(output.get("answer", "")).strip()
    reasoning_output = output.get("reasoning") if reasoning else None
    reasoning_text = None
    grounding = None
    if isinstance(reasoning_output, dict):
        reasoning_text = reasoning_output.get("text")
        grounding = reasoning_output.get("grounding")
    return answer, reasoning_text, grounding, response.metrics


async def eval_chartqa(cfg: EvalConfig) -> Dict[str, Any]:
    dataset = load_dataset("vikhyatk/chartqa", split=cfg.dataset_split)
    dataset = dataset.cast_column("image", HFImage(decode=False))
    limit = cfg.limit
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))

    engine = await create_engine(cfg)
    rows = list(dataset)
    total_questions = sum(len(row["qa"]) for row in rows)

    correct = 0
    total = 0
    human_correct = 0
    human_total = 0
    per_chart_results: List[List[Optional[Dict[str, Any]]]] = [
        [None] * len(row["qa"]) for row in rows
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_prefill_ms = 0.0
    total_decode_ms = 0.0
    request_count = 0
    request_times_ms: List[float] = []  # prefill + decode per request
    cache_hit_count = 0  # requests with cached_tokens > 0
    semaphore = (
        asyncio.Semaphore(cfg.concurrency)
        if cfg.concurrency is not None and cfg.concurrency > 0
        else None
    )
    dump_handle = None
    if cfg.dump_jsonl is not None:
        dump_path = cfg.dump_jsonl.expanduser()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_handle = dump_path.open("w", encoding="utf-8")

    def record_metrics(metrics: EngineMetrics) -> None:
        nonlocal total_input_tokens, total_output_tokens
        nonlocal total_prefill_ms, total_decode_ms, request_count
        nonlocal cache_hit_count
        total_input_tokens += metrics.input_tokens
        total_output_tokens += metrics.output_tokens
        total_prefill_ms += metrics.prefill_time_ms
        total_decode_ms += metrics.decode_time_ms
        request_count += 1
        request_times_ms.append(metrics.prefill_time_ms + metrics.decode_time_ms)
        if metrics.cached_tokens > 0:
            cache_hit_count += 1

    async def evaluate_single(
        row_idx: int,
        qa_idx: int,
        image: np.ndarray,
        qa: Dict[str, Any],
    ) -> Tuple[int, int, Dict[str, Any], bool, bool]:
        if semaphore is None:
            return await _evaluate_single_impl(row_idx, qa_idx, image, qa)
        async with semaphore:
            return await _evaluate_single_impl(row_idx, qa_idx, image, qa)

    async def _evaluate_single_impl(
        row_idx: int,
        qa_idx: int,
        image: np.ndarray,
        qa: Dict[str, Any],
    ) -> Tuple[int, int, Dict[str, Any], bool, bool]:
        question = qa["question"]
        answer = qa["answer"]
        source = qa.get("source", "model")

        prompt, reasoning, max_tokens, temperature = build_prompt_and_settings(
            question,
            use_pot=cfg.use_pot,
            cot_samples=cfg.cot_samples,
            base_temperature=cfg.temperature,
        )

        chart_reasoning = None
        chart_grounding = None
        metrics_for_dump = None

        if cfg.cot_samples > 1:
            if cfg.use_pot:
                raise ValueError("POT and CoT sampling cannot be combined.")
            candidates: List[str] = []
            for idx in range(cfg.cot_samples):
                (
                    candidate_answer,
                    candidate_reasoning,
                    candidate_grounding,
                    candidate_metrics,
                ) = await query_engine(
                    engine,
                    image=image,
                    prompt=prompt,
                    reasoning=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                record_metrics(candidate_metrics)
                candidate_answer = strip_trailing_percent(candidate_answer)
                candidates.append(candidate_answer)
                if idx == 0:
                    metrics_for_dump = candidate_metrics
                    chart_reasoning = candidate_reasoning
                    chart_grounding = candidate_grounding
            if candidates:
                counts = collections.Counter(candidates)
                model_answer = counts.most_common(1)[0][0]
            else:
                model_answer = ""
        else:
            (
                model_answer,
                chart_reasoning,
                chart_grounding,
                metrics,
            ) = await query_engine(
                engine,
                image=image,
                prompt=prompt,
                reasoning=cfg.cot_samples > 0,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            record_metrics(metrics)
            metrics_for_dump = metrics
            model_answer = strip_trailing_percent(model_answer)

        if cfg.use_pot:
            executed = execute_program_source(model_answer)
            executed = strip_trailing_percent(executed)
            if executed.lower() == "true":
                executed = "Yes"
            elif executed.lower() == "false":
                executed = "No"
            model_answer = executed

        is_correct, answer_list, prediction_list = evaluate_prediction(
            answer, model_answer
        )

        if cfg.debug:
            debug_parts = [
                f"Question: {question}",
                f"Ground Truth: {answer}",
                f"Model Answer: {model_answer}",
                f"Source: {source}",
                f"Is Correct: {is_correct}",
            ]
            if chart_reasoning:
                debug_parts.append(f"Reasoning: {chart_reasoning}")
            if cfg.use_pot:
                debug_parts.append("Mode: Program-of-Thought")
            elif cfg.cot_samples > 0:
                debug_parts.append(
                    f"Mode: Chain-of-Thought (samples={cfg.cot_samples})"
                )
            else:
                debug_parts.append("Mode: Direct")
            print("\n".join(debug_parts))
            print("-" * 40)

        result_entry = {
            "question": question,
            "ground_truth": answer_list,
            "model_answer": prediction_list,
            "reasoning": chart_reasoning,
            "grounding": chart_grounding,
            "is_correct": is_correct,
            "source": source,
        }

        if dump_handle is not None:
            metrics_dump = None
            if metrics_for_dump is not None:
                metrics_dump = {
                    "input_tokens": metrics_for_dump.input_tokens,
                    "output_tokens": metrics_for_dump.output_tokens,
                    "prefill_time_ms": metrics_for_dump.prefill_time_ms,
                    "decode_time_ms": metrics_for_dump.decode_time_ms,
                    "ttft_ms": metrics_for_dump.ttft_ms,
                    "cached_tokens": metrics_for_dump.cached_tokens,
                }
            dump_handle.write(
                json.dumps(
                    {
                        "row_idx": row_idx,
                        "qa_idx": qa_idx,
                        "question": question,
                        "ground_truth_raw": answer,
                        "model_answer_raw": model_answer,
                        "is_correct": is_correct,
                        "source": source,
                        "metrics": metrics_dump,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            dump_handle.flush()

        return row_idx, qa_idx, result_entry, is_correct, source == "human"

    # Run a warmup query to exclude startup time from measurements
    if rows:
        warmup_image = decode_image(rows[0]["image"])
        warmup_qa = rows[0]["qa"][0]
        warmup_prompt, warmup_reasoning, warmup_max_tokens, warmup_temp = build_prompt_and_settings(
            warmup_qa["question"],
            use_pot=cfg.use_pot,
            cot_samples=cfg.cot_samples,
            base_temperature=cfg.temperature,
        )
        for _ in range(2):
            await query_engine(
                engine,
                image=warmup_image,
                prompt=warmup_prompt,
                reasoning=warmup_reasoning,
                max_tokens=warmup_max_tokens,
                temperature=warmup_temp,
            )

    tasks: List[asyncio.Task[Tuple[int, int, Dict[str, Any], bool, bool]]] = []
    start_time = time.perf_counter()
    for row_idx, row in enumerate(rows):
        image = decode_image(row["image"])
        for qa_idx, qa in enumerate(row["qa"]):
            tasks.append(
                asyncio.create_task(evaluate_single(row_idx, qa_idx, image, qa))
            )

    progress = tqdm(total=total_questions, desc="ChartQA", disable=cfg.debug)
    try:
        for task in asyncio.as_completed(tasks):
            row_idx, qa_idx, result_entry, is_correct, is_human = await task
            per_chart_results[row_idx][qa_idx] = result_entry

            total += 1
            if is_correct:
                correct += 1
            if is_human:
                human_total += 1
                if is_correct:
                    human_correct += 1

            progress.update(1)
    finally:
        progress.close()
        if dump_handle is not None:
            dump_handle.close()
        await engine.shutdown()

    wall_time_s = max(time.perf_counter() - start_time, 0.0)

    results = [
        [entry for entry in chart_entries if entry is not None]
        for chart_entries in per_chart_results
    ]

    total_acc = (correct / total * 100.0) if total else 0.0
    human_acc = (human_correct / human_total * 100.0) if human_total else 0.0

    return {
        "acc": total_acc,
        "human_acc": human_acc,
        "total": total,
        "correct": correct,
        "human_total": human_total,
        "human_correct": human_correct,
        "results": results,
        "token_usage": {
            "request_count": request_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_prefill_ms": total_prefill_ms,
            "total_decode_ms": total_decode_ms,
            "request_times_ms": request_times_ms,
            "cache_hit_count": cache_hit_count,
        },
        "wall_time_s": wall_time_s,
        "prefix_cache_enabled": cfg.enable_prefix_cache,
    }


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate Kestrel on the ChartQA benchmark."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("~/code/moondream/model.pt"),
        help="Path to the Kestrel weights checkpoint.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum batch size for the inference engine.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="ChartQA split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of chart examples to evaluate (defaults to full split). Pass -1 to explicitly use the entire split.",
    )
    parser.add_argument(
        "--pot",
        action="store_true",
        help="Enable Program-of-Thought prompting.",
    )
    parser.add_argument(
        "--cot",
        type=int,
        default=0,
        nargs="?",
        const=1,
        help="Enable Chain-of-Thought prompting. Provide N for CoT@N sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Base sampling temperature. Applies to all modes unless overridden.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-question debugging output.",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching (enabled by default).",
    )
    parser.add_argument(
        "--dump-jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-question results as JSONL (debugging).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Limit number of concurrent in-flight questions (0 = unlimited).",
    )
    args = parser.parse_args()

    if args.pot and args.cot:
        parser.error("Program-of-Thought and Chain-of-Thought modes are mutually exclusive.")

    limit = None if args.limit is None or args.limit < 0 else int(args.limit)

    cfg = EvalConfig(
        weights=args.weights,
        max_batch_size=args.max_batch_size,
        dataset_split=args.split,
        limit=limit,
        use_pot=bool(args.pot),
        cot_samples=int(args.cot),
        temperature=float(args.temperature),
        debug=bool(args.debug),
        enable_prefix_cache=not args.disable_prefix_cache,
        dump_jsonl=args.dump_jsonl,
        concurrency=None if int(args.concurrency) <= 0 else int(args.concurrency),
    )
    return cfg


def percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_results(results: Dict[str, Any]) -> None:
    print("\nChartQA Evaluation Results")
    print(f"Total Accuracy: {results['acc']:.2f}% ({results['correct']} / {results['total']})")
    print(
        f"Human Accuracy: {results['human_acc']:.2f}% "
        f"({results['human_correct']} / {results['human_total']})"
    )

    wall_time_s = results.get("wall_time_s", 0.0) or 0.0
    usage = results.get("token_usage")
    request_count = usage.get("request_count", 0) if usage else 0

    if usage:
        total_input_tokens = usage.get("total_input_tokens", 0)
        total_output_tokens = usage.get("total_output_tokens", 0)
        total_prefill_ms = usage.get("total_prefill_ms", 0.0)
        total_decode_ms = usage.get("total_decode_ms", 0.0)
        request_times_ms = usage.get("request_times_ms", [])

        avg_input_tokens = (
            total_input_tokens / request_count if request_count else 0.0
        )
        avg_output_tokens = (
            total_output_tokens / request_count if request_count else 0.0
        )
        print(f"\nAvg Input Tokens / Request: {avg_input_tokens:.2f}")
        print(f"Avg Output Tokens / Request: {avg_output_tokens:.2f}")
        if wall_time_s > 0:
            print(f"Prefill Throughput: {total_input_tokens / wall_time_s:.2f} tok/s")
            print(f"Decode Throughput: {total_output_tokens / wall_time_s:.2f} tok/s")
        # Prefix cache stats (only when enabled)
        if results.get("prefix_cache_enabled", False) and request_count > 0:
            cache_hit_count = usage.get("cache_hit_count", 0)
            hit_rate = cache_hit_count / request_count * 100.0
            print(f"Prefix cache hit rate: {hit_rate:.1f}%")

        if wall_time_s > 0.0 and request_count > 0:
            requests_per_second = request_count / wall_time_s
            print(f"Requests per second: {requests_per_second:.2f}")

        # Request latency stats (prefill + decode)
        if request_times_ms:
            p50_ms = percentile(request_times_ms, 50)
            p90_ms = percentile(request_times_ms, 90)
            p99_ms = percentile(request_times_ms, 99)
            print(f"\nRequest Latency (prefill + decode):")
            print(f"  P50:  {p50_ms:.2f} ms")
            print(f"  P90:  {p90_ms:.2f} ms")
            print(f"  P99:  {p99_ms:.2f} ms")


async def async_main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = parse_args()
    results = await eval_chartqa(cfg)
    print_results(results)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
