import argparse
import asyncio
import collections
import contextlib
import io
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "`datasets` is required for the ChartQA evaluation. "
        "Install optional extras via `pip install kestrel[eval]` or "
        "`uv run --extra eval ...` before running this script."
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "`tqdm` is required for ChartQA evaluation progress reporting. "
        "Install optional extras via `pip install kestrel[eval]` or "
        "`uv run --extra eval ...` before running this script."
    ) from exc

import torch

# Ensure repo root (containing the ``kestrel`` package) is importable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import EngineMetrics, InferenceEngine

try:
    import pyvips  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "pyvips is required to run the ChartQA evaluation. "
        "Install pyvips and its dependencies before executing this script."
    ) from exc

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


def pil_to_pyvips(image: Any) -> pyvips.Image:
    """Convert a PIL.Image or pyvips.Image to pyvips.Image."""
    if isinstance(image, pyvips.Image):
        return image

    try:
        from PIL import Image as PILImage  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Pillow is required to convert dataset images for evaluation."
        ) from exc

    if not isinstance(image, PILImage.Image):
        raise TypeError(
            "Expected the dataset to yield PIL.Image or pyvips.Image objects."
        )

    mode = image.mode
    if mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    elif mode == "RGBA":
        image = image.convert("RGB")

    width, height = image.size
    bands = len(image.getbands())
    data = image.tobytes()

    return pyvips.Image.new_from_memory(data, width, height, bands, format="uchar")


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
        model_paths=ModelPaths(weights=cfg.weights.expanduser()),
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=cfg.max_batch_size,
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
    image: pyvips.Image,
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

    def record_metrics(metrics: EngineMetrics) -> None:
        nonlocal total_input_tokens, total_output_tokens
        nonlocal total_prefill_ms, total_decode_ms, request_count
        total_input_tokens += metrics.input_tokens
        total_output_tokens += metrics.output_tokens
        total_prefill_ms += metrics.prefill_time_ms
        total_decode_ms += metrics.decode_time_ms
        request_count += 1

    async def evaluate_single(
        row_idx: int,
        qa_idx: int,
        image: pyvips.Image,
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

        return row_idx, qa_idx, result_entry, is_correct, source == "human"

    tasks: List[asyncio.Task[Tuple[int, int, Dict[str, Any], bool, bool]]] = []
    for row_idx, row in enumerate(rows):
        image = pil_to_pyvips(row["image"])
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
        await engine.shutdown()

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
        },
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
    )
    return cfg


def print_results(results: Dict[str, Any]) -> None:
    print("\nChartQA Evaluation Results")
    print(f"Total Accuracy: {results['acc']:.2f}% ({results['correct']} / {results['total']})")
    print(
        f"Human Accuracy: {results['human_acc']:.2f}% "
        f"({results['human_correct']} / {results['human_total']})"
    )

    usage = results.get("token_usage")
    if usage:
        request_count = usage.get("request_count", 0)
        total_input_tokens = usage.get("total_input_tokens", 0)
        total_output_tokens = usage.get("total_output_tokens", 0)
        total_prefill_ms = usage.get("total_prefill_ms", 0.0)
        total_decode_ms = usage.get("total_decode_ms", 0.0)

        avg_input_tokens = (
            total_input_tokens / request_count if request_count else 0.0
        )
        avg_output_tokens = (
            total_output_tokens / request_count if request_count else 0.0
        )
        total_prefill_s = total_prefill_ms / 1000.0
        total_decode_s = total_decode_ms / 1000.0
        aggregate_prefill = (
            total_input_tokens / total_prefill_s if total_prefill_s > 0 else 0.0
        )
        aggregate_decode = (
            total_output_tokens / total_decode_s if total_decode_s > 0 else 0.0
        )

        print(f"Avg Input Tokens / Request: {avg_input_tokens:.2f}")
        print(f"Avg Output Tokens / Request: {avg_output_tokens:.2f}")
        print(f"Aggregate Prefill Throughput: {aggregate_prefill:.2f} tok/s")
        print(f"Aggregate Decode Throughput: {aggregate_decode:.2f} tok/s")


async def async_main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = parse_args()
    results = await eval_chartqa(cfg)
    print_results(results)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
