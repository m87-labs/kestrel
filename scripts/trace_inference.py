import argparse
import asyncio
import sys
import pyvips
import torch

from pathlib import Path
from typing import Any
from datasets import load_dataset

# Ensure repo root is importable when running via python scripts/...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine

PROMPT_PREFIX = (
    "Analyze the chart carefully, consider both visual features and data values, "
    "and provide a precise answer without any additional explanation or formatting. "
)
MAX_SAMPLES = 8


def pil_to_pyvips(image: Any) -> pyvips.Image:
    """Convert chart images from the dataset to pyvips for the engine."""
    if isinstance(image, pyvips.Image):
        return image

    try:
        from PIL import Image as PILImage  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Pillow is required to convert dataset images for evaluation."
        ) from exc

    if not isinstance(image, PILImage.Image):
        raise TypeError("ChartQA samples must provide PIL.Image or pyvips.Image objects.")

    converted = image
    if converted.mode not in ("RGB", "RGBA"):
        converted = converted.convert("RGB")
    elif converted.mode == "RGBA":
        converted = converted.convert("RGB")

    width, height = converted.size
    bands = len(converted.getbands())
    data = converted.tobytes()
    return pyvips.Image.new_from_memory(data, width, height, bands, format="uchar")


async def run(weights: Path, split: str, temperature: float) -> None:
    dataset = load_dataset("vikhyatk/chartqa", split=split)
    runtime_cfg = RuntimeConfig(
        model_paths=ModelPaths(weights=weights.expanduser()),
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=4,
    )
    engine = await InferenceEngine.create(runtime_cfg)

    processed = 0
    for row_idx, row in enumerate(dataset):
        if processed >= MAX_SAMPLES:
            break
        image = pil_to_pyvips(row["image"])
        for qa_idx, qa in enumerate(row.get("qa", [])):
            if processed >= MAX_SAMPLES:
                break
            question = qa["question"].strip()
            prompt = f"{PROMPT_PREFIX}{question}"
            response = await engine.query(
                image=image,
                question=prompt,
                reasoning=True,
                settings={
                    "temperature": max(temperature, 0.0),
                },
            )
            model_answer = str(response.output.get("answer", "")).strip()
            reasoning_output = response.output.get("reasoning")
            reasoning_text = None
            if isinstance(reasoning_output, dict):
                reasoning_text = reasoning_output.get("text")
            elif isinstance(reasoning_output, str):
                reasoning_text = reasoning_output.strip()
            print(f"[{processed + 1}] Chart #{row_idx} QA #{qa_idx}")
            print(f"Question : {question}")
            print(f"Ground Truth : {qa['answer']}")
            print(f"Model Answer : {model_answer}")
            if reasoning_text:
                print(f"Reasoning : {reasoning_text}")
            print("-" * 40)
            processed += 1

    await engine.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal ChartQA runner: load weights and sample the first 5 items."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the Kestrel weights checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="ChartQA split to sample from (default: test).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    await run(args.weights, args.split, args.temperature)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
