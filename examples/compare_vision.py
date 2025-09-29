"""Compare Moondream reference vision stack against the Kestrel runtime."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import pyvips
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

REFERENCE_ROOT = REPO_ROOT / "external" / "moondream"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.append(str(REFERENCE_ROOT))

from moondream.torch.config import MoondreamConfig as ReferenceConfig
from moondream.torch.moondream import MoondreamModel as ReferenceModel
from moondream.torch.weights import load_weights_into_model as load_reference_weights

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.moondream.runtime import MoondreamRuntime
from kestrel.utils.image import ensure_srgb


@dataclass
class GenerationResult:
    answer: str


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return mapping[key]


def run_reference(
    weights: Path,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
) -> GenerationResult:
    config = ReferenceConfig()
    model = ReferenceModel(config, dtype=dtype)
    load_reference_weights(str(weights), model)
    model.to(device=device, dtype=dtype)
    model.eval()

    with torch.inference_mode():
        response = model.query(
            image=image,
            question=prompt,
            reasoning=False,
            stream=False,
            settings={
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        answer = response["answer"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GenerationResult(answer=answer)


def run_kestrel(
    weights: Path,
    image: pyvips.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    *,
    enable_compile: bool,
    enable_cuda_graphs: bool,
    max_seq_length: int,
) -> GenerationResult:
    paths = ModelPaths(weights=weights)
    runtime_cfg = RuntimeConfig(
        model_paths=paths,
        device=str(device),
        dtype=dtype,
        enable_compile=enable_compile,
        enable_cuda_graphs=enable_cuda_graphs,
        max_batch_size=2,
        max_seq_length=max_seq_length,
    )

    runtime = MoondreamRuntime(runtime_cfg)
    answer, _ = runtime.greedy_generate(
        prompt,
        image=image,
        max_new_tokens=max_new_tokens,
    )

    del runtime
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GenerationResult(answer=answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vision outputs between reference Moondream and Kestrel runtime")
    parser.add_argument("--weights", type=Path, required=True, help="Path to Moondream multimodal weights")
    parser.add_argument("--image", type=Path, required=True, help="Path to test image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to run on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype (bf16/fp16/fp32)")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum new tokens to sample")
    parser.add_argument("--enable-compile", action="store_true", help="Enable torch.compile for Kestrel prefill path")
    parser.add_argument("--enable-cuda-graphs", action="store_true", help="Enable CUDA graphs for Kestrel decode path")
    parser.add_argument(
        "--kestrel-max-seq-length",
        type=int,
        default=2048,
        help="Override max sequence length for the Kestrel runtime (default: 2048)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for reference runs (default: 0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for reference runs (default: 1.0)")
    parser.add_argument(
        "--mode",
        choices=["reference", "kestrel"],
        required=True,
        help="Which implementation to run. Invoke twice (once per mode) to compare outputs without loading both models concurrently.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    if args.mode == "reference":
        image = Image.open(args.image).convert("RGB")
        result = run_reference(
            weights=args.weights,
            image=image,
            prompt=args.prompt,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("Reference answer:")
        print(result.answer)
    else:
        vips_image = ensure_srgb(
            pyvips.Image.new_from_file(str(args.image), access="sequential")
        )
        result = run_kestrel(
            weights=args.weights,
            image=vips_image,
            prompt=args.prompt,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            enable_compile=args.enable_compile,
            enable_cuda_graphs=args.enable_cuda_graphs,
            max_seq_length=args.kestrel_max_seq_length,
        )
        print("Kestrel answer:")
        print(result.answer)


if __name__ == "__main__":
    main()
