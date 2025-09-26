"""Calibrate FP8 KV cache scales for the Moondream runtime."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import pyvips

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.models import MoondreamTextRuntime
from kestrel.utils.image import ensure_srgb


FP8_FORMAT_MAX = {
    "e4m3": 448.0,
    "e4m3fn": 448.0,
    "e5m2": 57344.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", type=Path, required=True, help="Path to the model weights file")
    parser.add_argument("--config", type=Path, help="Optional config JSON override")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda", help="Device to run calibration on")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Inference dtype",
    )
    parser.add_argument("--max-batch-size", type=int, default=16, help="Runtime max batch size")
    parser.add_argument("--page-size", type=int, default=128, help="KV cache page size")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum total sequence length")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for prefill")
    parser.add_argument("--disable-cuda-graphs", action="store_true", help="Disable CUDA graph capture")

    parser.add_argument("--prompt", action="append", dest="prompts", default=[], help="Prompt string (repeatable)")
    parser.add_argument("--prompts-file", type=Path, help="Path to a text file with one prompt per line")
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        default=[],
        type=Path,
        help="Optional image path to attach (repeatable)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to attach; files are cycled if fewer than prompts",
    )

    parser.add_argument("--max-new-tokens", type=int, default=128, help="Decode budget per prompt")
    parser.add_argument("--decode-steps", type=int, help="Override number of decode steps (defaults to max-new-tokens)")

    parser.add_argument(
        "--fp8-format",
        default="e4m3",
        choices=sorted(FP8_FORMAT_MAX.keys()),
        help="FP8 format used for quantization",
    )
    parser.add_argument(
        "--guard-factor",
        type=float,
        default=0.98,
        help="Guard band applied to FP8 dynamic range (scale = amax / (guard*fp8_max))",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Optional percentile (0-100) to report alongside maxima",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file for calibration results",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise summary of scales to stdout",
    )

    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


@dataclass
class PromptSpec:
    text: str
    image: Optional[pyvips.Image]


class KVCalibrationCollector:
    """Collect per-layer/head activation maxima from KV cache updates."""

    def __init__(
        self,
        runtime: MoondreamTextRuntime,
        *,
        percentile: Optional[float] = None,
    ) -> None:
        self.runtime = runtime
        text_cfg = runtime.config.text
        self.num_layers = text_cfg.n_layers
        self.num_kv_heads = text_cfg.n_kv_heads
        self.device = runtime.device
        self.dtype = torch.float32
        self.percentile = percentile

        self.k_head_max = torch.zeros(
            (self.num_layers, self.num_kv_heads), device=self.device, dtype=self.dtype
        )
        self.v_head_max = torch.zeros_like(self.k_head_max)
        self.k_layer_max = torch.zeros(self.num_layers, device=self.device, dtype=self.dtype)
        self.v_layer_max = torch.zeros_like(self.k_layer_max)

        self._percentile_samples = percentile is not None
        if self._percentile_samples:
            self._k_samples: List[List[torch.Tensor]] = [list() for _ in range(self.num_layers)]
            self._v_samples: List[List[torch.Tensor]] = [list() for _ in range(self.num_layers)]
        else:
            self._k_samples = []
            self._v_samples = []

        self._orig_methods: List[Tuple[object, object]] = []

    def __enter__(self) -> "KVCalibrationCollector":
        for layer_idx, cache in enumerate(self.runtime.layer_caches):
            orig = cache.update

            def wrapped(cache_self, pos_ids, k_val, v_val, *, _layer_idx=layer_idx, _collector=self, _orig=orig):
                _collector._record(_layer_idx, k_val, v_val)
                return _orig(pos_ids, k_val, v_val)

            cache.update = wrapped.__get__(cache, cache.__class__)  # type: ignore[attr-defined]
            self._orig_methods.append((cache, orig))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for cache, orig in self._orig_methods:
            cache.update = orig  # type: ignore[attr-defined]
        self._orig_methods.clear()

    def _record(self, layer_idx: int, k_val: torch.Tensor, v_val: torch.Tensor) -> None:
        k_abs = k_val.to(self.dtype).abs()
        v_abs = v_val.to(self.dtype).abs()

        k_head = torch.amax(k_abs, dim=(0, 2, 3))
        v_head = torch.amax(v_abs, dim=(0, 2, 3))

        self.k_head_max[layer_idx] = torch.maximum(self.k_head_max[layer_idx], k_head)
        self.v_head_max[layer_idx] = torch.maximum(self.v_head_max[layer_idx], v_head)
        self.k_layer_max[layer_idx] = torch.maximum(
            self.k_layer_max[layer_idx], k_head.max()
        )
        self.v_layer_max[layer_idx] = torch.maximum(
            self.v_layer_max[layer_idx], v_head.max()
        )

        if self._percentile_samples:
            self._k_samples[layer_idx].append(k_head.detach().cpu())
            self._v_samples[layer_idx].append(v_head.detach().cpu())

    def finalize(self) -> dict:
        result: dict = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "k_head_amax": self.k_head_max.detach().cpu().tolist(),
            "v_head_amax": self.v_head_max.detach().cpu().tolist(),
            "k_layer_amax": self.k_layer_max.detach().cpu().tolist(),
            "v_layer_amax": self.v_layer_max.detach().cpu().tolist(),
        }
        if self._percentile_samples and self.percentile is not None:
            q = self.percentile
            if not (0.0 < q < 100.0):
                raise ValueError("percentile must be between 0 and 100")
            q_value = q / 100.0
            k_quantiles = []
            v_quantiles = []
            for layer in range(self.num_layers):
                if not self._k_samples[layer]:
                    k_quantiles.append([0.0] * self.num_kv_heads)
                    v_quantiles.append([0.0] * self.num_kv_heads)
                    continue
                stacked_k = torch.stack(self._k_samples[layer], dim=0)
                stacked_v = torch.stack(self._v_samples[layer], dim=0)
                k_quant = torch.quantile(stacked_k, q_value, dim=0)
                v_quant = torch.quantile(stacked_v, q_value, dim=0)
                k_quantiles.append(k_quant.tolist())
                v_quantiles.append(v_quant.tolist())
            result["k_head_percentile"] = k_quantiles
            result["v_head_percentile"] = v_quantiles
        return result


def _load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts_file:
        lines = args.prompts_file.read_text(encoding="utf-8").splitlines()
        prompts.extend(line.strip() for line in lines if line.strip())
    prompts.extend(p for p in args.prompts if p)
    deduped = [p for idx, p in enumerate(prompts) if p and p not in prompts[:idx]]
    if not deduped:
        raise ValueError("No prompts provided; use --prompt or --prompts-file")
    return deduped


def _collect_images(args: argparse.Namespace) -> List[pyvips.Image]:
    image_paths: List[Path] = []
    if args.image_dir:
        if not args.image_dir.is_dir():
            raise ValueError(f"Image directory {args.image_dir} does not exist")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(sorted(args.image_dir.glob(ext)))
    image_paths.extend(args.images)
    images: List[pyvips.Image] = []
    for path in image_paths:
        if not path.exists():
            raise ValueError(f"Image path {path} does not exist")
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        # pyvips streams may require a full copy before random access; copy once for safety.
        img = img.copy_memory()
        images.append(ensure_srgb(img))
    return images


def _pair_prompts(prompts: Sequence[str], images: Sequence[pyvips.Image]) -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for idx, prompt in enumerate(prompts):
        image = images[idx % len(images)] if images else None
        specs.append(PromptSpec(text=prompt, image=image))
    return specs


def _prepare_runtime(args: argparse.Namespace) -> MoondreamTextRuntime:
    dtype = _resolve_dtype(args.dtype)
    model_paths = ModelPaths(
        weights=args.weights,
        config_json=args.config,
        tokenizer=args.tokenizer,
    )
    cfg = RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=dtype,
        max_batch_size=args.max_batch_size,
        page_size=args.page_size,
        max_seq_length=args.max_seq_length,
        enable_compile=not args.disable_compile,
        enable_cuda_graphs=not args.disable_cuda_graphs,
    )
    return MoondreamTextRuntime(cfg)


def _compute_scales(
    amax: torch.Tensor,
    *,
    fp8_max: float,
    guard: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    denom = max(fp8_max * guard, eps)
    scale = (amax / denom).clamp(min=eps)
    inv_scale = scale.reciprocal()
    return scale, inv_scale


def main() -> None:
    args = _parse_args()

    prompts = _load_prompts(args)
    images = _collect_images(args)
    prompt_specs = _pair_prompts(prompts, images)

    runtime = _prepare_runtime(args)

    decode_steps = args.decode_steps or args.max_new_tokens
    fp8_format = args.fp8_format.lower()
    fp8_max = FP8_FORMAT_MAX[fp8_format]
    guard = args.guard_factor

    total_prefill_tokens = 0
    total_decode_tokens = 0

    percentile = args.percentile
    collector = KVCalibrationCollector(runtime, percentile=percentile)

    with collector:
        with torch.inference_mode():
            for spec in prompt_specs:
                state = None
                try:
                    state, logits = runtime.start_sequence(
                        question=spec.text,
                        image=spec.image,
                        max_new_tokens=args.max_new_tokens,
                    )
                    total_prefill_tokens += state.prompt_length or 0
                    next_token = torch.argmax(logits, dim=-1)

                    steps = min(decode_steps, state.remaining_new_tokens())
                    for _ in range(steps):
                        logits = runtime.decode(state, next_token)
                        next_token = torch.argmax(logits, dim=-1)
                        total_decode_tokens += 1
                finally:
                    if state is not None:
                        runtime.release_sequence(state)

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)

    stats = collector.finalize()

    k_head = torch.tensor(stats["k_head_amax"])
    v_head = torch.tensor(stats["v_head_amax"])
    k_layer = torch.tensor(stats["k_layer_amax"])
    v_layer = torch.tensor(stats["v_layer_amax"])

    k_head_scale, k_head_inv = _compute_scales(k_head, fp8_max=fp8_max, guard=guard)
    v_head_scale, v_head_inv = _compute_scales(v_head, fp8_max=fp8_max, guard=guard)
    k_layer_scale, k_layer_inv = _compute_scales(k_layer, fp8_max=fp8_max, guard=guard)
    v_layer_scale, v_layer_inv = _compute_scales(v_layer, fp8_max=fp8_max, guard=guard)

    result = {
        "metadata": {
            "prompts": len(prompt_specs),
            "images": sum(1 for spec in prompt_specs if spec.image is not None),
            "prefill_tokens": total_prefill_tokens,
            "decode_tokens": total_decode_tokens,
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "fp8_format": fp8_format,
            "fp8_max": fp8_max,
            "guard_factor": guard,
            "max_batch_size": runtime.max_batch_size,
            "page_size": runtime.page_size,
            "max_seq_length": runtime.max_seq_length,
        },
        "k": {
            "per_head_amax": stats["k_head_amax"],
            "per_head_scale": k_head_scale.tolist(),
            "per_head_inv_scale": k_head_inv.tolist(),
            "per_layer_amax": stats["k_layer_amax"],
            "per_layer_scale": k_layer_scale.tolist(),
            "per_layer_inv_scale": k_layer_inv.tolist(),
        },
        "v": {
            "per_head_amax": stats["v_head_amax"],
            "per_head_scale": v_head_scale.tolist(),
            "per_head_inv_scale": v_head_inv.tolist(),
            "per_layer_amax": stats["v_layer_amax"],
            "per_layer_scale": v_layer_scale.tolist(),
            "per_layer_inv_scale": v_layer_inv.tolist(),
        },
    }

    if "k_head_percentile" in stats:
        result["k"]["per_head_percentile"] = stats["k_head_percentile"]
        result["v"]["per_head_percentile"] = stats["v_head_percentile"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.summary:
        max_k_head = torch.max(k_head)
        max_v_head = torch.max(v_head)
        print("=== Calibration Summary ===")
        print(f"Prompts: {len(prompt_specs)} | Prefill tokens: {total_prefill_tokens} | Decode tokens: {total_decode_tokens}")
        print(f"Max K head amax: {max_k_head.item():.4f} | scale: {(max_k_head / (fp8_max * guard)).item():.6f}")
        print(f"Max V head amax: {max_v_head.item():.4f} | scale: {(max_v_head / (fp8_max * guard)).item():.6f}")


if __name__ == "__main__":
    main()
