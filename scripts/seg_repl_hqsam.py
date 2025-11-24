import argparse
import asyncio
import math
from pathlib import Path

import numpy as np
import pyvips
import torch
from PIL import Image

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.utils.image import ensure_srgb, vips_to_uint8_numpy
from kestrel.utils.svg import (
    PATH_COMMANDS,
    render_svg_to_mask,
    svg_from_path,
    tokens_to_raw_path,
)


DEFAULT_MAX_TOKENS = 768
DEFAULT_VIEWBOX = 960.0


def _build_overlay(np_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    overlay = np_image.copy()
    if not mask_bool.any():
        return overlay

    fill_color = np.array([255, 128, 0], dtype=np.uint8)
    border_color = np.array([255, 64, 0], dtype=np.uint8)
    fill_alpha = 0.4

    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * (1.0 - fill_alpha)
        + fill_color.astype(np.float32) * fill_alpha
    ).astype(np.uint8)

    border = np.zeros_like(mask_bool, dtype=bool)
    border[2:, :] |= mask_bool[2:, :] & ~mask_bool[:-2, :]
    border[:-2, :] |= mask_bool[:-2, :] & ~mask_bool[2:, :]
    border[:, 2:] |= mask_bool[:, 2:] & ~mask_bool[:, :-2]
    border[:, :-2] |= mask_bool[:, :-2] & ~mask_bool[:, 2:]
    overlay[border] = border_color
    return overlay


def _path_to_viewbox(path: str, viewbox: float = DEFAULT_VIEWBOX) -> str:
    parts = []
    token = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch in PATH_COMMANDS or ch in {" ", ","}:
            if token:
                try:
                    val = float(token) * viewbox
                    parts.append(str(int(round(val))))
                except Exception:
                    parts.append(token)
                token = ""
            if ch in PATH_COMMANDS:
                parts.append(ch)
            i += 1
            continue
        if ch == "-" and token:
            try:
                val = float(token) * viewbox
                parts.append(str(int(round(val))))
            except Exception:
                parts.append(token)
            token = "-"
            i += 1
            continue
        token += ch
        i += 1
    if token:
        try:
            val = float(token) * viewbox
            parts.append(str(int(round(val))))
        except Exception:
            parts.append(token)
    return " ".join(parts)


def _make_svg(segment: dict, width: int, height: int) -> tuple[str, np.ndarray]:
    refined_path = segment.get("refined_svg_path")
    refined_bbox = segment.get("refined_bbox")
    path = refined_path or segment.get("svg_path") or ""
    bbox = refined_bbox or segment.get("bbox") or {}
    cx = float(bbox.get("x_center", 0.5))
    cy = float(bbox.get("y_center", 0.5))
    bw = float(bbox.get("width", 1.0))
    bh = float(bbox.get("height", 1.0))

    if not path:
        raw_tokens = segment.get("path_tokens") or []
        if raw_tokens:
            viewbox_path = tokens_to_raw_path(raw_tokens)
        else:
            viewbox_path = ""
    else:
        viewbox_path = _path_to_viewbox(path, DEFAULT_VIEWBOX)
    if not viewbox_path:
        return "", np.zeros((height, width), dtype=bool)
    svg_full = svg_from_path(viewbox_path, width, height, [cx, cy, bw, bh], viewbox=DEFAULT_VIEWBOX)
    mask = render_svg_to_mask(svg_full, width, height)
    return svg_full, mask


async def _run(args: argparse.Namespace) -> None:
    model_paths = ModelPaths(
        weights=args.weights,
        config_json=args.config,
        tokenizer=args.tokenizer,
    )
    runtime_cfg = RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=1,
        page_size=128,
        max_seq_length=args.max_seq_length,
        enable_cuda_graphs=not args.disable_cuda_graphs,
        enable_sam_hq_refiner=True,
        sam_hq_checkpoint=args.sam_hq_checkpoint,
        sam_hq_model_type=args.sam_hq_model_type,
        sam_hq_device=args.sam_hq_device or args.device,
        sam_hq_iters=args.sam_hq_iters,
    )
    engine = await InferenceEngine.create(runtime_cfg)
    try:
        while True:
            image_path = input("Image path (or 'quit'): ").strip()
            if not image_path:
                continue
            if image_path.lower() in {"q", "quit", "exit"}:
                break

            prompt = input("Object prompt: ").strip()
            if not prompt:
                continue

            image = pyvips.Image.new_from_file(image_path, access="sequential")
            image = ensure_srgb(image)
            width, height = image.width, image.height
            np_image = vips_to_uint8_numpy(image)

            result = await engine.segment(
                image=image,
                object=prompt,
                settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": args.max_tokens,
                },
            )
            segment = (result.output.get("segments") or [{}])[0]
            svg_full, mask = _make_svg(segment, width, height)
            if not mask.any():
                print("[sam-debug] empty mask; skipping overlay")
                continue

            overlay = _build_overlay(np_image, mask)
            out_path = str(Path(image_path).with_suffix("")) + "_overlay.png"
            Image.fromarray(overlay).save(out_path)
            svg_path = str(Path(image_path).with_suffix("")) + "_mask.svg"
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_full)
            print(f"[sam-debug] saved overlay={out_path} svg={svg_path}")
    finally:
        await engine.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kestrel segmentation REPL with HQ-SAM refiner")
    parser.add_argument("--weights", type=Path, required=True, help="Path to text weights file")
    parser.add_argument("--config", type=Path, help="Optional model config JSON")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Computation dtype",
    )
    parser.add_argument("--sam-hq-checkpoint", type=Path, required=True, help="Path to SAM HQ checkpoint")
    parser.add_argument("--sam-hq-model-type", default="vit_h", help="SAM HQ model type (e.g., vit_h)")
    parser.add_argument("--sam-hq-device", type=str, help="Device for SAM HQ (defaults to --device)")
    parser.add_argument("--sam-hq-iters", type=int, default=3, help="Iterations for SAM HQ refiner")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max segmentation tokens")
    parser.add_argument("--max-seq-length", type=int, default=131072, help="Maximum total sequence length")
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph capture",
    )
    args = parser.parse_args(argv)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
