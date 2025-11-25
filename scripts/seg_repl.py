import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.refiner import render_svg_to_mask, svg_from_path
from kestrel.utils.image import ensure_srgb


DEFAULT_MAX_TOKENS = 768


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


def _bbox_minmax_to_cxcywh(bbox: dict) -> list[float]:
    x_min = float(bbox["x_min"])
    x_max = float(bbox["x_max"])
    y_min = float(bbox["y_min"])
    y_max = float(bbox["y_max"])
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return [cx, cy, w, h]


def _render_overlay(
    path: str, bbox: dict, width: int, height: int, np_image: np.ndarray, suffix: str
):
    cxcywh = _bbox_minmax_to_cxcywh(bbox)
    svg_full = svg_from_path(path, width, height, cxcywh)
    mask = render_svg_to_mask(svg_full, width, height)
    if not mask.any():
        raise RuntimeError("Rendered mask is empty")
    overlay = _build_overlay(np_image, mask)
    base = Path(suffix)
    overlay_path = str(base.with_suffix("")) + "_overlay.png"
    svg_path = str(base.with_suffix("")) + "_mask.svg"
    mask_path = str(base.with_suffix("")) + "_mask.png"
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_full)
    return overlay_path, svg_path, mask_path


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
        max_batch_size=args.max_batch_size,
        page_size=128,
        max_seq_length=args.max_seq_length,
        enable_cuda_graphs=not args.disable_cuda_graphs,
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

            try:
                pil_image = Image.open(image_path).convert("RGB")
                np_image = np.array(pil_image)
                height, width = np_image.shape[:2]

                result = await engine.segment(
                    image=np_image,
                    object=prompt,
                    settings={
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": args.max_tokens,
                    },
                )
                segment = (result.output.get("segments") or [{}])[0]

                svg_path = segment.get("svg_path") or ""
                bbox = segment.get("bbox")
                coarse_path = segment.get("coarse_path")
                coarse_bbox = segment.get("coarse_bbox")
                if not svg_path or not bbox:
                    raise RuntimeError("Missing svg_path or bbox in segment output")

                refined_base = Path(image_path).with_suffix("")
                overlay_refined, svg_refined, mask_refined = _render_overlay(
                    svg_path,
                    bbox,
                    width,
                    height,
                    np_image,
                    str(refined_base) + "_refined",
                )
                print(f"Saved refined overlay: {overlay_refined}")
                print(f"Saved refined SVG: {svg_refined}")
                print(f"Saved refined mask: {mask_refined}")

                if coarse_path and coarse_bbox:
                    overlay_coarse, svg_coarse, mask_coarse = _render_overlay(
                        coarse_path,
                        coarse_bbox,
                        width,
                        height,
                        np_image,
                        str(refined_base) + "_coarse",
                    )
                    print(f"Saved coarse overlay: {overlay_coarse}")
                    print(f"Saved coarse SVG: {svg_coarse}")
                    print(f"Saved coarse mask: {mask_coarse}")
            except Exception as exc:
                print(f"Error: {exc}")
                continue
    finally:
        await engine.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Segmentation REPL for HF SAM-HQ refiner output"
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to text weights file"
    )
    parser.add_argument("--config", type=Path, help="Optional model config JSON")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or path")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max segmentation tokens",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=2, help="Max batch size (>=2 for runtime)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=131072,
        help="Maximum total sequence length",
    )
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
