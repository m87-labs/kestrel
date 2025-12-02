import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.seg_refiner import svg_from_path, render_svg_to_soft_mask


@dataclass
class REPLConfig:
    kestrel_weights: Path
    head_weights: Optional[Path]
    refiner: str
    refiner_iters: int
    device: str


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) +
        np.array([0, 255, 0]) * alpha
    ).astype(np.uint8)
    return overlay


async def segment_and_visualize(
    engine: InferenceEngine,
    image_path: str,
    text_prompt: str,
    output_path: str,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)

    response = await engine.segment(
        image=np_image,
        object=text_prompt,
        settings={"temperature": 0.0, "max_tokens": 2000},
    )

    segment = response.output["segments"][0]
    svg_path = segment["svg_path"]
    bbox_dict = segment["bbox"]

    cx = (bbox_dict["x_min"] + bbox_dict["x_max"]) / 2.0
    cy = (bbox_dict["y_min"] + bbox_dict["y_max"]) / 2.0
    w = bbox_dict["x_max"] - bbox_dict["x_min"]
    h = bbox_dict["y_max"] - bbox_dict["y_min"]
    bbox_cxcywh = [cx, cy, w, h]

    img_h, img_w = np_image.shape[:2]
    full_svg = svg_from_path(svg_path, img_w, img_h, bbox_cxcywh)
    soft_mask = render_svg_to_soft_mask(full_svg, img_w, img_h, scale=2)
    pred_mask = (soft_mask > 0.5).astype(np.uint8)

    overlay = create_overlay(np_image, pred_mask)

    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "bbox": bbox_dict,
        "token_count": len(segment["token_ids"]),
        "output_path": output_path,
    }


async def init_engine(config: REPLConfig) -> InferenceEngine:
    runtime_cfg = RuntimeConfig(
        model_paths=ModelPaths(
            weights=config.kestrel_weights,
            head_refiner_weights=config.head_weights,
        ),
        device=config.device,
        dtype=torch.bfloat16,
        max_batch_size=2,
        enable_cuda_graphs=False,
        use_head_refiner=(config.refiner == "head"),
    )

    torch.set_float32_matmul_precision("high")
    return await InferenceEngine.create(runtime_cfg)


async def repl_loop(engine: InferenceEngine, config: REPLConfig):
    print(f"\nKestrel Segmentation REPL")
    print(f"Refiner: {config.refiner} (iters={config.refiner_iters})")
    print(f"Press Ctrl+C to exit\n")

    iteration = 0

    while True:
        iteration += 1
        print(f"=== Iteration {iteration} ===")

        image_path = input("Image path: ").strip()
        if not image_path:
            continue

        text_prompt = input("Text prompt: ").strip()
        if not text_prompt:
            continue

        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base}_seg_{config.refiner}_{iteration}.png"

        print(f"\nProcessing: {image_path}")
        print(f"Prompt: {text_prompt}")

        result = await segment_and_visualize(engine, image_path, text_prompt, output_path)

        print(f"✓ Saved: {result['output_path']}")
        print(f"  BBox: {result['bbox']}")
        print(f"  Tokens: {result['token_count']}\n")


async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kestrel-weights", required=True, type=Path)
    parser.add_argument("--head-weights", type=Path)
    parser.add_argument("--refiner", choices=["head", "sam"], required=True)
    parser.add_argument("--refiner-iters", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.refiner == "head" and not args.head_weights:
        sys.exit("--head-weights required with --refiner head")

    config = REPLConfig(
        kestrel_weights=args.kestrel_weights,
        head_weights=args.head_weights,
        refiner=args.refiner,
        refiner_iters=args.refiner_iters,
        device=args.device,
    )

    print("Initializing engine...")
    engine = await init_engine(config)

    try:
        await repl_loop(engine, config)
    except KeyboardInterrupt:
        print("\n")
    finally:
        await engine.shutdown()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
