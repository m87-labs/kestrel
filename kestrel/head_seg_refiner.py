"""Head-based mask refinement and SVG conversion utilities."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .seg_refiner import bitmap_to_path, svg_from_path, render_svg_to_soft_mask, _clean_mask, _expand_bbox, _paste_mask, _ensure_numpy_rgb
from .moondream.vision import vision_encoder, vision_encoder_multiscale, create_patches


def head_refine(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    head_refiner,
    vision_module,
    vision_config,
    iters: int = 6,
) -> np.ndarray:
    """
    Head-based iterative mask refinement.

    Args:
        image: RGB numpy array (H, W, 3), uint8.
        coarse_mask: Binary mask (H, W), uint8.
        head_refiner: HeadRefiner instance.
        vision_module: runtime.model.vision (ModuleDict).
        vision_config: runtime.config.vision (VisionConfig).
        iters: Number of refinement iterations.

    Returns:
        Refined binary mask (H, W), uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if coarse_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

    device = next(head_refiner.parameters()).device
    img_h, img_w = image.shape[:2]

    if coarse_mask.shape != (img_h, img_w):
        coarse_mask = cv2.resize(coarse_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Resize to 378x378
    img_resized = cv2.resize(image, (378, 378), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(coarse_mask, (378, 378), interpolation=cv2.INTER_NEAREST)

    # Normalize image
    img_norm = torch.from_numpy(img_resized).float().to(device)
    img_norm = img_norm.permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    img_norm = (img_norm - mean) / std
    img_norm = img_norm.to(torch.bfloat16)

    # Convert mask to [0, 1] tensor (mask is already binary 0/1)
    mask_t = torch.from_numpy(mask_resized).float().to(device).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    # Get vision features using vision_encoder function
    with torch.no_grad():
        features = vision_encoder(img_norm, vision_module, vision_config)  # (1, 729, 1152)
        refined_mask = head_refiner(features, mask_t, n_iters=iters)  # (1, 1, 378, 378)

    # Resize back to original size
    refined_mask_np = refined_mask.squeeze(0).squeeze(0).float().cpu().numpy()
    refined_mask_full = cv2.resize(refined_mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    refined_mask_binary = (refined_mask_full > 0.5).astype(np.uint8)

    return refined_mask_binary


def refine_segmentation_with_head(
    image,
    svg_path: str,
    bbox: dict,
    head_refiner,
    vision_module,
    vision_config,
    iters: int = 6,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Refine a coarse SVG segmentation using head refiner.

    Args:
        image: RGB image (numpy array or pyvips Image).
        svg_path: SVG path string from model output.
        bbox: Bbox dict with x_min, y_min, x_max, y_max (normalized 0-1).
        head_refiner: HeadRefiner instance.
        vision_module: runtime.model.vision (ModuleDict).
        vision_config: runtime.config.vision (VisionConfig).
        iters: Number of refinement iterations.

    Returns:
        (refined_svg_path, refined_bbox) or (None, None) on failure.
    """
    if head_refiner is None:
        print("[head_refiner] head_refiner is None, skipping refinement")
        return None, None

    try:
        image = _ensure_numpy_rgb(image)
        img_h, img_w = image.shape[:2]

        # Convert bbox from minmax to cxcywh for SVG rendering
        cx = (bbox["x_min"] + bbox["x_max"]) / 2
        cy = (bbox["y_min"] + bbox["y_max"]) / 2
        bw = bbox["x_max"] - bbox["x_min"]
        bh = bbox["y_max"] - bbox["y_min"]
        bbox_cxcywh = [cx, cy, bw, bh]

        # Render coarse SVG to mask
        full_svg = svg_from_path(svg_path, img_w, img_h, bbox_cxcywh)
        coarse_soft = render_svg_to_soft_mask(full_svg, img_w, img_h)
        coarse_mask = (coarse_soft > 0.5).astype(np.uint8)

        if coarse_mask.sum() == 0:
            print("[head_refiner] coarse_mask is empty, skipping refinement")
            return None, None

        # Crop around bbox with margin
        crop_xyxy = _expand_bbox(bbox, img_w, img_h, margin=0.25)
        x1, y1, x2, y2 = crop_xyxy
        crop_img = image[y1:y2, x1:x2, :]
        crop_mask = coarse_mask[y1:y2, x1:x2]

        if crop_mask.sum() == 0:
            print("[head_refiner] crop_mask is empty, skipping refinement")
            return None, None

        # Run head refinement
        refined_crop = head_refine(crop_img, crop_mask, head_refiner, vision_module, vision_config, iters=iters)

        # Paste back to full size and clean up
        refined_mask = _paste_mask(img_h, img_w, refined_crop, crop_xyxy)
        refined_mask = _clean_mask(refined_mask).astype(np.uint8)

        if refined_mask.sum() == 0:
            print("[head_refiner] refined_mask is empty, skipping refinement")
            return None, None

        result = bitmap_to_path(refined_mask)
        if result is None:
            print("[head_refiner] bitmap_to_path failed, skipping refinement")
            return None, None

        refined_path, refined_bbox = result
        return refined_path, refined_bbox

    except Exception:
        import traceback
        traceback.print_exc()
        return None, None


def hqsam_head_refine(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    hqsam_head_refiner,
    vision_module,
    vision_config,
    iters: int = 6,
) -> np.ndarray:
    """HQ-SAM head-based iterative mask refinement with multi-scale features."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if coarse_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

    device = next(hqsam_head_refiner.parameters()).device
    img_h, img_w = image.shape[:2]

    if coarse_mask.shape != (img_h, img_w):
        coarse_mask = cv2.resize(coarse_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    img_resized = cv2.resize(image, (378, 378), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(coarse_mask, (378, 378), interpolation=cv2.INTER_NEAREST)

    img_norm = torch.from_numpy(img_resized).float().to(device)
    img_norm = img_norm.permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    img_norm = (img_norm - mean) / std
    img_norm = img_norm.to(torch.bfloat16)

    mask_t = torch.from_numpy(mask_resized).float().to(device).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    with torch.no_grad():
        early_features, final_features = vision_encoder_multiscale(img_norm, vision_module, vision_config)
        refined_mask = hqsam_head_refiner(final_features, early_features, mask_t, n_iters=iters)

    refined_mask_np = refined_mask.squeeze(0).squeeze(0).float().cpu().numpy()
    refined_mask_full = cv2.resize(refined_mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    refined_mask_binary = (refined_mask_full > 0.5).astype(np.uint8)

    return refined_mask_binary


def refine_segmentation_with_hqsam_head(
    image,
    svg_path: str,
    bbox: dict,
    hqsam_head_refiner,
    vision_module,
    vision_config,
    iters: int = 6,
) -> Tuple[Optional[str], Optional[dict]]:
    """Refine a coarse SVG segmentation using HQ-SAM head refiner with multi-scale features."""
    if hqsam_head_refiner is None:
        return None, None

    try:
        image = _ensure_numpy_rgb(image)
        img_h, img_w = image.shape[:2]

        cx = (bbox["x_min"] + bbox["x_max"]) / 2
        cy = (bbox["y_min"] + bbox["y_max"]) / 2
        bw = bbox["x_max"] - bbox["x_min"]
        bh = bbox["y_max"] - bbox["y_min"]
        bbox_cxcywh = [cx, cy, bw, bh]

        full_svg = svg_from_path(svg_path, img_w, img_h, bbox_cxcywh)
        coarse_soft = render_svg_to_soft_mask(full_svg, img_w, img_h)
        coarse_mask = (coarse_soft > 0.5).astype(np.uint8)

        if coarse_mask.sum() == 0:
            return None, None

        crop_xyxy = _expand_bbox(bbox, img_w, img_h, margin=0.25)
        x1, y1, x2, y2 = crop_xyxy
        crop_img = image[y1:y2, x1:x2, :]
        crop_mask = coarse_mask[y1:y2, x1:x2]

        if crop_mask.sum() == 0:
            return None, None

        refined_crop = hqsam_head_refine(crop_img, crop_mask, hqsam_head_refiner, vision_module, vision_config, iters=iters)

        refined_mask = _paste_mask(img_h, img_w, refined_crop, crop_xyxy)
        refined_mask = _clean_mask(refined_mask).astype(np.uint8)

        if refined_mask.sum() == 0:
            return None, None

        result = bitmap_to_path(refined_mask)
        if result is None:
            return None, None

        refined_path, refined_bbox = result
        return refined_path, refined_bbox

    except Exception:
        import traceback
        traceback.print_exc()
        return None, None


__all__ = [
    "head_refine",
    "refine_segmentation_with_head",
    "hqsam_head_refine",
    "refine_segmentation_with_hqsam_head",
]
