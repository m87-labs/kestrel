"""Legacy HQ-SAM segmentation refiner.

This refiner was used prior to the current SegmentRefiner-based pipeline.
It refines a coarse raster mask derived from the model's SVG output using
an HQ-SAM model, then vectorizes the refined bitmap back into an SVG path.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .seg_refiner import (
    SegmentRefineBitmapsResult,
    _HAS_SEG_DEPS,
    _clean_mask,
    _encode_mask_png_base64,
    _ensure_numpy_rgb,
    _expand_bbox,
    _paste_mask,
    bitmap_to_path,
    render_svg_to_soft_mask,
    svg_from_path,
)

try:  # Optional dependency: OpenCV (used by HQ-SAM prompting utilities)
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:  # Optional dependency: transformers (HQ-SAM model loader)
    from transformers import AutoModel
except ImportError:  # pragma: no cover
    AutoModel = None


_HAS_HQSAM_DEPS = bool(_HAS_SEG_DEPS and cv2 is not None and AutoModel is not None)

_DEFAULT_MODEL_ID = "moondream/hqsam-vith-meta"
_DEFAULT_ITERS = 6  # Matches historical Kestrel usage.


def _from_pretrained(model_id: str, *, local_files_only: bool) -> torch.nn.Module:
    token = os.environ.get("HF_TOKEN")
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if token:
        # Transformers versions differ on whether `token` is accepted.
        try:
            return AutoModel.from_pretrained(model_id, token=token, **kwargs)  # type: ignore[misc]
        except TypeError:
            return AutoModel.from_pretrained(model_id, use_auth_token=token, **kwargs)  # type: ignore[misc]
    return AutoModel.from_pretrained(model_id, **kwargs)  # type: ignore[misc]


def build_sam_model(device: torch.device, model_id: str = _DEFAULT_MODEL_ID) -> torch.nn.Module:
    """Load HQ-SAM (vit_h) from HuggingFace (AutoModel with remote code)."""

    if AutoModel is None:  # pragma: no cover
        raise ImportError("HQ-SAM refiner requires transformers to be installed")

    try:
        model = _from_pretrained(model_id, local_files_only=True)
    except Exception:
        model = _from_pretrained(model_id, local_files_only=False)

    model = model.to(device)
    model.eval()

    # MetaSamHQModel exposes .sam and .resize.
    if not hasattr(model, "sam"):
        raise AttributeError("Loaded HQ-SAM model is missing required attribute: sam")
    if not hasattr(model, "resize"):
        raise AttributeError("Loaded HQ-SAM model is missing required attribute: resize")
    return model


def _extract_points_and_mask(
    pred_mask: torch.Tensor, gamma: float
) -> Tuple[list, list, list, torch.Tensor]:
    """Extract prompt points/box and Gaussian mask prompt from the current mask."""

    if cv2 is None:  # pragma: no cover
        raise ImportError("HQ-SAM refiner requires opencv-python to be installed")

    pred_mask_np = pred_mask.detach().cpu().numpy().astype(np.uint8)
    device = pred_mask.device

    if pred_mask_np.sum() == 0:
        pred_mask_dt_np = np.zeros_like(pred_mask_np, dtype=np.float32)
    else:
        pred_mask_dt_np = cv2.distanceTransform(pred_mask_np, distanceType=cv2.DIST_L2, maskSize=3)
    pred_mask_dt = torch.from_numpy(pred_mask_dt_np).to(device, torch.float32)

    pred_max_dist = pred_mask_dt.max()
    coords_y, coords_x = torch.where(pred_mask_dt == pred_max_dist)
    if coords_y.numel() == 0:
        pos_x, pos_y = 0, 0
    else:
        pos_x, pos_y = int(coords_x[0]), int(coords_y[0])

    coord = torch.nonzero(pred_mask)
    if coord.numel() == 0:
        ymin, xmin, ymax, xmax = 0, 0, 0, 0
    else:
        y_coord, x_coord = coord[:, 0], coord[:, 1]
        ymin = int(y_coord.min())
        xmin = int(x_coord.min())
        ymax = int(y_coord.max())
        xmax = int(x_coord.max())

    # Find a negative point: background pixel inside bbox with max distance from foreground.
    pred_mask_rev_np = (pred_mask_np == 0).astype(np.uint8)
    box_mask_np = np.zeros_like(pred_mask_np, dtype=np.uint8)
    box_mask_np[ymin : ymax + 1, xmin : xmax + 1] = 1
    bg_in_box = pred_mask_rev_np & box_mask_np

    neg_x, neg_y = None, None
    if bg_in_box.sum() > 0:
        bg_dt = cv2.distanceTransform(pred_mask_rev_np, distanceType=cv2.DIST_L2, maskSize=3)
        bg_dt[box_mask_np == 0] = 0
        if bg_dt.max() > 0:
            bg_dt_t = torch.from_numpy(bg_dt).to(device, torch.float32)
            coords_y_neg, coords_x_neg = torch.where(bg_dt_t == bg_dt_t.max())
            if coords_y_neg.numel() > 0:
                neg_x, neg_y = int(coords_x_neg[0]), int(coords_y_neg[0])

    mask_area = max(pred_mask.sum() / gamma, 1)
    gaus_dt = pred_mask_dt - pred_mask_dt.max()
    gaus_dt = torch.exp(-gaus_dt * gaus_dt / mask_area)
    gaus_dt[pred_mask_dt == 0] = 0

    if neg_x is not None:
        point_coords = [[pos_x, pos_y], [neg_x, neg_y]]
        point_labels = [1, 0]
    else:
        point_coords = [[pos_x, pos_y]]
        point_labels = [1]
    box = [xmin, ymin, xmax, ymax]
    return point_coords, point_labels, box, gaus_dt


def _build_mask_inputs(
    pred_mask: torch.Tensor,
    gaus_dt: torch.Tensor,
    target_size: Tuple[int, int],
    strength: float,
) -> torch.Tensor:
    """Build SAM mask input prompt tensor (1, 1, 256, 256)."""

    pred_masks = pred_mask.float().unsqueeze(0).unsqueeze(0)
    gaus = gaus_dt.float().unsqueeze(0).unsqueeze(0)

    neg_one = pred_masks.new_full((), -1.0)
    pos_one = pred_masks.new_full((), 1.0)
    pred_masks = torch.where(pred_masks == 0, neg_one, pos_one)

    pred_masks = F.interpolate(pred_masks, target_size, mode="bilinear", align_corners=False)
    gaus = F.interpolate(gaus, target_size, mode="bilinear", align_corners=False)

    h, w = pred_masks.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    pred_masks = F.pad(pred_masks, (0, padw, 0, padh), "constant", -1.0)
    gaus = F.pad(gaus, (0, padw, 0, padh), "constant", 0.0)

    pred_masks = F.interpolate(pred_masks, (256, 256), mode="bilinear", align_corners=False)
    gaus = F.interpolate(gaus, (256, 256), mode="bilinear", align_corners=False)

    neg_strength = pred_masks.new_full((), -float(strength))
    pos_strength = pred_masks.new_full((), float(strength))
    pred_masks = torch.where(pred_masks <= 0, neg_strength, pos_strength)

    gaus = torch.where(gaus <= 0, torch.ones_like(gaus), gaus)
    return pred_masks * gaus


def sam_refine(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    sam_model: torch.nn.Module,
    *,
    iters: int,
    strength: float = 30.0,
    gamma: float = 4.0,
) -> np.ndarray:
    """Run HQ-SAM iterative refinement on an RGB crop and coarse binary mask."""

    if cv2 is None:  # pragma: no cover
        raise ImportError("HQ-SAM refiner requires opencv-python to be installed")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if coarse_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

    sam = sam_model.sam  # type: ignore[attr-defined]
    resize = getattr(sam_model, "resize", None)
    if resize is None:
        raise AttributeError("HQ-SAM model is missing required attribute: resize")
    device = next(sam_model.parameters()).device

    img_h, img_w = image.shape[:2]
    if coarse_mask.shape != (img_h, img_w):
        coarse_mask = cv2.resize(coarse_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    current_mask = torch.tensor(coarse_mask, dtype=torch.uint8, device=device)

    resized_img = resize.apply_image(image)
    resized_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float().to(device)
    input_images = torch.stack([sam.preprocess(resized_tensor)], dim=0)

    with torch.no_grad():
        image_embeddings, interm_embeddings = sam.image_encoder(input_images)
    interm_embeddings = interm_embeddings[0]

    for _ in range(max(1, int(iters))):
        point_coords, point_labels, box, gaus_dt = _extract_points_and_mask(current_mask, gamma)
        mask_inputs = _build_mask_inputs(current_mask, gaus_dt, resized_tensor.shape[-2:], strength).to(device)

        points_np = np.array(point_coords, dtype=np.float32)
        points_t = torch.from_numpy(points_np).unsqueeze(0).to(device)
        points_scaled = resize.apply_coords_torch(points_t, (img_h, img_w))
        input_points = points_scaled.unsqueeze(0)

        input_labels = (
            torch.tensor(point_labels, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        )

        box_np = np.array(box, dtype=np.float32)
        box_t = torch.from_numpy(box_np).unsqueeze(0).to(device)
        box_scaled = resize.apply_boxes_torch(box_t, (img_h, img_w))
        input_boxes = box_scaled.unsqueeze(0)

        batched_input = [
            {
                "image": resized_tensor,
                "original_size": (img_h, img_w),
                "point_coords": input_points.squeeze(0),
                "point_labels": input_labels.squeeze(0),
                "boxes": input_boxes.squeeze(0),
                "mask_inputs": mask_inputs,
            }
        ]

        with torch.no_grad():
            outputs = sam.forward_with_image_embeddings(
                image_embeddings=image_embeddings,
                interm_embeddings=interm_embeddings,
                batched_input=batched_input,
                multimask_output=True,
                hq_token_only=False,
            )

        out = outputs[0]
        sam_ious = out["iou_predictions"]
        sam_masks = out["masks"]

        best_idx = sam_ious[0].argmax()
        best_mask_logits = sam_masks[0, best_idx : best_idx + 1]
        current_mask = (best_mask_logits > 0).squeeze(0).to(torch.uint8)

    return current_mask.detach().cpu().numpy()


class HQSamRefiner:
    """Legacy HQ-SAM refiner wrapper matching SegmentRefiner's call surface."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_id: str = _DEFAULT_MODEL_ID,
        iters: int = _DEFAULT_ITERS,
    ) -> None:
        if not _HAS_HQSAM_DEPS:
            raise ImportError(
                "HQ-SAM refinement requires optional dependencies: "
                "pip install transformers opencv-python-headless pillow resvg pypotrace"
            )
        self._device = device
        self._iters = max(1, int(iters))
        self._model = build_sam_model(device, model_id=model_id)

    def refine_with_bitmaps(
        self,
        image: np.ndarray | bytes,
        svg_path: str,
        bbox: dict,
        *,
        return_base64: bool = False,
    ) -> SegmentRefineBitmapsResult:
        coarse_mask_b64: Optional[str] = None
        refined_mask_b64: Optional[str] = None

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

            if return_base64:
                coarse_mask_b64 = _encode_mask_png_base64(coarse_mask)
                # Best-effort default if refinement cannot run.
                refined_mask_b64 = coarse_mask_b64

        except Exception:
            traceback.print_exc()
            return SegmentRefineBitmapsResult(None, None, None, None)

        if coarse_mask.sum() == 0 and not return_base64:
            return SegmentRefineBitmapsResult(None, None, None, None)

        refined_path: Optional[str] = None
        refined_bbox: Optional[dict] = None

        try:
            crop_xyxy = _expand_bbox(bbox, img_w, img_h, margin=0.25)
            x1, y1, x2, y2 = crop_xyxy
            crop_img = image[y1:y2, x1:x2, :]
            crop_mask = coarse_mask[y1:y2, x1:x2]

            if crop_mask.sum() == 0:
                if not return_base64:
                    return SegmentRefineBitmapsResult(None, None, None, None)
                return SegmentRefineBitmapsResult(None, None, coarse_mask_b64, refined_mask_b64)

            refined_crop = sam_refine(crop_img, crop_mask, self._model, iters=self._iters)

            refined_mask = _paste_mask(img_h, img_w, refined_crop, crop_xyxy)
            refined_mask = _clean_mask(refined_mask).astype(np.uint8)

            if return_base64:
                refined_mask_b64 = _encode_mask_png_base64(refined_mask)

            if refined_mask.sum() == 0:
                if not return_base64:
                    return SegmentRefineBitmapsResult(None, None, None, None)
                return SegmentRefineBitmapsResult(None, None, coarse_mask_b64, refined_mask_b64)

            result = bitmap_to_path(refined_mask)
            if result is not None:
                refined_path, refined_bbox = result

            return SegmentRefineBitmapsResult(refined_path, refined_bbox, coarse_mask_b64, refined_mask_b64)

        except Exception:
            traceback.print_exc()
            if not return_base64:
                return SegmentRefineBitmapsResult(None, None, None, None)
            return SegmentRefineBitmapsResult(None, None, coarse_mask_b64, refined_mask_b64)

    def __call__(
        self, image: np.ndarray | bytes, svg_path: str, bbox: dict
    ) -> Tuple[Optional[str], Optional[dict]]:
        result = self.refine_with_bitmaps(image, svg_path, bbox, return_base64=False)
        if result.refined_svg_path is None or result.refined_bbox is None:
            return None, None
        return result.refined_svg_path, result.refined_bbox


__all__ = [
    "HQSamRefiner",
    "_HAS_HQSAM_DEPS",
]

