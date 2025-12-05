"""HQ-SAM head-based mask refinement and SVG conversion utilities."""

import io
import re
import threading
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from resvg import render, usvg

from .moondream.vision import vision_encoder
from .moondream.config import VisionConfig
from .hqsam_head_refiner import HQSAMHeadRefiner

# Number of refinement iterations for HQ-SAM head
REFINER_ITERS = 5

# Lazy imports for optional dependencies
_potrace = None
_resvg_ctx = None
_resvg_tls = threading.local()


class SegmentRefiner:
    """Refines coarse segmentation masks using HQ-SAM head."""

    def __init__(self, hqsam_head: HQSAMHeadRefiner, vision_module: nn.Module, vision_config: VisionConfig):
        self._hqsam_head = hqsam_head
        self._vision_module = vision_module
        self._vision_config = vision_config

    @property
    def device(self) -> torch.device:
        return next(self._hqsam_head.parameters()).device

    def _refine_mask(self, image: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """Refine a coarse binary mask using vision features and HQ-SAM head."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if coarse_mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

        device = self.device
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

        mask_t = (
            torch.from_numpy(mask_resized)
            .float()
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            final_features, early_features = vision_encoder(
                img_norm, self._vision_module, self._vision_config, early_layer=8
            )
            refined_mask = self._hqsam_head(
                final_features, early_features, mask_t, n_iters=REFINER_ITERS
            )

        refined_mask_np = refined_mask.squeeze(0).squeeze(0).float().cpu().numpy()
        refined_mask_full = cv2.resize(refined_mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        return (refined_mask_full > 0.5).astype(np.uint8)

    def __call__(self, image, svg_path: str, bbox: dict) -> Tuple[Optional[str], Optional[dict]]:
        """Refine a coarse SVG segmentation.

        Args:
            image: RGB image (numpy array or pyvips Image).
            svg_path: SVG path string from model output.
            bbox: Bbox dict with x_min, y_min, x_max, y_max (normalized 0-1).

        Returns:
            (refined_svg_path, refined_bbox) or (None, None) on failure.
        """
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

            refined_crop = self._refine_mask(crop_img, crop_mask)

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
            traceback.print_exc()
            return None, None


# --- SVG Rendering -----------------------------------------------------------

def _get_resvg_ctx():
    ctx = getattr(_resvg_tls, "ctx", None)
    if ctx is None:
        fontdb = usvg.FontDatabase.default()
        fontdb.load_system_fonts()
        opts = usvg.Options.default()
        ctx = (opts, fontdb)
        _resvg_tls.ctx = ctx
    return ctx


def svg_from_path(svg_path: str, width: float, height: float, bbox: List[float]) -> str:
    """Build full SVG from path string (0-1 coords) and bbox [cx, cy, w, h] in normalized coords."""
    x0 = bbox[0] - bbox[2] / 2
    y0 = bbox[1] - bbox[3] / 2
    sx = bbox[2]
    sy = bbox[3]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" '
        f'preserveAspectRatio="none" width="{width}" height="{height}">'
        f'<path d="{svg_path}" fill="white" transform="translate({x0},{y0}) scale({sx},{sy})"/></svg>'
    )


def render_svg_to_soft_mask(svg: str, width: int, height: int, scale: int = 2) -> np.ndarray:
    """Render SVG to a soft mask using resvg backend. Returns float32 [0,1] array."""
    width = int(round(width))
    height = int(round(height))
    scale = max(1, int(scale))

    opts, fontdb = _get_resvg_ctx()

    normalized_svg = (
        svg.replace(",M", " M").replace(",m", " m")
        .replace(",L", " L").replace(",l", " l")
        .replace(",C", " C").replace(",c", " c")
        .replace(",Z", " Z").replace(",z", " z")
    )

    render_width = max(1, int(round(width * scale)))
    render_height = max(1, int(round(height * scale)))
    normalized_svg = re.sub(r'width="[0-9.]+"', f'width="{render_width}"', normalized_svg)
    normalized_svg = re.sub(r'height="[0-9.]+"', f'height="{render_height}"', normalized_svg)

    tree = usvg.Tree.from_str(normalized_svg, opts, fontdb)
    png_bytes = bytes(render(tree, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))

    pil_image = Image.open(io.BytesIO(png_bytes))
    if pil_image.mode in ('RGBA', 'LA'):
        alpha_channel = pil_image.getchannel('A')
    else:
        alpha_channel = pil_image.convert('L')

    mask = np.array(alpha_channel, dtype=np.float32)
    if scale > 1:
        mask = mask.reshape(int(height), scale, int(width), scale).mean(axis=(1, 3))

    return mask / 255.0


# --- Bitmap to SVG (potrace) -------------------------------------------------


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[List[float], dict]]:
    """Return (cxcywh, minmax) bbox from mask, or None if empty.

    cxcywh: [cx, cy, w, h] for SVG path coordinate mapping
    minmax: {x_min, y_min, x_max, y_max} for output
    Both normalized to [0,1].
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    # Pixel-inclusive width/height
    bw = max(1.0, x1 - x0 + 1)
    bh = max(1.0, y1 - y0 + 1)
    cx = (x0 + x1 + 1) / 2.0
    cy = (y0 + y1 + 1) / 2.0
    cxcywh = [cx / w, cy / h, bw / w, bh / h]
    minmax = {
        "x_min": x0 / w,
        "y_min": y0 / h,
        "x_max": (x1 + 1) / w,
        "y_max": (y1 + 1) / h,
    }
    return cxcywh, minmax


def _coord(pt: np.ndarray, bbox_norm: List[float], img_w: int, img_h: int, scale: int) -> str:
    """Map traced point to bbox-normalized 0-1 path coords."""
    cx, cy, bw, bh = bbox_norm
    x_img = (pt[0] * scale) / img_w
    y_img = (pt[1] * scale) / img_h
    x0 = cx - bw * 0.5
    y0 = cy - bh * 0.5
    x_rel = (x_img - x0) / max(bw, 1e-12)
    y_rel = (y_img - y0) / max(bh, 1e-12)
    return f"{x_rel:.3f},{y_rel:.3f}".replace("0.", ".").replace("-.", "-0.")


def _curve_to_path(curve, bbox_norm: List[float], img_w: int, img_h: int, scale: int) -> str:
    """Convert a potrace curve to SVG path segment."""
    parts = [f"M{_coord(curve.start_point, bbox_norm, img_w, img_h, scale)}"]
    for seg in curve.segments:
        if seg.is_corner:
            parts.append(
                f"L{_coord(seg.c, bbox_norm, img_w, img_h, scale)}"
                f"L{_coord(seg.end_point, bbox_norm, img_w, img_h, scale)}"
            )
        else:
            parts.append(
                "C"
                f"{_coord(seg.c1, bbox_norm, img_w, img_h, scale)},"
                f"{_coord(seg.c2, bbox_norm, img_w, img_h, scale)},"
                f"{_coord(seg.end_point, bbox_norm, img_w, img_h, scale)}"
            )
    parts.append("z")
    return "".join(parts)


def bitmap_to_path(
    mask: np.ndarray,
    *,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    downsample: int = 1,
) -> Optional[Tuple[str, dict]]:
    """Trace a binary mask into SVG path string and bbox.

    Returns (svg_path, bbox_minmax) or None if mask is empty.
    svg_path uses 0-1 coords relative to the bbox.
    bbox_minmax is {x_min, y_min, x_max, y_max} normalized to image dims.
    """
    global _potrace
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return None

    bbox_result = _mask_bbox(mask)
    if bbox_result is None:
        return None
    bbox_cxcywh, bbox_minmax = bbox_result

    if _potrace is None:
        import potrace as _potrace

    h, w = mask.shape[:2]
    arr = (mask > 0).astype(np.uint8)
    scale = 1
    if downsample > 1:
        scaled_w = max(1, w // downsample)
        scaled_h = max(1, h // downsample)
        arr = np.asarray(
            Image.fromarray(arr * 255).resize((scaled_w, scaled_h), resample=Image.NEAREST)
        )
        arr = (arr > 128).astype(np.uint8)
        scale = downsample

    bmp = _potrace.Bitmap(arr)
    trace = bmp.trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=1 if alphamax else 0,
        opttolerance=opttolerance,
    )

    svg_paths: List[str] = []
    for curve in trace:
        curve_obj = curve.curve if hasattr(curve, "curve") else curve
        svg_paths.append(_curve_to_path(curve_obj, bbox_cxcywh, w, h, scale))

    if not svg_paths:
        return None
    return "".join(svg_paths), bbox_minmax


# --- Mask post-processing ----------------------------------------------------

def _clean_mask(mask: np.ndarray, area_frac: float = 0.0015) -> np.ndarray:
    """Remove small holes/islands and apply morphological close."""
    h, w = mask.shape
    area_thresh = max(1.0, area_frac * h * w)
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    for fill_holes in [True, False]:
        working = ((mask == 0) if fill_holes else mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working, 8)
        sizes = stats[1:, -1]
        small = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if small:
            fill = [0] + small if fill_holes else [i for i in range(n_labels) if i not in [0] + small]
            if not fill_holes and not fill:
                fill = [int(np.argmax(sizes)) + 1]
            mask = np.isin(regions, fill).astype(np.uint8)

    return cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel) > 0


def _expand_bbox(bbox_minmax: dict, img_w: int, img_h: int, margin: float = 0.25) -> Tuple[int, int, int, int]:
    """Expand bbox by margin and clip to image bounds. Returns (x1, y1, x2, y2) in pixels."""
    x_min = bbox_minmax["x_min"] * img_w
    y_min = bbox_minmax["y_min"] * img_h
    x_max = bbox_minmax["x_max"] * img_w
    y_max = bbox_minmax["y_max"] * img_h

    bw = x_max - x_min
    bh = y_max - y_min
    expand_x = bw * margin
    expand_y = bh * margin

    x1 = max(0, int(x_min - expand_x))
    y1 = max(0, int(y_min - expand_y))
    x2 = min(img_w, int(x_max + expand_x))
    y2 = min(img_h, int(y_max + expand_y))

    return x1, y1, x2, y2


def _paste_mask(full_h: int, full_w: int, crop_mask: np.ndarray, crop_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """Paste cropped mask back into full-size mask."""
    x1, y1, x2, y2 = crop_xyxy
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)

    crop_h, crop_w = crop_mask.shape[:2]
    target_h, target_w = y2 - y1, x2 - x1

    if crop_h != target_h or crop_w != target_w:
        crop_mask = cv2.resize(crop_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    full_mask[y1:y2, x1:x2] = crop_mask
    return full_mask


def _ensure_numpy_rgb(image) -> np.ndarray:
    """Convert image to RGB numpy array (H, W, 3) uint8. Accepts numpy or pyvips."""
    if isinstance(image, np.ndarray):
        return image
    # Assume pyvips
    if image.bands == 4:
        image = image.extract_band(0, n=3)
    elif image.bands == 1:
        image = image.bandjoin([image, image])
    mem = image.write_to_memory()
    return np.frombuffer(mem, dtype=np.uint8).reshape(image.height, image.width, image.bands)


__all__ = [
    "SegmentRefiner",
    "svg_from_path",
    "render_svg_to_soft_mask",
    "bitmap_to_path",
]
