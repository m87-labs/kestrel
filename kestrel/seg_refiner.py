"""HQ-SAM mask refinement and SVG conversion utilities."""

import io
import re
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

# Lazy imports for optional dependencies
_potrace = None
_resvg_ctx = None
_resvg_tls = threading.local()


# --- SVG Rendering -----------------------------------------------------------

def _get_resvg_ctx():
    ctx = getattr(_resvg_tls, "ctx", None)
    if ctx is None:
        from resvg import usvg
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
    from resvg import render, usvg

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


# --- HQ-SAM Refiner ----------------------------------------------------------

def _extract_points_and_mask(pred_mask: torch.Tensor, gamma: float) -> Tuple[list, list, list, torch.Tensor]:
    """Extract positive point, negative point, bounding box, and Gaussian mask prompt."""
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

    # Find negative point: background pixel inside bbox with max distance from foreground
    pred_mask_rev_np = (pred_mask_np == 0).astype(np.uint8)
    box_mask_np = np.zeros_like(pred_mask_np, dtype=np.uint8)
    box_mask_np[ymin:ymax+1, xmin:xmax+1] = 1
    bg_in_box = pred_mask_rev_np & box_mask_np

    neg_x, neg_y = None, None
    if bg_in_box.sum() > 0:
        bg_dt = cv2.distanceTransform(pred_mask_rev_np, distanceType=cv2.DIST_L2, maskSize=3)
        bg_dt[box_mask_np == 0] = 0  # only consider inside bbox
        if bg_dt.max() > 0:
            bg_dt_t = torch.from_numpy(bg_dt).to(device, torch.float32)
            coords_y_neg, coords_x_neg = torch.where(bg_dt_t == bg_dt_t.max())
            if coords_y_neg.numel() > 0:
                neg_x, neg_y = int(coords_x_neg[0]), int(coords_y_neg[0])

    mask_area = max(pred_mask.sum() / gamma, 1)
    pred_max_dist_tensor = pred_mask_dt.max()
    gaus_dt = pred_mask_dt - pred_max_dist_tensor
    gaus_dt = torch.exp(-gaus_dt * gaus_dt / mask_area)
    gaus_dt[pred_mask_dt == 0] = 0

    # Only include negative point if we found a valid one
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
    target_size: tuple,
    strength: float,
) -> torch.Tensor:
    """Build mask input for SAM following SAMRefiner's extract_mask exactly.

    Args:
        pred_mask: Binary mask (H, W), uint8 tensor
        gaus_dt: Gaussian distance transform (H, W), float tensor
        target_size: (H, W) of resized image
        strength: Mask prompt strength (default 30)

    Returns:
        Mask input tensor (1, 1, 256, 256)
    """
    # Convert to float and add batch/channel dims: (1, 1, H, W)
    pred_masks = pred_mask.float().unsqueeze(0).unsqueeze(0)
    gaus = gaus_dt.float().unsqueeze(0).unsqueeze(0)

    # Convert binary mask to signed: 0 -> -1, 1 -> +1
    pred_masks = torch.where(pred_masks == 0, torch.tensor(-1.0, device=pred_masks.device), torch.tensor(1.0, device=pred_masks.device))

    # Resize both to target size
    pred_masks = F.interpolate(pred_masks, target_size, mode="bilinear", align_corners=False)
    gaus = F.interpolate(gaus, target_size, mode="bilinear", align_corners=False)

    # Pad to 1024x1024
    h, w = pred_masks.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    pred_masks = F.pad(pred_masks, (0, padw, 0, padh), "constant", -1.0)  # pad with -1 (background)
    gaus = F.pad(gaus, (0, padw, 0, padh), "constant", 0.0)  # pad with 0

    # Resize to 256x256 for SAM
    pred_masks = F.interpolate(pred_masks, (256, 256), mode="bilinear", align_corners=False)
    gaus = F.interpolate(gaus, (256, 256), mode="bilinear", align_corners=False)

    # Apply strength: negative regions get -strength, positive get +strength
    pred_masks = torch.where(pred_masks <= 0, -strength * torch.ones_like(pred_masks), strength * torch.ones_like(pred_masks))

    # Multiply by gaus_dt (background gaus is 0, set to 1 so it becomes -strength)
    gaus = torch.where(gaus <= 0, torch.ones_like(gaus), gaus)
    mask_inputs = pred_masks * gaus

    return mask_inputs


def build_sam_model(device: Optional[torch.device] = None, model_id: str = "moondream/hqsam-vith-meta"):
    """Build HQ-SAM (vit_h) from HuggingFace (AutoModel with remote code)."""
    from transformers import AutoModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # MetaSamHQModel exposes .sam and .resize
    if not hasattr(model, "sam"):
        raise AttributeError("Loaded model missing sam attribute.")
    if not hasattr(model, "resize"):
        raise AttributeError("Loaded model missing resize transform.")

    return model


def sam_refine(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    sam_model,
    iters: int = 8,
    strength: float = 30.0,
    gamma: float = 4.0,
) -> np.ndarray:
    """
    HQ-SAM iterative mask refinement.

    Args:
        image: RGB numpy array (H, W, 3), uint8.
        coarse_mask: Binary mask (H, W), uint8.
        sam_model: Namespace with fields: sam (HQ-SAM), resize (ResizeLongestSide), device.
        iters: Number of refinement iterations.
        strength: Mask prompt strength.
        gamma: Gaussian spread for mask prompt.

    Returns:
        Refined binary mask (H, W), uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if coarse_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

    sam = sam_model.sam
    resize = getattr(sam_model, "resize", None)
    if resize is None:
        raise AttributeError("sam_model missing resize transform.")
    device = next(sam_model.parameters()).device

    img_h, img_w = image.shape[0], image.shape[1]

    if coarse_mask.shape != (img_h, img_w):
        coarse_mask = cv2.resize(coarse_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    current_mask = torch.tensor(coarse_mask, dtype=torch.uint8, device=device)

    # Prepare resized image and embeddings once
    resized_img = resize.apply_image(image)
    resized_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float().to(device)  # CHW
    input_images = torch.stack([sam.preprocess(resized_tensor)], dim=0)

    with torch.no_grad():
        image_embeddings, interm_embeddings = sam.image_encoder(input_images)
    # HQ-SAM returns a list of intermediate embeddings; use the first one
    interm_embeddings = interm_embeddings[0]

    for _ in range(iters):
        point_coords, point_labels, box, gaus_dt = _extract_points_and_mask(current_mask, gamma)

        # Build mask prompt at resized resolution
        mask_inputs = _build_mask_inputs(current_mask, gaus_dt, resized_tensor.shape[-2:], strength).to(device)

        # Scale prompts with ResizeLongestSide utilities
        points_np = np.array(point_coords, dtype=np.float32)
        points_t = torch.from_numpy(points_np).unsqueeze(0).to(device)  # 1 x N x 2
        points_scaled = resize.apply_coords_torch(points_t, (img_h, img_w))
        input_points = points_scaled.unsqueeze(0)  # 1 x 1 x N x 2

        input_labels = torch.tensor(point_labels, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)

        box_np = np.array(box, dtype=np.float32)
        box_t = torch.from_numpy(box_np).unsqueeze(0).to(device)  # 1 x 4
        box_scaled = resize.apply_boxes_torch(box_t, (img_h, img_w))
        input_boxes = box_scaled.unsqueeze(0)  # 1 x 1 x 4

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
        sam_masks = out["masks"]  # already upsampled to original size

        best_idx = sam_ious[0].argmax()
        best_mask_logits = sam_masks[0, best_idx:best_idx+1]

        # Keep mask 2D for downstream processing
        current_mask = (best_mask_logits > 0).squeeze(0).to(torch.uint8)

    return current_mask.detach().cpu().numpy()


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


# --- High-level refinement API -----------------------------------------------

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


def refine_segmentation(
    image,
    svg_path: str,
    bbox: dict,
    sam_model,
    iters: int = 8,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Refine a coarse SVG segmentation using HQ-SAM.

    Args:
        image: RGB image (numpy array or pyvips Image).
        svg_path: SVG path string from model output.
        bbox: Bbox dict with x_min, y_min, x_max, y_max (normalized 0-1).
        sam_model: HQ-SAM wrapper from build_sam_model.
        iters: Number of refinement iterations.

    Returns:
        (refined_svg_path, refined_bbox) or (None, None) on failure.
    """
    if sam_model is None:
        print("[refiner] sam_model is None, skipping refinement")
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
            print("[refiner] coarse_mask is empty, skipping refinement")
            return None, None

        # Crop around bbox with margin
        crop_xyxy = _expand_bbox(bbox, img_w, img_h, margin=0.25)
        x1, y1, x2, y2 = crop_xyxy
        crop_img = image[y1:y2, x1:x2, :]
        crop_mask = coarse_mask[y1:y2, x1:x2]

        if crop_mask.sum() == 0:
            print("[refiner] crop_mask is empty, skipping refinement")
            return None, None

        # Run SAM refinement
        refined_crop = sam_refine(crop_img, crop_mask, sam_model, iters=iters)

        # Paste back to full size and clean up
        refined_mask = _paste_mask(img_h, img_w, refined_crop, crop_xyxy)
        refined_mask = _clean_mask(refined_mask).astype(np.uint8)

        if refined_mask.sum() == 0:
            print("[refiner] refined_mask is empty, skipping refinement")
            return None, None

        result = bitmap_to_path(refined_mask)
        if result is None:
            print("[refiner] bitmap_to_path failed, skipping refinement")
            return None, None

        refined_path, refined_bbox = result
        return refined_path, refined_bbox

    except Exception:
        import traceback
        traceback.print_exc()
        return None, None


__all__ = [
    "build_sam_model",
    "refine_segmentation",
    "render_svg_to_soft_mask",
    "svg_from_path",
    "bitmap_to_path",
]
