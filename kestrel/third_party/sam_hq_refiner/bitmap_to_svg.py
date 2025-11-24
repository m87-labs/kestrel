from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import potrace
from PIL import Image

TOKEN_BASE = 960
CLAMP_MIN, CLAMP_MAX = -39, 999


@dataclass
class SvgResult:
    svg: str
    paths: List[str]


def bitmap_to_svg(
    mask: np.ndarray,
    *,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
    downsample: int = 1,
) -> Optional[SvgResult]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return None

    bbox_norm = _mask_bbox_norm(mask)
    if bbox_norm is None:
        return None

    h, w = mask.shape[:2]
    arr = (mask > 0).astype(np.uint8)
    scale = 1
    if downsample > 1:
        scaled_w = max(1, w // downsample)
        scaled_h = max(1, h // downsample)
        arr = np.asarray(
            Image.fromarray(arr * 255).resize(
                (scaled_w, scaled_h), resample=Image.NEAREST
            )
        )
        arr = (arr > 128).astype(np.uint8)
        scale = downsample

    bmp = potrace.Bitmap(arr)
    trace = bmp.trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=1 if alphamax else 0,
        opttolerance=opttolerance,
    )

    svg_paths: List[str] = []
    for curve in trace:
        curve_obj = curve.curve if hasattr(curve, "curve") else curve
        svg_paths.append(_curve_to_path(curve_obj, bbox_norm, w, h, scale))

    if not svg_paths:
        return None

    template = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{TOKEN_BASE}" height="{TOKEN_BASE}" '
        f'viewBox="0 0 {TOKEN_BASE} {TOKEN_BASE}" preserveAspectRatio="none" fill="none">'
        f'<path d="|PATH|" fill="white" fill-rule="evenodd"/></svg>'
    )
    svg_str = template.replace("|PATH|", "".join(svg_paths))
    return SvgResult(svg=svg_str, paths=svg_paths)


def _mask_bbox_norm(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    if bw == 0 or bh == 0:
        return None
    return [cx / w, cy / h, bw / w, bh / h]


def _curve_to_path(
    curve, bbox_norm: List[float], img_w: int, img_h: int, scale: int
) -> str:
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


def _coord(
    pt: np.ndarray, bbox_norm: List[float], img_w: int, img_h: int, scale: int
) -> str:
    cx, cy, bw, bh = bbox_norm
    x_img = (pt[0] * scale) / img_w
    y_img = (pt[1] * scale) / img_h
    x0 = cx - bw * 0.5
    y0 = cy - bh * 0.5
    x_rel = int(((x_img - x0) / max(bw, 1e-12)) * TOKEN_BASE)
    y_rel = int(((y_img - y0) / max(bh, 1e-12)) * TOKEN_BASE)
    x_rel = min(max(x_rel, CLAMP_MIN), CLAMP_MAX)
    y_rel = min(max(y_rel, CLAMP_MIN), CLAMP_MAX)
    return f"{x_rel},{y_rel}"


def clamp_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x, y, w, h = bbox
    x = max(0.0, min(x, width - 1))
    y = max(0.0, min(y, height - 1))
    w = max(0.0, min(w, width - x))
    h = max(0.0, min(h, height - y))
    return [x, y, w, h]
