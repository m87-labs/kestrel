"""Mask refinement and SVG conversion utilities."""

import io
import os
import re
import threading
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from resvg import render, usvg

from .moondream.vision import vision_encoder
from .moondream.config import VisionConfig

# Number of refinement iterations
_REFINER_ITERS = 5

# Lazy imports for optional dependencies
_potrace = None
_resvg_ctx = None
_resvg_tls = threading.local()


# --- Refinement head model ---------------------------------------------------

class _LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _MaskEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, embed_dim // 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, embed_dim // 16),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 16, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, embed_dim // 8), embed_dim // 8),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(16, embed_dim // 4), embed_dim // 4),
            nn.GELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
        )

    def forward(self, mask):
        x = self.conv1(mask)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.interpolate(x, size=(27, 27), mode="bilinear", align_corners=False)
        return x


class _FeatureFusion(nn.Module):
    def __init__(self, enc_dim=1152, embed_dim=256):
        super().__init__()
        self.early_proj = nn.Sequential(
            nn.Linear(enc_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )

    def forward(self, early_feat, final_feat):
        B = early_feat.shape[0]
        early = self.early_proj(early_feat)
        early = early.transpose(1, 2).view(B, -1, 27, 27)
        fused = torch.cat([early, final_feat], dim=1)
        return self.fusion(fused)


class _TwoWayBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=8):
        super().__init__()
        self.output_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, output_tokens, img_tokens, output_pe, img_pe):
        q = k = output_tokens + output_pe
        attn_out = self.output_self_attn(q, k, output_tokens)[0]
        output_tokens = self.norm1(output_tokens + attn_out)

        q = output_tokens + output_pe
        k = img_tokens + img_pe
        attn_out = self.cross_attn_token_to_image(q, k, img_tokens)[0]
        output_tokens = self.norm2(output_tokens + attn_out)

        output_tokens = self.norm3(output_tokens + self.mlp(output_tokens))

        q = img_tokens + img_pe
        k = output_tokens + output_pe
        attn_out = self.cross_attn_image_to_token(q, k, output_tokens)[0]
        img_tokens = self.norm4(img_tokens + attn_out)

        return output_tokens, img_tokens


class _RefinementHead(nn.Module):
    def __init__(self, enc_dim=1152, embed_dim=256, num_heads=8, num_layers=2, num_masks=4):
        super().__init__()
        self.enc_dim = enc_dim
        self.embed_dim = embed_dim
        self.num_masks = num_masks

        self.image_proj = nn.Conv2d(enc_dim, embed_dim, 1)
        self.image_pe = nn.Parameter(torch.randn(1, embed_dim, 27, 27) * 0.02)

        self.hq_fusion = _FeatureFusion(enc_dim, embed_dim)
        self.mask_encoder = _MaskEncoder(embed_dim)

        self.mask_tokens = nn.Parameter(torch.randn(num_masks, embed_dim) * 0.02)
        self.iou_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.output_pe = nn.Parameter(torch.randn(num_masks + 1, embed_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList(
            [_TwoWayBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.final_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            _LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            _LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            _LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
            _LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            _LayerNorm2d(embed_dim // 8),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 8, kernel_size=3, padding=1),
            _LayerNorm2d(embed_dim // 8),
            nn.GELU(),
        )

        self.final_q_proj = nn.Linear(embed_dim, embed_dim // 8)
        self.final_attn_up = nn.MultiheadAttention(embed_dim // 8, num_heads=4, batch_first=True)
        self.final_norm_up = nn.LayerNorm(embed_dim // 8)

        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim // 8, embed_dim // 8),
                nn.GELU(),
                nn.Linear(embed_dim // 8, embed_dim // 8),
                nn.GELU(),
                nn.Linear(embed_dim // 8, embed_dim // 8),
            )
            for _ in range(num_masks)
        ])

        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_masks),
        )

    def forward(self, final_features, early_features, coarse_mask):
        B = final_features.shape[0]
        H, W = coarse_mask.shape[-2:]

        img = final_features.view(B, 27, 27, self.enc_dim).permute(0, 3, 1, 2)
        img = self.image_proj(img)
        img = self.hq_fusion(early_features, img)

        mask_embed = self.mask_encoder(coarse_mask)
        img = img + mask_embed

        img_tokens = img.flatten(2).transpose(1, 2)
        img_pe = self.image_pe.flatten(2).transpose(1, 2).expand(B, -1, -1)

        output_tokens = torch.cat([
            self.mask_tokens.unsqueeze(0).expand(B, -1, -1),
            self.iou_token.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)
        output_pe = self.output_pe.unsqueeze(0).expand(B, -1, -1)

        for block in self.transformer_blocks:
            output_tokens, img_tokens = block(output_tokens, img_tokens, output_pe, img_pe)

        q = output_tokens + output_pe
        k = img_tokens + img_pe
        attn_out = self.final_token_to_image(q, k, img_tokens)[0]
        output_tokens = self.final_norm(output_tokens + attn_out)

        img_up = img_tokens.transpose(1, 2).view(B, self.embed_dim, 27, 27)
        img_up = self.upsample(img_up)

        img_up_tokens = img_up.flatten(2).transpose(1, 2)
        mask_tokens = output_tokens[:, :self.num_masks]
        mask_tokens_proj = self.final_q_proj(mask_tokens)
        mask_tokens_refined = (
            mask_tokens_proj
            + self.final_attn_up(self.final_norm_up(mask_tokens_proj), img_up_tokens, img_up_tokens)[0]
        )

        masks = []
        for i in range(self.num_masks):
            mask_weights = self.mask_mlps[i](mask_tokens_refined[:, i])
            mask = torch.einsum("bc,bchw->bhw", mask_weights, img_up)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

        iou_token = output_tokens[:, self.num_masks]
        iou_pred = self.iou_head(iou_token)

        return masks, iou_pred


# --- SegmentRefiner ----------------------------------------------------------

class SegmentRefiner:
    """Refines coarse segmentation masks."""

    def __init__(self, vision_module: nn.Module, vision_config: VisionConfig, device: torch.device):
        self._device = device
        self._vision_module = vision_module
        self._vision_config = vision_config

        # Cached normalization constants
        self._img_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self._img_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

        self._head = _RefinementHead(enc_dim=1152, embed_dim=256, num_masks=4)
        weights_path = hf_hub_download(
            repo_id="moondream/SegHeadRefiner",
            filename="model.pt",
            token=os.environ.get("HF_TOKEN"),
        )
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        self._head.load_state_dict(ckpt["head"])
        self._head = self._head.to(device).to(torch.bfloat16)
        self._head.eval()

    def _refine_mask(self, image: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """Refine a coarse binary mask using vision features and refinement head."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if coarse_mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {coarse_mask.shape}")

        device = self._device
        img_h, img_w = image.shape[:2]

        if coarse_mask.shape != (img_h, img_w):
            coarse_mask = cv2.resize(coarse_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        img_resized = cv2.resize(image, (378, 378), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(coarse_mask, (378, 378), interpolation=cv2.INTER_NEAREST)

        img_norm = torch.from_numpy(img_resized).float().to(device)
        img_norm = img_norm.permute(2, 0, 1).unsqueeze(0) / 255.0
        img_norm = (img_norm - self._img_mean) / self._img_std
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
            # Iterative refinement
            current_mask = mask_t
            for _ in range(_REFINER_ITERS):
                all_logits, iou_pred = self._head(final_features, early_features, current_mask)
                best_idx = iou_pred.argmax(dim=1)
                batch_idx = torch.arange(all_logits.shape[0], device=all_logits.device)
                best_logits = all_logits[batch_idx, best_idx]
                current_mask = torch.sigmoid(best_logits).unsqueeze(1)

        refined_mask_np = current_mask.squeeze(0).squeeze(0).float().cpu().numpy()
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
