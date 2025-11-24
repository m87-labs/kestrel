# HQ-SAM Refiner Overview (Kestrel)

This document explains how segmentation refinement is implemented in Kestrel using HQ-SAM. It covers inputs/outputs, the data flow, cropping logic, and the REPL script.

## High-level flow
1. The model generates a coarse SVG path (unit coordinates) plus a normalized bbox.
2. We rasterize the coarse path into a mask, crop around the bbox, and run HQ-SAM refinement on that crop.
3. The refined mask is converted back to an SVG path and returned alongside the coarse outputs. Refinement is mandatory when enabled; failures raise.

## Inputs to the refiner
- `path_tokens`: decoded SVG token strings (unscaled, viewbox-based).
- `bbox`: normalized center/size (`x_center`, `y_center`, `width`, `height` with `x_min/x_max/y_min/y_max`).
- `image`: original pyvips image.
- Config: `sam_hq_checkpoint`, `sam_hq_model_type` (vit_h), `sam_hq_device`, `sam_hq_iters` (default 8), enabled flag.

## Data flow inside `_refine_segment_output` (engine)
1. **Coarse SVG → mask**  
   - `tokens_to_raw_path(path_tokens)` to get viewbox path (960 default).  
   - `svg_from_path` applies bbox transform into image space and produces a full-image SVG.  
   - `render_svg_to_mask` rasterizes to a coarse boolean mask.
2. **Crop with margin**  
   - Convert bbox to pixel XYXY.  
   - Grow width/height by 25% and clamp to image bounds.  
   - Crop image + coarse mask to that region.
3. **HQ-SAM refinement** (`kestrel/third_party/sam_hq_refiner/sam_refiner.py`)  
   - Build prompts from the coarse crop mask:  
     - Box prompt (mask bbox or provided box).  
     - Point prompts from geodesic distance (farthest interior positive, optional negative).  
     - Mask prompt: coarse mask scaled to ±strength and multiplied by a gaussian of the distance transform.  
   - Iterate `sam_hq_iters` (default 8), picking highest-IoU mask each iteration.  
   - Returns refined crop mask and timing.
4. **Reproject to full image**  
   - Paste refined crop mask back into full-image canvas.  
   - Compute refined bbox from nonzero pixels.  
   - Convert refined mask to SVG via `bitmap_to_svg` (turdsize=2, alphamax=1.0, opttolerance=0.2), tokenize and scale to unit coords → `refined_svg_path`.
5. **Return fields**  
   - `refined_svg_path`, `refined_bbox`, `sam_refine_timing`. Missing/empty refined outputs raise.

## Segment output dict (when refinement succeeds)
```python
{
    "object": "...",
    "text": "...",                  # coarse path text
    "svg_path": "M0.12 0.34 …",     # coarse path (unit)
    "path_tokens": ["M", "123", ...],
    "token_ids": [...],
    "points": [...],
    "bbox": {                       # coarse bbox (normalized with min/max)
        "x_center": ..., "y_center": ...,
        "width": ..., "height": ...,
        "x_min": ..., "x_max": ...,
        "y_min": ..., "y_max": ...,
    },
    "sizes": [{"width": ..., "height": ...}],
    # Refiner:
    "refined_svg_path": "M0.11 0.33 …",   # refined path (unit coords)
    "refined_bbox": {                     # normalized with min/max
        "x_center": ..., "y_center": ...,
        "width": ..., "height": ...,
        "x_min": ..., "x_max": ...,
        "y_min": ..., "y_max": ...,
    },
    "sam_refine_timing": {
        "iter_s": [...],                  # per-iteration timings
        "image_encoding_s": ...,
        "total_refine_s": ...,
    },
    # Optional: "parse_error" if coarse path failed to parse.
}
```

## Coordinate handling (bbox ↔ image)
- Paths are unit/viewbox (default 960). `svg_from_path` takes the normalized bbox [cx, cy, w, h] and scales/translates the path into image space, preserving aspect via `preserveAspectRatio="none"` with explicit width/height.
- Bbox is normalized to [0,1] relative to image width/height. Pixel XYXY is computed as:
  - `x1 = (cx - w/2) * img_w`, `x2 = (cx + w/2) * img_w`, similarly for y, then grown by 25% and clamped.

## REPL cropping logic (`scripts/seg_repl_hqsam.py`)
- Uses the final segment output and requires refined fields. It builds the SVG using `refined_svg_path`/`refined_bbox` (falls back to tokens only if path missing, but raises if no refined outputs).
- Saves refined overlay (`*_overlay_refined.png`) and refined SVG (`*_mask_refined.svg`). If refined outputs are missing or mask is empty, it raises.
- Bbox→image scaling matches engine: normalized bbox to pixel XYXY, then `svg_from_path` for rasterization.

## Running the REPL
```
uv run python -m scripts.seg_repl_hqsam \
  --weights /path/to/model.pt \
  --sam-hq-checkpoint /path/to/sam_hq_vit_h.pth \
  --sam-hq-iters 8 \
  --max-batch-size 2
```
`--tokenizer` is optional; defaults follow main runtime. Outputs are saved alongside the input image.

## Notes
- Refinement is mandatory when enabled: missing/empty refined outputs fail the request.
- HQ-SAM is loaded lazily when first used; refinement is mandatory when enabled.
- Bbox growth for refinement crops is 25% of width/height (clamped to image).***
