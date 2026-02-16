# Old Refiner Deep Dive (HQ-SAM) + Commit `7e2abc7…` Context

## TL;DR

There are two distinct “old” states in the history around your referenced commit:

1. **Commit `7e2abc712855fe5f80111dca7c78792f7132fca4` (Mon Nov 17 2025)**: segmentation returned the **raw SVG path decoded from model tokens** and **did not run any bitmap refiner**.
2. **Immediately before `9696d5c…` (Dec 5 2025)**: Kestrel used a **HQ-SAM (High Quality Segment Anything) iterative refiner** implemented in `kestrel/seg_refiner.py`. This is the “different refiner” that existed before the current `SegmentRefiner` landed in `9696d5c…`.

This doc focuses on the HQ-SAM refiner (the one you likely want to support alongside the current refiner), and also notes what existed at `7e2abc7…`.

## Chronology (What Changed When)

- `7e2abc7…` (Nov 17 2025): `segment` skill decodes SVG tokens and returns `svg_path`. No refinement stage.
  - `kestrel/skills/segment.py` (commit `7e2abc7`) lines `125-167`.
- `6ce05e3…` (Dec 5 2025 03:10): HQ-SAM refiner is wired into runtime + segment skill.
  - `kestrel/moondream/runtime.py` (commit `6ce05e3`) lines `50` and `416`.
  - `kestrel/skills/segment.py` (commit `6ce05e3`) lines `171-182`.
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) implements HQ-SAM refinement.
- `9696d5c…` (Dec 5 2025 04:11): “Update segmentation refiner model (#11)” replaces HQ-SAM path with the newer `SegmentRefiner` module.
  - Commit message explicitly: “Replace the previous HQ-SAM segmentation path with the new SegmentRefiner module.”

## Old Refiner = HQ-SAM Iterative Mask Refinement

### What It Was

The “old refiner” was an **HQ-SAM model** (loaded via HuggingFace `transformers.AutoModel` with `trust_remote_code=True`) used to iteratively refine a coarse raster mask derived from the model’s SVG output.

- Build/load: `build_sam_model(...)` in `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `319-347`.
- Default HF model id: `"moondream/hqsam-vith-meta"` (`build_sam_model` signature, line `319`).

### Where It Lived

Everything was in `kestrel/seg_refiner.py` (commit `6ce05e3`):

- SVG -> raster: `svg_from_path(...)` (lines `34-45`) + `render_svg_to_soft_mask(...)` (lines `47-83`) using `resvg`.
- Raster -> SVG: `bitmap_to_path(...)` (lines `148-204`) using `potrace`.
- HQ-SAM model load: `build_sam_model(...)` (lines `319-347`).
- HQ-SAM refinement loop: `sam_refine(...)` (lines `350-451`).
- High-level refinement API used by Kestrel: `refine_segmentation(...)` (lines `527-603`).

### How Kestrel Called It (Runtime Wiring + Skill Call Site)

Runtime loads the HQ-SAM model once at startup and stores it:

- `kestrel/moondream/runtime.py` (commit `6ce05e3`) line `50`: `from ..seg_refiner import build_sam_model`
- `kestrel/moondream/runtime.py` (commit `6ce05e3`) line `416`: `self.sam_model = build_sam_model(device=self.device)`

Segmentation skill calls the HQ-SAM refiner during finalize:

- `kestrel/skills/segment.py` (commit `6ce05e3`) line `22`: `from ..seg_refiner import refine_segmentation`
- `kestrel/skills/segment.py` (commit `6ce05e3`) lines `171-182`:
  - condition: `if svg_path and bbox and not parse_error and self._request.image is not None:`
  - call: `refine_segmentation(self._request.image, svg_path, bbox, runtime.sam_model, iters=6)`
  - on success: replaces `svg_path` and `bbox` with refined values

So, in Kestrel usage, HQ-SAM ran for **6 refinement iterations** (`iters=6`), even though `refine_segmentation(..., iters=8)` and `sam_refine(..., iters=8)` default to 8.

### Inputs / Outputs (API Contract)

#### `refine_segmentation(...)`

Signature (commit `6ce05e3`):

- `refine_segmentation(image, svg_path: str, bbox: dict, sam_model, iters: int = 8) -> (Optional[str], Optional[dict])`
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `527-546`.

Inputs:

- `image`: numpy RGB or `pyvips.Image` (converted internally).
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `514-525` (`_ensure_numpy_rgb`).
- `svg_path`: SVG path string in bbox-relative 0-1 coordinates (from the model).
- `bbox`: **min/max normalized bbox**: `{x_min, y_min, x_max, y_max}`.
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) line `540`.
- `sam_model`: output of `build_sam_model(...)`.
- `iters`: number of HQ-SAM iterations (Kestrel used 6).

Outputs:

- `refined_svg_path`: traced from the refined bitmap; still bbox-relative 0-1 coords.
- `refined_bbox`: min/max normalized bbox **recomputed from the refined bitmap**.
  - Comes from `bitmap_to_path(...)` which calls `_mask_bbox(...)`.
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `88-114` and `148-204`.

Failure behavior:

- Returns `(None, None)` on failure, printing a reason or stack trace:
  - `sam_model is None` guard: lines `547-549`
  - empty coarse/crop/refined masks: lines `567-590`
  - `bitmap_to_path` failure: lines `592-595`
  - broad exception handler: lines `600-603`

### Algorithm (Step By Step)

High-level flow inside `refine_segmentation(...)` (commit `6ce05e3`) lines `551-599`:

1. Convert input image to `np.ndarray` RGB uint8 (`_ensure_numpy_rgb`).
2. Convert `bbox` min/max to `bbox_cxcywh` for mapping the SVG path into global coordinates.
   - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `555-560`.
3. Rasterize the model SVG path into a coarse bitmap mask:
   - Build full SVG with `<path ... transform="translate(...) scale(...)"/>`: `svg_from_path` lines `34-45`
   - Render with `resvg` into an alpha mask (`render_svg_to_soft_mask` lines `47-83`)
   - Threshold at `> 0.5` into a binary `coarse_mask`: line `565`
4. Crop around the bbox (+25% margin) to reduce HQ-SAM work:
   - `_expand_bbox(..., margin=0.25)`: `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `479-497`
   - Used at line `572`.
5. Run HQ-SAM iterative refinement on the crop:
   - `refined_crop = sam_refine(crop_img, crop_mask, sam_model, iters=iters)`: line `582`
6. Paste crop back into full-res mask, then clean up:
   - `_paste_mask(...)`: lines `499-512`
   - `_clean_mask(...)` removes small holes/islands via connected-components and applies morphological close: lines `456-475`
7. Vectorize refined bitmap back into SVG:
   - `bitmap_to_path(refined_mask)` uses `potrace` and recomputes bbox from nonzero pixels: lines `148-204`
8. Return `(refined_path, refined_bbox)` if successful.

### HQ-SAM Iteration Details (`sam_refine`)

Core loop lives in `sam_refine(...)` (commit `6ce05e3`) lines `350-451`:

- Precomputes HQ-SAM image embeddings once:
  - `image_embeddings, interm_embeddings = sam.image_encoder(...)`: lines `392-400`
- Each iteration:
  1. Extract prompts from the current mask (`_extract_points_and_mask`):
     - positive point: max distance-transform point inside mask
     - optional negative point: background pixel inside bbox farthest from foreground
     - bbox: tight bbox around mask
     - gaussian distance-transform prompt
     - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `208-267`
  2. Build SAM’s `mask_inputs` tensor (resize -> pad 1024 -> resize 256 -> strength/gamma):
     - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `270-316`
  3. Scale prompts via `sam_model.resize` (ResizeLongestSide):
     - points: lines `408-413`
     - boxes: lines `416-419`
  4. Run `sam.forward_with_image_embeddings(...)` with `multimask_output=True`, choose the best output by `iou_predictions`, and update `current_mask`.
     - forward call: lines `432-439`
     - selection: lines `441-449`

Key parameters (defaults in `sam_refine`):

- `strength=30.0` (mask prompt strength): line `355`
- `gamma=4.0` (gaussian spread): line `356`

### Dependency Surface / Operational Notes

HQ-SAM path depended on:

- `transformers.AutoModel` with `trust_remote_code=True` to load `"moondream/hqsam-vith-meta"`:
  - `kestrel/seg_refiner.py` (commit `6ce05e3`) lines `319-337`
- `torchvision` (required by the HQ-SAM model’s remote code; `transformers` will raise an `ImportError` if missing)
- `cv2` for distance transform + connected components + morphology:
  - `_extract_points_and_mask`: lines `208-256`
  - `_clean_mask`: lines `456-475`
- `resvg` for rasterization: `render_svg_to_soft_mask`: lines `47-83`
- `potrace` for vectorization: `bitmap_to_path`: lines `148-204`
- `PIL.Image` used for downsample/resize inside `bitmap_to_path`: lines `182-186`

This is likely why it was replaced: it’s heavy (HQ-SAM vit_h), uses remote-code model loading, and adds multiple nontrivial dependencies.

## What Actually Existed At Commit `7e2abc7…` (The Commit You Referenced)

At `7e2abc7…` specifically, there is **no HQ-SAM** and **no bitmap refiner** wired into `segment`:

- The segmentation output is produced entirely in `SegmentSkillState.finalize()` by decoding SVG tokens:
  - `kestrel/skills/segment.py` (commit `7e2abc7`) lines `125-167`
- Token decoding/parsing utilities live in `kestrel/utils/svg.py` (commit `7e2abc7`) lines `33-108`.

So if you want “support both” behavior that includes the exact `7e2abc7…` semantics, that corresponds to **“no-op refiner”** (return coarse SVG unchanged).
