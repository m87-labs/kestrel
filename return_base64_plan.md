# Return Base64 Mask For Segmentation (Plan)

## Goal
Add an optional `return_base64=True` flag for segmentation requests (both:
1) direct Python calls via `InferenceEngine.segment(...)`, and
2) HTTP `/v1/segment`

When `return_base64` is true, the response should include:
- the existing SVG mask output (current behavior), and
- a base64-encoded bitmap mask (PNG) aligned to the input image.

This must be:
- clean and minimal (default behavior unchanged),
- performance-aware (no extra work unless requested),
- consistent across Python + HTTP.

---

## Current State (What Happens Today)

### 1) Model output: SVG path + bbox
`kestrel/skills/segment.py` builds segmentation output in `SegmentSkillState.finalize()`:
- decodes model-emitted SVG path tokens into `svg_path`
- builds `bbox` from coord/size tokens
- returns `result.output["segments"][0]["svg_path"]` and `["bbox"]`

Key code:
- `kestrel/skills/segment.py`: `SegmentSkillState.finalize()`
  - `svg_path_from_token_ids(...)`
  - `_build_bbox(...)`
  - output dict fields: `svg_path`, `bbox`, `coarse_path`, `coarse_bbox`, etc.

### 2) Optional refinement: bitmap mask -> traced SVG
If refinement deps exist and an image is present, `SegmentSkillState.finalize()` calls:
```py
refined_path, refined_bbox = runtime.seg_refiner(image, svg_path, bbox)
```
`runtime.seg_refiner` is a `SegmentRefiner` instance (`kestrel/seg_refiner.py`).

Inside `SegmentRefiner.__call__()`:
1. Render the coarse SVG path into a full-resolution bitmap mask (via `resvg`).
2. Crop to bbox region (+ margin).
3. Run a refinement head to improve the mask (GPU).
4. Paste back to full image.
5. Post-process and binarize.
6. Convert refined bitmap mask to an SVG path (via `potrace`) and return it.

Key code:
- `kestrel/moondream/runtime.py`: creates `runtime.seg_refiner` when `_HAS_SEG_DEPS`
- `kestrel/seg_refiner.py`: `SegmentRefiner.__call__()` constructs `refined_mask` then `bitmap_to_path(refined_mask)`

### 3) HTTP response shaping
`kestrel/server/http.py` `/v1/segment`:
- parses JSON payload
- calls `engine.segment(...)` (non-stream) or `engine.submit_streaming(...)` (stream)
- returns a simplified response:
  - `path` (from `segments[0]["svg_path"]`)
  - `bbox`

Key code:
- `kestrel/server/http.py`: `_ServerState.handle_segment()`

---

## Proposed API (Minimal + Backwards Compatible)

### A) Direct Python API
Add a new keyword-only argument:
```py
await engine.segment(image, object, return_base64=True, ...)
```

Rationale:
- clear intent: output formatting flag, not a sampling parameter
- backwards-compatible: new kw-only arg with default `False`

### B) HTTP API
Accept a new top-level boolean field:
```json
{
  "object": "dog",
  "image_url": "<base64 image bytes>",
  "return_base64": true,
  "settings": { ... },
  "spatial_refs": ...
}
```

Rationale:
- keeps `settings` reserved for decoding/sampling
- avoids silently ignoring an output-control flag inside `settings`

### C) Output shape
Keep existing output unchanged by default.

When `return_base64=true`, add a new response field:
- For Python: `result.output["segments"][0]["mask_base64"]`
- For HTTP: `mask_base64` at the top-level response (mirrors the existing `path`/`bbox` flattening)

Mask encoding:
- `mask_base64` is a base64-encoded PNG (single-channel grayscale) where:
  - background = 0
  - foreground = 255
  - dimensions match the decoded input image (H x W)

Rationale:
- PNG is compact for binary masks, widely supported, and self-describing (width/height embedded)
- keeping it as base64 makes it JSON-transport friendly for HTTP and consistent with existing image base64 conventions in the codebase

---

## Required Code Changes (End-to-End)

### 1) Request plumbing: carry `return_base64` to the skill

#### 1.1 `kestrel/skills/segment.py`
Change: extend `SegmentRequest` dataclass to include a `return_base64: bool = False` field.

Why:
- `SegmentSkillState.finalize()` only sees `self._request`; it needs the flag to decide whether to compute/attach the base64 mask.

Notes:
- Add the field at the end of the dataclass (it must have a default to avoid breaking construction).
- No default-behavior changes.

#### 1.2 `kestrel/engine.py`
Change: add `return_base64: bool = False` kw-only arg to `InferenceEngine.segment(...)`.
Propagate into `SegmentRequest(..., return_base64=return_base64)`.

Why:
- provides direct-call support with clean call-site ergonomics
- keeps `settings` reserved for decoding params and adapter selection

#### 1.3 `kestrel/server/http.py`
Change: parse `return_base64` from the JSON body in `/v1/segment`.

Implementation:
- `return_base64 = _parse_bool(payload.get("return_base64", False), "return_base64")`
- Pass it through:
  - streaming: include it in the `SegmentRequest(...)` passed to `submit_streaming`
  - non-stream: call `engine.segment(..., return_base64=return_base64)`

Why:
- explicit, opt-in behavior for large payloads
- default remains unchanged

Optional input validation (recommended for API clarity):
- If `return_base64=True` and no image is provided, return HTTP 400.
  - Reason: a bitmap mask cannot be defined without image dimensions.

---

### 2) Producing the base64 mask (where it should live)

#### Design choice (recommended)
Compute the bitmap mask in `SegmentRefiner` and return it (encoded) to the skill.

Why this is the cleanest approach:
- the bitmap mask already exists inside `SegmentRefiner.__call__()` (both coarse and refined)
- avoids re-rendering the SVG to a bitmap a second time (extra cost and potential mismatch)
- keeps optional heavy deps (`cv2`, `PIL`, `resvg`) encapsulated within `kestrel/seg_refiner.py`

#### 2.1 `kestrel/seg_refiner.py`: add a new method
Add a new method on `SegmentRefiner`, keeping `__call__` unchanged for compatibility:

Example shape (exact names TBD during implementation):
```py
@dataclass(slots=True)
class SegmentRefineResult:
    refined_svg_path: Optional[str]
    refined_bbox: Optional[dict]
    mask_base64: Optional[str]

def refine(
    self,
    image: np.ndarray | bytes,
    svg_path: str,
    bbox: dict,
    *,
    return_mask_base64: bool = False,
) -> SegmentRefineResult:
    ...

def __call__(...) -> Tuple[Optional[str], Optional[dict]]:
    result = self.refine(..., return_mask_base64=False)
    return result.refined_svg_path, result.refined_bbox
```

Why:
- avoids changing the `__call__` return type (low risk / unintrusive)
- provides a structured place to add new optional outputs

#### 2.2 Encoding helper (inside `kestrel/seg_refiner.py`)
Add a private helper (kept local to seg deps):
```py
def _mask_to_base64_png(mask_u8: np.ndarray) -> str:
    # mask_u8 is 0/1 or 0/255, single-channel uint8
```

Implementation preference:
- use `cv2.imencode(".png", mask_255)` if `cv2` is present (it is when `_HAS_SEG_DEPS`)
- fall back to PIL if needed (also part of seg deps)

Why:
- avoids introducing new dependencies in core `kestrel/utils/image.py`
- keeps “segmentation optional deps” contained in one module

#### 2.3 Coarse vs refined mask selection
Within `SegmentRefiner.refine(..., return_mask_base64=True)`:
- Always compute `coarse_mask` (already done today).
- Attempt refinement; if refinement *and* `bitmap_to_path(refined_mask)` succeeds:
  - return `refined_svg_path/refined_bbox`
  - base64-encode `refined_mask`
- Otherwise:
  - return `(None, None)` for refined path/bbox (so caller keeps coarse SVG)
  - base64-encode `coarse_mask`

Why:
- guarantees `mask_base64` always corresponds to the SVG returned to the user:
  - refined SVG -> refined mask
  - fallback coarse SVG -> coarse rasterized mask
- avoids the subtle mismatch where refinement creates a bitmap but path tracing fails (potrace missing) and the caller falls back to the coarse SVG.

---

### 3) Attach `mask_base64` to the skill output only when requested

#### 3.1 `kestrel/skills/segment.py`
Change: in `SegmentSkillState.finalize()`:
- read `want_base64 = self._request.return_base64`
- if `want_base64` and `runtime.seg_refiner` is available:
  - call `runtime.seg_refiner.refine(..., return_mask_base64=True)`
  - update `svg_path/bbox` if `refined_*` present (same as today)
  - set `segment["mask_base64"] = result.mask_base64` (only when requested)

Why:
- keeps the default response unchanged
- ensures the base64 payload is only computed/serialized when explicitly requested
- keeps output construction centralized in the skill (consistent with existing `segment` dict fields)

Edge behavior (recommended):
- If `want_base64=True` but `runtime.seg_refiner is None` (missing optional deps):
  - omit `mask_base64` and include a short `mask_error` string, OR
  - omit silently.

I recommend including an error string only if the caller opted in (so the API is self-explanatory),
but the simplest implementation is to omit the field. This is a small decision to confirm before implementation.

---

### 4) HTTP response: return the mask only when asked

#### 4.1 Non-streaming `/v1/segment`
Change: include `mask_base64` in JSONResponse only if `return_base64` was true.

Why:
- prevents large response bodies by default
- preserves the existing API shape for current clients

#### 4.2 Streaming `/v1/segment?stream=true`
Change: include `mask_base64` in the final SSE payload only when requested.

Why:
- the mask is only available at finalize time anyway
- avoids sending large base64 data in incremental deltas

---

## Edge Cases + Expected Behavior

1) `return_base64=false` (default)
- No change to outputs.
- No additional compute.

2) `return_base64=true` but segmentation deps are missing (`runtime.seg_refiner is None`)
- Recommended behavior: still return SVG path and bbox, but no base64 mask.
- Optionally include a clear `mask_error` message (only when requested).

3) `return_base64=true` but `image` is missing
- Recommended:
  - Python API: raise `ValueError` early (cannot produce bitmap without dimensions).
  - HTTP API: return 400.

4) Model output parse error (`parse_error` set, `svg_path==""`)
- Return current fields (`parse_error`, empty `path`).
- Do not attempt to produce `mask_base64`.

5) Empty masks
- If the coarse rasterized mask is empty, refinement already returns `(None, None)`.
- For base64: omit `mask_base64` (or return `null`) since a PNG would be meaningless without a defined foreground.

---

## Performance Considerations

- Base64 output should be strictly opt-in.
- Avoid double rasterization:
  - do not render the refined SVG back to a bitmap just to emit `mask_base64`
  - reuse the bitmap mask already computed during refinement / coarse rendering
- Encode PNG from a single-channel `uint8` array for compactness.
- Keep the returned mask full-resolution to match the input image (most useful, avoids ambiguity).

---

## Testing + Validation Plan

Because full segmentation/refinement requires heavyweight optional deps and model weights, focus testing on:

1) Unit tests (lightweight)
- Verify request plumbing:
  - `SegmentRequest` default `return_base64 == False`
  - `InferenceEngine.segment(..., return_base64=True)` populates the request context (can be tested by isolating request building if possible)
  - `/v1/segment` rejects non-boolean `return_base64`

2) Optional-deps tests (skipped if deps missing)
- Test `_mask_to_base64_png` roundtrip:
  - create a small synthetic mask (e.g. 10x10 square)
  - encode to base64 PNG
  - decode base64, load PNG (via PIL), confirm pixels match expected 0/255

3) Manual integration check (recommended)
- Run `/v1/segment` with `return_base64=true` on a known image/object:
  - confirm `mask_base64` decodes to an image of same dims as input
  - visually overlay mask on the source image
  - confirm `path` still works as before

---

## Implementation Steps (Order)

1) Add `return_base64` field to `SegmentRequest` (`kestrel/skills/segment.py`).
2) Add `return_base64` kw-only parameter to `InferenceEngine.segment` and propagate (`kestrel/engine.py`).
3) Update HTTP handler parsing + pass-through + conditional response field (`kestrel/server/http.py`).
4) Extend `SegmentRefiner` with `refine(..., return_mask_base64=...)` and PNG base64 helper (`kestrel/seg_refiner.py`).
5) Update `SegmentSkillState.finalize()` to request/attach `mask_base64` only when requested (`kestrel/skills/segment.py`).
6) Add unit tests for encoding helper and request parsing (where feasible).
7) Update README snippet for segmentation to mention `return_base64` (optional but recommended to document the feature).

---

## Decisions To Confirm Before Coding

1) Field naming:
- Recommend: `mask_base64`

2) Error behavior when `return_base64=true` but base64 cannot be produced:
- Option A (simplest): omit `mask_base64`
- Option B (more explicit): include `mask_base64: null` + `mask_error: "..."` (only when requested)

3) HTTP strictness:
- Should `/v1/segment` return 400 when `return_base64=true` and `image_url` missing?
  - I recommend yes (clear contract), but it is a behavior choice.

