# Segmentation: Return Coarse + Refined Bitmap Masks (Plan + Working Notes)

## Goal
Extend segmentation output so callers can opt-in to receiving **two bitmap masks**:
1) **Coarse rasterized mask**: rasterization of the model-emitted SVG path *before refinement*.
2) **Refined mask**: bitmap mask *after refinement* (the mask used to derive the refined SVG path today).

This is in addition to the existing refined `svg_path` output (unchanged default behavior).

Scope includes:
- Direct Python calls (`InferenceEngine.segment(...)`).
- HTTP `/v1/segment` (streaming + non-streaming).

Primary constraints:
- Clean + minimal changes.
- No extra compute / payload unless explicitly requested.
- Keep current refined-SVG behavior as-is (don’t change refinement quality/logic unless required to emit masks).

---

## Current Code Path (Citations)

### Segment skill output + refinement hook
`SegmentSkillState.finalize()`:
- decodes model output to `svg_path`
- builds `bbox`
- optionally calls `runtime.seg_refiner(image, svg_path, bbox)` and replaces `svg_path/bbox` with refined versions

Code:
- Refinement callsite: `kestrel/skills/segment.py:154`–`160`
- Segment output payload construction: `kestrel/skills/segment.py:162`–`186`
- `SegmentRequest` currently contains only `object/image/stream/settings/spatial_refs`: `kestrel/skills/segment.py:37`–`46`

### Segment refiner builds coarse + refined bitmaps today (but only returns SVG path)
`SegmentRefiner.__call__()` currently:
- rasterizes model SVG to a coarse bitmap mask:
  - `full_svg = svg_from_path(...)`: `kestrel/seg_refiner.py:362`
  - `coarse_soft = render_svg_to_soft_mask(...)`: `kestrel/seg_refiner.py:363`
  - `coarse_mask = (coarse_soft > 0.5).astype(np.uint8)`: `kestrel/seg_refiner.py:364`
- refines that mask to `refined_mask`:
  - `refined_crop = self._refine_mask(...)`: `kestrel/seg_refiner.py:377`
  - `refined_mask = _paste_mask(...)`: `kestrel/seg_refiner.py:379`
  - `refined_mask = _clean_mask(...).astype(np.uint8)`: `kestrel/seg_refiner.py:380`
- traces `refined_mask` to `refined_path/refined_bbox`:
  - `result = bitmap_to_path(refined_mask)`: `kestrel/seg_refiner.py:385`
  - `return refined_path, refined_bbox`: `kestrel/seg_refiner.py:389`–`390`

Important: coarse/refined masks already exist in-memory; the only missing capability is returning them.

### HTTP `/v1/segment` response is flattened
Non-streaming response returns only `path` + `bbox`:
- call: `kestrel/server/http.py:647`–`652`
- response: `kestrel/server/http.py:663`–`674`

Streaming final SSE payload similarly returns only `path` + `bbox`:
- streaming request construction: `kestrel/server/http.py:575`–`592`
- final event payload: `kestrel/server/http.py:622`–`640`

### Optional dependency boundary
`runtime.seg_refiner` exists only when seg deps are installed:
- `self.seg_refiner = SegmentRefiner(...) if _HAS_SEG_DEPS else None`: `kestrel/moondream/runtime.py:490`–`493`

Implication: bitmap generation should remain within `SegmentRefiner` to avoid leaking optional deps into core paths.

---

## Proposed External API (Minimal + Consistent)

### Opt-in flag
Continue using the previously discussed opt-in: `return_base64: bool = False`
- Python: `await engine.segment(..., return_base64=True)`
- HTTP: JSON field `"return_base64": true`

Justification:
- Returning two full-resolution masks can significantly increase payload size; must be opt-in.
- Boolean flag is minimal and consistent with prior request.

### Output fields (base64 PNG)
When `return_base64=true`, include **two base64-encoded PNG masks**:
- `coarse_mask_base64`: rasterization of the model SVG (pre-refinement)
- `refined_mask_base64`: refined bitmap mask (post-refinement)

Placement:
- Python (`EngineResult.output["segments"][0]`):
  - add keys directly on the segment dict, alongside `svg_path`/`bbox`.
- HTTP (flattened response, consistent with current `path`/`bbox` flattening):
  - add top-level keys `coarse_mask_base64` and `refined_mask_base64`.
- Streaming SSE final event:
  - add those keys to the `"type": "final"` payload only.

Encoding contract:
- PNG grayscale, single channel, dimensions match the decoded input image (H x W).
- Pixels are `0` for background, `255` for foreground.
- Returned string is raw base64 (no data URL prefix).

Justification:
- PNG is compact for binary masks and broadly compatible.
- Keeping masks full-res avoids ambiguity and makes overlay easy.

---

## Proposed Internal Design (Optimal + Minimal)

### Principle
Compute and encode masks **inside `SegmentRefiner`**, because:
- It already rasterizes the coarse mask and creates the refined mask (`kestrel/seg_refiner.py:362`–`380`).
- It keeps optional deps (`cv2`, `PIL`, `resvg`) isolated behind `_HAS_SEG_DEPS` (`kestrel/moondream/runtime.py:490`–`493`).
- It avoids duplicating rasterization elsewhere (performance + correctness).

### Keep current `SegmentRefiner.__call__` signature stable
`SegmentSkillState.finalize()` currently calls `runtime.seg_refiner(...)` expecting `(Optional[str], Optional[dict])` (`kestrel/skills/segment.py:154`–`160`).

To keep changes unintrusive:
- Leave `__call__` return type unchanged.
- Add a new method (or a small result struct) that can optionally return masks.

Proposed structure:
- Add a `@dataclass(slots=True)` result type:
  - `refined_svg_path: Optional[str]`
  - `refined_bbox: Optional[dict]`
  - `coarse_mask_base64: Optional[str]`
  - `refined_mask_base64: Optional[str]`
- Add `SegmentRefiner.refine_with_bitmaps(..., return_base64: bool) -> SegmentRefineBitmapsResult`
- Implement `__call__` in terms of the new method with `return_base64=False` to preserve current behavior.

Justification:
- Lowest risk: existing callers keep working.
- Explicitly separates “refine SVG path” from “also emit bitmaps”.

---

## Detailed Change List (Every File)

### 1) `kestrel/skills/segment.py`

#### 1.1 Add opt-in to request context
Change:
- Extend `SegmentRequest` with `return_base64: bool = False` (add at the end).

Citations:
- Current `SegmentRequest` definition: `kestrel/skills/segment.py:37`–`46`

Justification:
- `SegmentSkillState.finalize()` only has access to `self._request` (`kestrel/skills/segment.py:102`–`110`) so this is the cleanest way to pass output-control flags down to refinement/output logic.
- Default remains unchanged.

#### 1.2 Attach both masks when requested
Change:
- In `SegmentSkillState.finalize()`:
  - read `want_masks = self._request.return_base64`
  - if `want_masks` and `svg_path/bbox/image` are present and `runtime.seg_refiner` exists:
    - call `runtime.seg_refiner.refine_with_bitmaps(image, svg_path, bbox, return_base64=True)`
    - if `refined_svg_path/refined_bbox` returned, replace `svg_path/bbox` exactly like today
    - attach `segment["coarse_mask_base64"] = ...` and `segment["refined_mask_base64"] = ...`

Citations:
- Current refinement + replacement logic: `kestrel/skills/segment.py:154`–`160`
- Segment dict assembly: `kestrel/skills/segment.py:162`–`170`

Justification:
- Keeps default behavior identical.
- Ensures we never compute/encode masks unless requested.
- Keeps all segmentation outputs centralized in the skill’s final payload.

Edge behavior (explicitly defined):
- If `want_masks=True` but refinement deps missing (`runtime.seg_refiner is None`), omit both mask fields.
- If `want_masks=True` but `image is None`, raise `ValueError` (Python) / return 400 (HTTP) because we cannot define full-res bitmaps without dimensions.

Why strictness is better:
- Prevents silent partial responses that look “successful” but omit the requested data.

### 2) `kestrel/seg_refiner.py`

#### 2.1 Add PNG base64 encoder helper
Change:
- Add `_encode_mask_png_base64(mask: np.ndarray) -> str` that:
  - accepts `uint8` (0/1 or 0/255) single-channel
  - converts to 0/255 `uint8`
  - encodes via `cv2.imencode(".png", mask_255)`
  - base64 encodes bytes

Citations:
- Coarse mask is `np.uint8` 0/1 today: `kestrel/seg_refiner.py:364`
- Refined mask is `np.uint8` 0/1 today: `kestrel/seg_refiner.py:380`

Justification:
- Uses `cv2` which is already required for refinement (`kestrel/seg_refiner.py:19`–`25`).
- Keeps encoding colocated with mask creation.

#### 2.2 Add `refine_with_bitmaps(...)`
Change:
- Implement a new method that returns both mask base64 strings (when requested) plus refined path/bbox.

Key logic details (important for correctness):
- Always compute `coarse_mask` from the model SVG (`kestrel/seg_refiner.py:362`–`365`).
- Always attempt refinement to produce `refined_mask` (`kestrel/seg_refiner.py:377`–`380`) when possible.
- Attempt `bitmap_to_path(refined_mask)` to produce the refined SVG path (`kestrel/seg_refiner.py:385`–`390`).

Critical behavior change (to support debug/inspection):
- Do NOT early-return before capturing `coarse_mask` when it’s empty (`kestrel/seg_refiner.py:366`–`367` today).
  - If `coarse_mask.sum() == 0`, refinement is pointless, but returning the coarse bitmap is useful for debugging model output / bbox mapping.

Also:
- Do NOT require `bitmap_to_path(...)` to succeed to return `refined_mask_base64`.
  - Tracing can fail (e.g. missing `potrace` import inside `bitmap_to_path`), but the refined bitmap still exists and is valuable.

Justification:
- User explicitly wants the bitmaps, independent of the SVG trace step.
- Preserves the existing “only replace SVG when tracing succeeded” behavior by returning `refined_svg_path/refined_bbox` only on success.

#### 2.3 Keep `__call__` API stable
Change:
- Rewrite `__call__` as:
  - call `refine_with_bitmaps(..., return_base64=False)`
  - return `(result.refined_svg_path, result.refined_bbox)`

Citations:
- Current `__call__` signature/return type: `kestrel/seg_refiner.py:341`–`351`

Justification:
- avoids touching all call sites
- keeps type expectations stable

### 3) `kestrel/engine.py`

#### 3.1 Add kw-only flag to `InferenceEngine.segment(...)`
Change:
- Add `return_base64: bool = False` kw-only argument to the signature.
- Validate:
  - if `return_base64` and `image is None`: raise `ValueError`
- Pass to `SegmentRequest(..., return_base64=return_base64)`.

Citations:
- Current signature and request construction: `kestrel/engine.py:648`–`692`

Justification:
- Clean API for direct callers (no “settings” misuse).
- Avoids silent missing mask outputs when no image is provided.

### 4) `kestrel/server/http.py`

#### 4.1 Parse `return_base64`
Change:
- In `handle_segment`, parse:
  - `return_base64 = _parse_bool(payload.get(\"return_base64\", False), \"return_base64\")`

Citations:
- Current payload parsing region: `kestrel/server/http.py:525`–`562`
- `_parse_bool` implementation: `kestrel/server/http.py:727`–`730`

Justification:
- Strong type validation (rejects `"true"` strings).
- Keeps flag top-level alongside `stream`, not buried in `settings`.

#### 4.2 Enforce image requirement when requested
Change:
- If `return_base64` and `image is None`: return 400 with clear error.

Justification:
- Matches Python behavior.
- Prevents clients from thinking they’ll get masks when it’s impossible.

#### 4.3 Pass flag to engine/skill
Non-streaming:
- Call `engine.segment(..., return_base64=return_base64, ...)` (requires engine signature change).

Streaming:
- Include `return_base64` in the `SegmentRequest(...)` passed to `submit_streaming` (`kestrel/server/http.py:575`–`586` today).

Justification:
- Both pathways use the same skill code path; this keeps output consistent.

#### 4.4 Add optional fields to responses
Non-streaming:
- If `return_base64`:
  - pull `coarse_mask_base64` and `refined_mask_base64` from `segment` output
  - include in `response_payload` (`kestrel/server/http.py:663`–`674`)

Streaming:
- If `return_base64`, include both fields in final SSE payload (`kestrel/server/http.py:627`–`640`).

Justification:
- Keeps default response identical.
- Avoids large payloads unless requested.

---

## Semantics + Edge Cases (Explicit)

1) `return_base64=false` (default)
- No new fields, no encoding work, no behavior change.

2) `return_base64=true` with seg deps available
- Always return `coarse_mask_base64` (even if empty), as long as rasterization succeeded.
- Return `refined_mask_base64` only if refinement produced a `refined_mask` (even if SVG trace fails).
- `svg_path/bbox` behavior remains current: replaced only when refined trace succeeded (`kestrel/skills/segment.py:158`–`160`).

3) `return_base64=true` but no image
- Python: raise `ValueError` in `InferenceEngine.segment`.
- HTTP: return 400.

4) `runtime.seg_refiner is None` (missing optional deps)
- Return SVG path as today, omit both mask fields.

5) Model parse error / empty SVG path
- Refinement is skipped today (`kestrel/skills/segment.py:154` condition includes `svg_path` and `not parse_error`).
- For masks: also skip and omit both mask fields.

---

## Performance Notes

- The expensive part (refinement) already exists; masks reuse intermediate arrays already computed.
- Extra work when requested:
  - PNG encode + base64 encode two grayscale images.
- No extra rendering passes (avoid re-rasterizing SVG outside `SegmentRefiner`).

---

## Testing Plan (Practical)

Unit-ish tests that don’t require model weights:
1) Test `_encode_mask_png_base64` roundtrip (skip if seg deps missing).
2) Test HTTP JSON validation rejects non-bool `return_base64`.
3) Test HTTP 400 when `return_base64=true` and `image_url` missing.

Integration testing (manual or optional CI job if env supports seg deps + weights):
- Request segmentation with `return_base64=true` and confirm:
  - both base64 strings decode to correct dimensions
  - coarse vs refined masks differ in expected cases

---

## Implementation Checklist (Working Notes)

- [x] Add `return_base64` to `SegmentRequest` (`kestrel/skills/segment.py`)
- [x] Add `return_base64` to `InferenceEngine.segment` + validation (`kestrel/engine.py`)
- [x] Parse + validate `return_base64` in `/v1/segment` (`kestrel/server/http.py`)
- [x] Plumb flag through streaming `SegmentRequest` (`kestrel/server/http.py`)
- [x] Add `SegmentRefiner.refine_with_bitmaps` + encoder helper (`kestrel/seg_refiner.py`)
- [x] Update `SegmentSkillState.finalize` to attach both masks when requested (`kestrel/skills/segment.py`)
- [x] Add tests (skip if deps missing) (`tests/...`)
- [x] Manual smoke test: direct `engine.segment(..., return_base64=True)` (real engine + seg weights)
- [ ] Manual smoke test: `/v1/segment` non-stream + stream (real engine + weights)

Remote test runs (SSH `/workspace/kestrel`):
- `pytest -q tests/test_seg_refiner_mask_encoding.py tests/test_segment_request_flags.py`
- `pytest -q tests/test_http_segment_return_base64.py`
- `pytest -q tests/test_segment_skill_mask_outputs.py`

---

## Bugs / Debug Log (Fill In During Implementation)

### Template
- Date:
- Trigger:
- Observed behavior:
- Expected behavior:
- Root cause:
- Fix:
- Debugging steps:

### 2026-02-15: `pip install potrace` Not Found (Linux)
- Trigger: Installing optional tracing deps for `bitmap_to_path()` on the SSH machine.
- Observed behavior: `ERROR: Could not find a version that satisfies the requirement potrace`.
- Expected behavior: A package providing `import potrace` should be installable.
- Root cause: PyPI distribution name is not `potrace` for the `potrace` module used here.
- Fix: Install `pypotrace` (it provides the `potrace` Python module).
- Debugging steps:
  - Tried `pip install potrace` (no distributions).
  - Tried `pip install pypotrace`, initially failed due to missing `libagg` headers.
  - Installed `libagg-dev` and retried; `pypotrace` built successfully and `import potrace` worked.

### 2026-02-15: `pypotrace` Build Failed (Missing `libagg`)
- Trigger: `pip install pypotrace` while preparing segmentation tracing deps.
- Observed behavior: build error mentioning missing `libagg` in `pkg-config` (`No package 'libagg' found`).
- Expected behavior: `pypotrace` builds a wheel successfully.
- Root cause: missing system package providing `libagg.pc` / headers.
- Fix: `apt-get install -y libagg-dev` then re-run `pip install pypotrace`.

### 2026-02-15: `resvg` Built From Source (Rust Toolchain Needed)
- Trigger: `pip install resvg` for SVG rasterization in `render_svg_to_soft_mask()`.
- Observed behavior: pip fetched `resvg-0.1.2.tar.gz` and built a wheel (no prebuilt wheel on that machine).
- Expected behavior: `resvg` installs successfully and provides `from resvg import render, usvg`.
- Root cause: platform/python combo required building from sdist.
- Fix: install Rust toolchain (`apt-get install -y rustc cargo`), then re-run `pip install resvg`.

### 2026-02-16: Segment Returned Empty SVG (Wrong Weights)
- Trigger: Running a real segmentation request (`engine.segment(image=..., object=\"dog\", return_base64=True)`) using the default downloaded `moondream/moondream3-preview:model_fp8.pt`.
- Observed behavior:
  - `segments[0].bbox` was present (coord/size tokens emitted),
  - but `segments[0].svg_path` was empty and `token_ids` contained only `[0]`,
  - therefore `coarse_mask_base64` and `refined_mask_base64` were `null` (refinement never ran because there was no SVG path).
- Expected behavior: Non-empty SVG path and both masks when `return_base64=true`.
- Root cause: The default `model_fp8.pt` weights are not the segmentation-capable checkpoint used in production. The fal deployment uses a dedicated segmentation checkpoint (`seg-model.pt`) from `vikhyatk/moondream-next` pinned by revision.
- Fix:
  - Download and use `seg-model.pt` (repo `vikhyatk/moondream-next`, revision `664f745b71962510348d84bf1269fd37665fb3fb`) as `RuntimeConfig(model_path=...)`.
  - Re-run segmentation; both masks are produced and coarse/refined differ.
- Debugging steps:
  - Compared with production deployment code in `fal_inference/sync_api.py` (engine weights mapping for `seg` stage).
  - Re-ran the same image+prompt using `seg-model.pt`; confirmed:
    - `mask_shape (1213, 1546)`,
    - `coarse_nonzero 484200`, `refined_nonzero 508642`,
    - `diff_pixels 69248`.

### 2026-02-16: Precompiled `kestrel-kernels` Needed Runtime Libs + Version Alignment
- Trigger: Initializing the engine on the H100 box after recreating a clean venv and installing `kestrel-kernels==0.1.2`.
- Observed behavior:
  - first failure: missing `libcuda_dialect_runtime.so` at load time for precompiled flash-attn decode,
  - later failure: symbol mismatch (`*_args_spec` lookup) when `nvidia-cutlass-dsl` resolved to a newer version than the repo lock.
- Expected behavior: precompiled flash-attn loads cleanly (or falls back).
- Root cause:
  - dynamic loader needs `LD_LIBRARY_PATH` to include vendored runtime libs (`nvidia_cutlass_dsl/lib`, `tvm_ffi/lib`),
  - `pip install kestrel-kernels==0.1.2` can pull `nvidia-cutlass-dsl` versions newer than `uv.lock` (>=4.3.4), and precompiled modules can be sensitive to that.
- Fix:
  - Set `LD_LIBRARY_PATH` to include:
    - `.venv/lib/python3.10/site-packages/nvidia_cutlass_dsl/lib`
    - `.venv/lib/python3.10/site-packages/tvm_ffi/lib`
  - Force reinstall `nvidia-cutlass-dsl==4.3.5` to match `uv.lock`.

### 2026-02-16: HQ-SAM Legacy Refiner Required `torchvision`
- Trigger: Enabling the legacy HQ-SAM refiner and initializing the runtime.
- Observed behavior: `ImportError` from `transformers` dynamic module loader stating the model requires `torchvision`.
- Expected behavior: HQ-SAM weights load and the legacy refiner initializes.
- Root cause: HQ-SAM model’s remote code depends on `torchvision`, but it was not installed in the venv.
- Fix:
  - Pin `torchvision==0.24.1` in `pyproject.toml` (matches `torch==2.9.1`) and regenerate `uv.lock`.
  - For the manual pip path on CUDA boxes, install a matching wheel without pulling a different torch:
    - `pip install --no-deps torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128`

### 2026-02-16: `pip install torchvision` Tried To Upgrade Torch (Broke Venv)
- Trigger: Running `pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu128` without pinning versions.
- Observed behavior: pip selected `torchvision-0.25.0+cu128` which depends on `torch-2.10.0+cu128`, uninstalled `torch-2.9.1+cu128`, then was interrupted mid-install leaving torch broken (`libtorch_global_deps.so` missing).
- Expected behavior: Install torchvision without altering the pinned torch version.
- Root cause: Unpinned `torchvision` resolution pulled a newer torch.
- Fix:
  - Force reinstall the pinned torch: `pip install --force-reinstall torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128`
  - Install pinned torchvision with `--no-deps` (see previous entry).

### 2026-02-16: `flashinfer` JIT Compilation Failed (Missing `ninja`)
- Trigger: After repairing the venv, engine warmup hit `flashinfer`’s JIT sampling module build path.
- Observed behavior: `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.
- Expected behavior: `flashinfer` JIT builds (or loads cached) sampling module successfully.
- Root cause: `ninja` build tool not installed on the machine (required by the JIT build pipeline).
- Fix: `apt-get update && apt-get install -y ninja-build`.
