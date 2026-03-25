# Changelog

All notable changes since `v0.1.2` are documented in this file.

## 0.2.1 — 2026-03-25

### Performance

- **Up to ~55% throughput improvement on H100** at batch sizes 16–64, driven
  by kernel dispatch overhead reduction, allocation elimination across the
  decode path, and improved prefill scheduling leading to higher prefix cache
  hit rates. Moondream 2 at batch=64: 43.6 → 62.8 req/s (+44%);
  Moondream 3: 37.5 → 58.0 req/s (+55%).
- **Replace `F.linear` with cublasLt** — All attention projections, MoE
  router, and LM head now use a cublasLt linear kernel with plan caching,
  bypassing PyTorch's cuBLAS dispatch overhead (~10 µs per call). 1.1–2.8×
  faster than `F.linear` for M ≤ 512.
- **Fused O-projection + residual add** — Replaces the separate attention
  output projection and `aten::add` with a single `fused_linear_bias_residual`
  cublasLt call, saving ~6 µs per layer per decode step.
- **Prefill scratch buffer pool** — Pre-allocates reusable scratch buffers for
  QKV, tau, and router outputs during prefill, eliminating 77 `torch.empty`
  allocations per prefill.
- **Precompiled CuTe GELU kernel** — Replaces `F.gelu` in the tau attention
  path with a vectorized CuTe DSL kernel and pre-allocated scratch buffer.
- **Smarter page-table sync** — Decode steps that don't modify page tables
  skip the H2D commit entirely; dirty-row tracking avoids redundant copies.
- **Release allocator cache before CUDA graph capture** — Frees PyTorch
  allocator slack before graph capture, preventing OOM on memory-constrained
  systems.
- **Improved prefill scheduling** — Just-in-time classification seeds prefix
  cache misses when launch capacity allows, harvests reusable prefills first
  otherwise, and limits one uncached request per image cohort per batch.

### Changed

- **HTTP server removed** — The built-in HTTP server (`kestrel serve`) and its
  `starlette`, `uvicorn`, and `transformers` dependencies have been removed.
  Kestrel is now library-only; use `InferenceEngine` directly from Python.
- **`api_key` parameter on `InferenceEngine.create`** — Accepts an optional
  `api_key` keyword argument, falling back to the `MOONDREAM_API_KEY`
  environment variable if not provided.
- **Simplified licensing** — Replaced license verification with per-token
  billing via the telemetry endpoint. Startup raises `RuntimeError` for
  missing/invalid keys instead of `SystemExit`, and warns (but starts) if the
  API is unreachable.
- **Unified wheel for all Linux platforms** — `kestrel-kernels` is now a single
  universal wheel covering x86_64, aarch64 (Jetson), and GH200. The separate
  `jetson-jp6`/`jetson-jp61`/`jetson-jp62` install extras have been removed.
- **H200 and GH200 support** — Added MoE tuning configs and flash-attention
  SM-count mappings for H200 and GH200.

### Proprietary Stack (`kestrel-proprietary v0.2.0 → v0.2.1`)

- Cut native kernel dispatch overhead with a CPython C extension bridge, TVM
  FFI dispatch, faster stream/device metadata lookup, and Python-side wrapper
  validation — eliminating the PyTorch C++ ABI dependency from CMake kernels.
- Added a cublasLt linear projection kernel with plan caching and scratch
  buffer pooling, and a shared cublasLt workspace across linear and fused MLP
  kernels.
- Added a precompiled plain GELU CuTe DSL kernel with optional scratch-buffer
  reuse (2.15× faster than `F.gelu` on GH200 decode).
- Precompile CuTe MoE for all shipped configs via a new manifest-driven build,
  with added H200 and GH200 tuning configs.
- Universal aarch64 wheel builds replacing per-JetPack Jetson variants.

## 0.2.0 — 2026-03-18

### New Features

- **A100, A40, A10, and RTX 3090 support** — Kestrel now runs on SM80 and
  SM86 GPUs with precompiled attention, rotary, KV-cache, and FP8 kernels.
- **Jetson Orin support** — SM87 kernels for Jetson AGX Orin / Orin NX,
  with JetPack 6.1 and 6.2 dependency groups (`pip install kestrel[jetson-jp61]`).
- **Triton Inference Server backend** — deploy Kestrel as a Triton model
  with streaming support across all skills (query, caption, detect, point).

### Performance

- **~5.4x faster startup** (14.4s → 2.7s on H100).
- Faster LoRA inference via dense decode backend.
- Refreshed A100 and A10 benchmark results in PERFORMANCE.md.

### Bug Fixes

- Fixed eviction of prefix cache nodes that were still in use during prefill.
- Fixed reliability of engine pause/resume under concurrent GPU workloads.
- Fixed LoRA scaling for MoE expert adapters, improving training stability.
- Fixed top-p sampling edge case that could read out of bounds.

## 0.1.3 — 2026-02-20

### Performance

- ~26% end-to-end throughput improvement on the ChartQA benchmark (31.4 → 39.5 req/s on H100), driven by prefill scheduling rework, a faster precompiled sampling kernel, native image pipeline optimizations, and better MoE kernel tuning.
- Reworked prefill scheduling for better GPU utilization under load: token-budget packing, prepared-prefill queue top-up each cycle, skip-ahead batching of compatible requests, and active-slot-aware headroom with decode dispatch capping.
- Reduced peak memory during `.pt` weight loading by removing eager full-tensor conversion map. Moondream 3 now initializes in under 16 GB of RAM.

### Changed

- `nvcc` is no longer required at runtime. Sampling was migrated from flashinfer (which JIT-compiles CUDA kernels) to precompiled kernels in `kestrel-kernels`, enabling deployment on lighter-weight CUDA base images.

### Bug Fixes

- Fixed BOS handling by explicitly prepending BOS in skill prompt construction paths.

### Proprietary Stack (`kestrel-proprietary v0.1.2 → v0.1.3`)

- Added CuTe sampling kernel, enabled by default on CUDA for supported vocab sizes.
- Added flash-attention `seqused_q` support for batched varlen prefill (SM90 precompiled path and torch SDPA fallback, with regression tests).
- Improved `kestrel-native` image pipeline performance: optimized Lanczos resize, FIR rayon parallelism, libwebp decode fast path, tuned large-downscale thresholds.
- Expanded `kestrel-native` test coverage and CI compatibility for aarch64/x86_64.
- Updated H100/L4 MoE configs with ~2% faster MoE execution via improved [tuning cost model](https://x.com/vikhyatk/status/2023749843186078144).
