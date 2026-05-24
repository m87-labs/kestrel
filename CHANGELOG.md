# Changelog

All notable changes since `v0.1.2` are documented in this file.

## 0.4.1 — 2026-05-23

This release lifts FP8 MoE performance on NVIDIA B200 through the
`kestrel-kernels` 0.4.1 bump, finishes the MoE LoRA migration onto
bundled CuTe kernels, and adds spatial references to point prompts.

### FP8 MoE performance on B200

- Decode FP8 MoE routes through a new warp-GEMV path that consumes
  top-k routing directly — roughly 3× faster at decode batch sizes than
  the previous CuTe MoE path.
- Prefill and high-batch decode FP8 MoE route through a new
  compact-route GEMM that drops expert-padded buffers — 1.15–2.40×
  faster at batches of 256–1024 routed tokens, lifting ChartQA
  end-to-end throughput by ~8% with ~16% lower P99 latency.

### MoE LoRA

- MoE LoRA execution moved end-to-end onto bundled CuTe kernels for
  both prefill and decode. The decode path is now CUDA-graph-stable
  through compact and token-major metadata writes into fixed-capacity
  slot buffers, and the legacy single-LoRA prefill / route-ID decode
  paths are removed. Triton is no longer pulled in transitively.

### Point prompts

- `InferenceEngine.point` accepts an optional `spatial_refs=` keyword
  argument to anchor point prompts to a referenced region (e.g.
  identifying which subject's gaze to follow). Mirrors the existing
  behavior in `query` and `segment`.

## 0.4.0 — 2026-05-18

This release fixes Apple Silicon installs breaking after PyTorch upgrades,
improves prefix-cache reuse and startup recovery, and adds rank-32 LoRA adapter
support.

### Apple Silicon install compatibility

- Apple Silicon installs now work across supported PyTorch 2.9–2.12 builds.
  Previously, the native MPS extension was tied to a single PyTorch minor
  version and could fail to import after a PyTorch upgrade.

### Prefix-cache and startup reliability

- Completed responses are now retained in the prefix cache, so follow-up
  requests that repeat or continue an earlier prompt start faster.
- A failed engine start, such as a warmup or API-key validation failure, no
  longer leaves the engine half-initialized. Starting again retries from a clean
  state.

### LoRA and MoE

- LoRA adapters now support rank 32, up from rank 16.
- MoE execution, including LoRA warmup and prefill, now runs through the shared
  `kestrel-kernels` runtime path used by the CUDA and Metal kernel backends.

## 0.3.1 — 2026-05-01

- **Python 3.14 support** across Linux x86_64 / aarch64, Windows x86_64,
  and macOS arm64.

## 0.3.0 — 2026-05-01

The previous release of kestrel ran on NVIDIA Linux only, Ampere
through Hopper. This release ships kestrel on four new platforms
simultaneously — **Apple Silicon**, **Windows (NVIDIA)**, **NVIDIA
Blackwell** (data-center and workstation), and **NVIDIA Jetson
Thor** — and it makes every existing target faster.

### Run on Apple Silicon

Kestrel now runs on Apple M-series Macs from macOS 13 (Ventura)
onward, on Python 3.12. `pip install kestrel` works on a stock
Apple Silicon Mac with no NVIDIA CUDA and no Triton — full
inference for both Moondream 2 and Moondream 3 against native
Metal kernels for the entire decode path (paged attention, rotary,
KV cache, MoE routing, sampling, layer norm). KV cache size
auto-tunes to your machine's unified memory.

Reference throughput (ChartQA, batch=4 direct mode):

| Hardware                       | MD2 req/s | MD3 req/s |
|--------------------------------|-----------|-----------|
| MacBook Pro (M5 Max, 48 GB)    | 7.26      | 4.58      |
| Mac mini (M2, 24 GB)           | 0.79      | 0.55      |
| Mac mini (M4 base, 16 GB)      | 0.84      | —         |

Full breakdown including batch=1/16 and CoT mode in `PERFORMANCE.md`.

### Run on Windows

Windows x86_64 is now a fully supported target — not just a
packaging change. We rewrote kestrel's kernel-loading runtime to
be cross-platform (MSVC compatibility, Windows DLL loading
semantics through the C extension layer, library-naming abstraction
across kestrel-kernels), so `pip install kestrel` on Python
3.10–3.13 installs and runs natively on Windows: no Linux
container, no WSL. Same CUDA kernels as Linux x86_64.

### Run on Blackwell

Both **B200** (data-center, sm_100) and **NVIDIA RTX PRO 6000**
(workstation, sm_120) are now supported. B200 is the fastest
hardware kestrel runs on:

| Batch | Moondream 2 (req/s) | Moondream 3 (req/s) |
|-------|---------------------|---------------------|
| 1     | 44.16               | 33.38               |
| 64    | 93.61               | 71.27               |

That's **1.49× H100** at MD2 B=64 and **1.23× H100** at MD3 B=64.
Behind those numbers: a Blackwell-tuned mixture-of-experts kernel
running up to **1.77× faster than the prior Triton baseline**, plus
dedicated Blackwell flash-attention kernels for both decode and
prefill (no portable-fallback paths).

RTX PRO 6000 hits 39.3 req/s (MD2) / 39.7 req/s (MD3) at B=64 —
measurably faster than L40S on a workstation card.

### Run on Jetson Thor

NVIDIA Jetson AGX Thor 64 GB (sm_110) is supported on JetPack 7
(CUDA 13 — required, since CUDA 12 can't target sm_110).
kestrel-kernels now ships a **multi-CUDA aarch64 wheel** that
bundles cu12 and cu13 builds, so the same install command works on
Thor (uses cu13) and on JetPack 6 systems running Jetson Orin or
GH200 (uses cu12).

| Batch | Moondream 2 (req/s) | Moondream 3 (req/s) |
|-------|---------------------|---------------------|
| 1     | 6.58                | 6.80                |
| 64    | 14.53               | 12.05               |

### Faster on existing NVIDIA hardware

- **Faster FP8 prefill on Ada and Jetson Orin** (the existing arches
  most users running with FP8 KV cache are on). New native paged
  flash-attention kernels — dense, paged, paged-varlen, and
  paged-prefix prefill — replace the previous prefill codepaths,
  visible end-to-end on L40S, RTX 4090, and Jetson Orin.
- **MoE inference faster on every GPU**, with new native FP8 MoE
  microkernels and per-card retunes (A100 SXM4 / 40 GB, A10 /
  A10G, L4, RTX 6000).
- **Lower per-call dispatch overhead** across every kestrel kernel
  (flash attention, MoE, GELU, FP8 quantization, sampling, layer
  norm). Most visible at low batch sizes, where per-step overhead
  used to dominate.
- **More consistent tail latency** on small-to-medium batches on
  Ada, Blackwell, and RTX 6000.

### Install matrix

Kestrel and its dependencies ship binary wheels for:

- Linux x86_64 — Python 3.10 – 3.13
- Linux aarch64 — Python 3.10 – 3.13 (single multi-CUDA wheel
  bundling cu12 + cu13: cu12 covers Jetson Orin / GH200 on
  JetPack 6, cu13 covers Jetson Thor on JetPack 7)
- Windows x86_64 — Python 3.10 – 3.13
- macOS arm64 — Python 3.12

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
