# Changelog

All notable changes since `v0.1.2` are documented in this file.

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
