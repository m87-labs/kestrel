# Fused MoE wrapper

This package holds the Moondream module wrapper and expert-weight containers.
Kernel execution lives in `kestrel-kernels` behind the `kestrel_kernels.moe`
runtime API.

Key files:
- `module.py`: Module wrapper that normalizes expert weights, caches
  `MoeHandle`s, and calls `kestrel_kernels.moe.forward`.
- `weights.py`: Expert weight containers used when building Moondream layers.
- `routing.py`: Compatibility routing helpers used by existing MoE LoRA tests.
