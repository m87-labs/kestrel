# kestrel-kernels

Prebuilt CUDA kernels for Kestrel.

This package is intended to be used as a local dependency from the Kestrel monorepo.

## Build notes

- Default CUDA arch: SM90a (H100). Override with `-DKESTREL_KERNELS_CUDA_ARCH=...` via `CMAKE_ARGS`.
- Kestrel assumes these kernels are built and importable; there is no runtime JIT fallback.
