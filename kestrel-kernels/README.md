# kestrel-kernels

Prebuilt CUDA kernels for Kestrel.

This package is intended to be used as a local dependency from the Kestrel monorepo.

## Build notes

- Default CUDA arch: SM90a (H100). Override with `-DKESTREL_KERNELS_CUDA_ARCH=...` via `CMAKE_ARGS`.
- Some kernels (topk, cute_moe, moe_align) are precompiled during wheel build and require an H100 GPU.
- Other kernels (flash_attn) are JIT-compiled at runtime.

## Building a wheel

Wheels must be built on a machine with an H100 GPU (e.g., p1) since precompilation requires CUDA.

```bash
# Install build dependencies
~/.local/bin/uv pip install build

# Build the wheel
cd ~/code/kestrel/kestrel-kernels
CUDACXX=/usr/local/cuda/bin/nvcc ~/.local/bin/uv run --extra dev python -m build --wheel

# Wheel is output to dist/
ls dist/*.whl
```

## Releasing to GitHub

We use a rolling `kestrel-kernels-nightly` tag for nightly builds.

**First release:**
```bash
gh release create kestrel-kernels-nightly dist/*.whl \
  --repo m87-labs/kestrel \
  --title "kestrel-kernels nightly" \
  --notes "Latest nightly build with precompiled kernels for H100 (SM90)" \
  --prerelease
```

**Update existing release:**
```bash
gh release upload kestrel-kernels-nightly dist/*.whl --clobber --repo m87-labs/kestrel
```

## Installing from GitHub release

```bash
pip install https://github.com/m87-labs/kestrel/releases/download/kestrel-kernels-nightly/kestrel_kernels-0.1.0-cp310-cp310-linux_x86_64.whl
```
