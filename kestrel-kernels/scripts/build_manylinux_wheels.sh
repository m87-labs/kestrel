#!/usr/bin/env bash
set -euo pipefail

PY_VERSIONS=("3.10" "3.11" "3.12" "3.13")
PLAT="${PLAT:-manylinux_2_34_x86_64}"
# External deps provided by CUDA/PyTorch installs at runtime.
EXCLUDES=(
  "libtorch*.so"
  "libc10*.so"
  "libcuda.so*"
  "libcudart.so*"
  "libcublas*.so*"
  "libcublasLt.so*"
  "libnvrtc*.so*"
  "libcudnn*.so*"
  "libnccl*.so*"
  "libcuda_dialect_runtime.so*"
  "libtvm_ffi.so*"
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v auditwheel >/dev/null 2>&1; then
  echo "error: auditwheel not found. Install with: ~/.local/bin/uv tool install auditwheel" >&2
  exit 1
fi

rm -rf dist dist-raw dist-all build *.egg-info
mkdir -p dist-raw dist-all

for pyver in "${PY_VERSIONS[@]}"; do
  echo "=== Building for Python $pyver ==="
  rm -rf dist build *.egg-info
  TORCH_CUDA_ARCH_LIST=9.0a CUDACXX=/usr/local/cuda/bin/nvcc \
    ~/.local/bin/uv build --wheel --python "$pyver"

  raw_wheel="$(ls dist/*.whl)"
  mv "$raw_wheel" dist-raw/
  raw_wheel="dist-raw/$(basename "$raw_wheel")"

  echo "=== Repairing to $PLAT ==="
  EXCLUDE_ARGS=()
  for ex in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$ex")
  done
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    auditwheel repair --plat "$PLAT" -w dist-all "${EXCLUDE_ARGS[@]}" "$raw_wheel"
done

echo "=== Repaired wheels ==="
ls -la dist-all
