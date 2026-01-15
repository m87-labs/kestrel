# Development & Release

Instructions for developing, building, and publishing kestrel-kernels.

## Development Setup

### Prerequisites

- CUDA 12.x with SM90 support
- Python 3.10+
- PyTorch 2.9.1
- NVIDIA CuTe DSL 4.3.3

### Install from source

```bash
cd kestrel-kernels
CUDACXX=/usr/local/cuda/bin/nvcc uv sync --extra dev
```

### Running Tests

Tests require an H100 GPU (SM90+):

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_topk.py -v
```

### Tuning Kernels

The grid search script for CuTe MoE kernel autotuning is located at `scripts/grid_search_cute_moe.py` in the main kestrel repo. It sweeps tile sizes, warp counts, and pipeline stages to find optimal configurations.

```bash
# Run quick grid search for specific token counts
ssh p1 'cd ~/code/kestrel && KESTREL_CUTE_MOE_JIT=1 ~/.local/bin/uv run python \
  scripts/grid_search_cute_moe.py --num-tokens 8 16 32 --quick --output results.json'

# Run full grid search (takes hours)
ssh p1 'cd ~/code/kestrel && KESTREL_CUTE_MOE_JIT=1 ~/.local/bin/uv run python \
  scripts/grid_search_cute_moe.py --output /tmp/cute_moe_grid_search.json'
```

Results should be saved to `python/kestrel_kernels/configs/` after analysis.

## Platform Support

- **OS**: Linux only (H100s run on Linux servers)
- **CPU**: x86_64 (works on both Intel and AMD - same wheel)
- **GPU**: H100 (SM90) required at runtime
- **Python**: 3.10, 3.11, 3.12, 3.13

## Build Notes

- Default CUDA arch: SM90a (H100). Override with `-DKESTREL_KERNELS_CUDA_ARCH=...` via `CMAKE_ARGS`.
- Some kernels (topk, cute_moe, moe_align, flash_attn decode) are precompiled during wheel build and require an H100 GPU.
- Other kernels (flash_attn prefill/backward) may JIT-compile at runtime if not precompiled.
- `TORCH_CUDA_ARCH_LIST=9.0a` ensures precompiled kernels use the `sm90a` architecture string, which is required for production deployments.
- PyPI/TestPyPI require manylinux tags; use `auditwheel repair` to produce
  `manylinux_2_34_x86_64` wheels (see below).

## Building Wheels (requires H100 GPU)

Wheels must be built on a machine with an H100 GPU. We build separate wheels for each Python version.

```bash
# On H100 machine (e.g., p1)
cd ~/code/kestrel/kestrel-kernels

# Install all Python versions
~/.local/bin/uv python install 3.10 3.11 3.12 3.13

# Install auditwheel (for manylinux repair)
~/.local/bin/uv tool install auditwheel

# Build + repair wheels for all Python versions
./scripts/build_manylinux_wheels.sh

# Verify wheels were created
ls -la dist-all/
# Should show:
#   kestrel_kernels-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
#   kestrel_kernels-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
#   kestrel_kernels-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
#   kestrel_kernels-0.1.0-cp313-cp313-manylinux_2_34_x86_64.whl
```

## Releasing to GitHub

We use a rolling `kestrel-kernels-nightly` tag for nightly builds.

**First release:**
```bash
gh release create kestrel-kernels-nightly dist-all/*.whl \
  --repo m87-labs/kestrel \
  --title "kestrel-kernels nightly" \
  --notes "Latest nightly build with precompiled kernels for H100 (SM90)" \
  --prerelease
```

**Update existing release:**
```bash
gh release upload kestrel-kernels-nightly dist-all/*.whl --clobber --repo m87-labs/kestrel
```

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens:
   - PyPI: Account Settings → API tokens → Add API token
   - TestPyPI: Same process on test.pypi.org

### Test Upload (TestPyPI)

Always test on TestPyPI first:

```bash
# Install twine
~/.local/bin/uv tool install twine

# Upload to TestPyPI
~/.local/bin/uv tool run twine upload --repository testpypi dist-all/*

# Test installation in a fresh environment
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ kestrel-kernels
```

### Production Upload (PyPI)

```bash
~/.local/bin/uv tool run twine upload dist-all/*
```

### Using API Tokens

Configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-<your-token>

[testpypi]
username = __token__
password = pypi-<your-token>
```

Or use environment variables:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-token>
~/.local/bin/uv tool run twine upload dist-all/*
```

### Versioning

Update version in `pyproject.toml` before each release:

```toml
[project]
version = "0.1.0"  # Increment for each release
```

## Quick Release Script

For convenience, here's a complete release script:

```bash
#!/bin/bash
set -e

VERSION=${1:?Usage: $0 <version>}
cd ~/code/kestrel/kestrel-kernels

# Clean previous builds
rm -rf dist dist-all build *.egg-info
mkdir -p dist-all

# Ensure tools are installed
~/.local/bin/uv tool install auditwheel
~/.local/bin/uv tool install twine

# Build for all Python versions
for pyver in 3.10 3.11 3.12 3.13; do
    echo "=== Building for Python $pyver ==="
    rm -rf dist build *.egg-info
    TORCH_CUDA_ARCH_LIST=9.0a CUDACXX=/usr/local/cuda/bin/nvcc \
        ~/.local/bin/uv build --wheel --python $pyver
    raw_wheel=$(ls dist/*.whl)
    ~/.local/bin/uv tool run auditwheel repair --plat manylinux_2_34_x86_64 -w dist-all "$raw_wheel"
done

echo "=== Built wheels ==="
ls -la dist-all/

echo "=== Upload to PyPI? (ctrl-c to cancel) ==="
read -p "Press enter to upload..."
~/.local/bin/uv tool run twine upload dist-all/*
```
