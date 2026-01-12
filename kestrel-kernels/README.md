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
TORCH_CUDA_ARCH_LIST=9.0a CUDACXX=/usr/local/cuda/bin/nvcc ~/.local/bin/uv run python -m build --wheel

# Wheel is output to dist/
ls dist/*.whl
```

**Important:** `TORCH_CUDA_ARCH_LIST=9.0a` ensures precompiled kernels use the `sm90a` architecture string, which is required for production deployments.

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

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens:
   - PyPI: Account Settings → API tokens → Add API token
   - TestPyPI: Same process on test.pypi.org
3. Install build tools:
   ```bash
   pip install build twine
   ```

### Build Process (requires H100 GPU)

The wheel must be built on a machine with an H100 GPU to compile the CUDA kernels.

```bash
# 1. Precompile kernels (requires H100)
python -m scripts.precompile

# 2. Build wheel
python -m build --wheel

# 3. Verify wheel contents (no kernel source code)
unzip -l dist/kestrel_kernels-*.whl | grep -E "\.py$"
# Should NOT show: cute_moe_*.py, flash_*.py, etc.
```

### Test Upload (TestPyPI)

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ kestrel-kernels
```

### Production Upload (PyPI)

```bash
# Upload to PyPI
twine upload dist/*
```

### Using API Tokens

Store your API token in `~/.pypirc`:

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
twine upload dist/*
```

### Versioning

Update version in `pyproject.toml` before each release:

```toml
[project]
version = "0.1.0"  # Increment for each release
```

### CI/CD Publishing (Optional)

For automated releases, add to GitHub Actions:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  build:
    runs-on: [self-hosted, gpu]  # Requires H100 runner
    steps:
      - uses: actions/checkout@v4
      - name: Build wheel
        run: |
          pip install build
          python -m scripts.precompile
          python -m build --wheel
      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```
