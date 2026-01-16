# Release Process

## Prerequisites

- PyPI account with upload permissions for `kestrel`
- PyPI API token in `~/.pypirc`:
  ```ini
  [distutils]
  index-servers =
      pypi

  [pypi]
  username = __token__
  password = pypi-YOUR_TOKEN_HERE
  ```

## Publishing a New Version

### 1. Update the version

Edit `pyproject.toml` and bump the version:

```toml
[project]
version = "X.Y.Z"
```

### 2. Clean and build

```bash
rm -rf dist/ build/ *.egg-info
uv build
```

This creates:
- `dist/kestrel-X.Y.Z-py3-none-any.whl` (wheel - what users install)
- `dist/kestrel-X.Y.Z.tar.gz` (source distribution)

### 3. Verify the build

Check wheel contents:
```bash
unzip -l dist/*.whl
```

Check sdist contents:
```bash
tar -tzf dist/*.tar.gz
```

Ensure no unwanted files (tests, scripts, docs, assets, kestrel-kernels) are included.

### 4. Publish to PyPI

```bash
uv publish --username __token__ --password "$(grep -A2 '^\[pypi\]' ~/.pypirc | grep password | cut -d'=' -f2 | tr -d ' ')"
```

Or if you have the token in an environment variable:
```bash
uv publish --token "$PYPI_TOKEN"
```

### 5. Verify the release

```bash
curl -s https://pypi.org/pypi/kestrel/json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['info']['version'])"
```

Or visit: https://pypi.org/project/kestrel/

## Testing on TestPyPI (Optional)

Before publishing to production PyPI:

```bash
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --username __token__ \
  --password "$(grep -A2 '^\[testpypi\]' ~/.pypirc | grep password | cut -d'=' -f2 | tr -d ' ')"
```

Install from TestPyPI to verify:
```bash
pip install --index-url https://test.pypi.org/simple/ kestrel
```

## Package Contents

The published package includes only:
- `kestrel/` - Main Python package
- `README.md` - PyPI description
- `LICENSE.md` - License file

Excluded via `MANIFEST.in`:
- `tests/`
- `scripts/`
- `docs/`
- `assets/`
- `kestrel-kernels/` (separate package)

## Dependencies

The `kestrel-kernels` package is a separate dependency that must be published independently. See `kestrel-kernels/RELEASE.md` for its release process.
