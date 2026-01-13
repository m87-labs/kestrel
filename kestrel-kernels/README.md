# kestrel-kernels

Prebuilt CUDA kernels for Kestrel. Optimized for H100 (SM90a) inference.

## Kernel Library

### CUDA Kernels (compiled via CMake)

These kernels are implemented in CUDA C++ and compiled during wheel build.

#### `activation` - GELU Residual Activation
Computes `GELU(h) * (g + 1)` fused activation for gated architectures.
- **Input**: BF16 tensors
- **Features**: Vectorized 16-byte loads/stores, handles arbitrary shapes

#### `fused_linear_residual` - Linear + Bias + Residual
Fused `out = x @ W.T + bias + residual` using cuBLASLt epilogues.
- **Input**: BF16 tensors
- **Features**: Single kernel launch eliminates intermediate buffers

#### `fused_mlp` - Fused MLP with cuBLASLt
Fused MLP computation using cuBLASLt with custom epilogues.
- **Input**: BF16 tensors
- **Features**: Workspace caching, automatic heuristic selection

#### `kv_cache_write` - KV Cache Write with Optional FP8
Writes key/value tensors to paged KV cache with optional FP8 quantization.
- **Input**: BF16/FP16 keys and values
- **Output**: BF16/FP16 or FP8 (e4m3fn) KV cache
- **Features**: Vectorized writes, per-head scaling for FP8

#### `layernorm_cuda` - Fast LayerNorm Forward
Optimized LayerNorm forward pass for common hidden dimensions.
- **Input**: BF16 tensors
- **Specialized for**: N=1152, N=2048 (4 rows/block, warp reductions)
- **Features**: Vectorized 16-byte loads, two epilogue strategies for different occupancy tradeoffs

#### `moe_sum` - MoE Output Summation
Fast reduction over top-k expert outputs.
- **Input**: BF16 tensors, topk=8
- **Features**: Fully unrolled k=8 reduction, vectorized 16-byte memory ops

#### `rotary_embedding` - GPT-NeoX Rotary Position Embedding
Applies rotary position embedding to query and key tensors.
- **Input**: BF16 query/key, FP32 cos/sin cache
- **Supports**: GPT-NeoX layout (split rotary dims)
- **Features**: FP32 math with BF16 output, vectorized pair processing

#### `fp8_quant` - FP8 Quantization
Converts BF16 tensors to FP8 (e4m3fn) with dynamic scale computation.
- **Input**: BF16 tensors
- **Output**: FP8 (e4m3fn) tensors with scales
- **Modes**: Per-tensor or per-row scale computation
- **Features**: Vectorized 16-byte stores, fused absmax reduction

#### `tau_tail` - TAU Attention Bias
Applies TAU (Token-Aware Unet) position-dependent attention biases.
- **Input**: FP16/BF16 QKV tensors
- **Features**: Warp-per-head processing, vectorized loads

---

### CuTe DSL Kernels (precompiled for wheel distribution)

These kernels are written in NVIDIA CuTe DSL (Python) and precompiled to `.so` files during wheel build. The kernel source templates are excluded from wheel distribution.

#### `topk` - Bitonic Top-K Selection
Fast GPU top-k selection using bitonic sort network.
- **Input**: BF16 scores
- **Output**: Top-k values (optionally with softmax) and indices
- **Features**: Warp-level bitonic sort, optional fused softmax

**Python API:**
```python
from kestrel_kernels.topk import topk_fwd

values, indices = topk_fwd(scores, k=8, softmax=True)
```

#### `cute_moe` - MoE Matrix Multiplications
Fused Mixture-of-Experts up/down projections optimized for H100.
- **Variants**: BF16 warp, BF16 WGMMA, FP8 warp, FP8 WGMMA
- **Operations**: Up projection (gate + up), Down projection
- **Features**: Expert-parallel block scheduling, configurable tile sizes

**Python API:**
```python
from kestrel_kernels import (
    invoke_cute_moe_up,
    invoke_cute_moe_down,
    invoke_cute_moe_up_fp8,
    invoke_cute_moe_down_fp8,
)

# BF16 up projection
out_up = invoke_cute_moe_up(
    hidden_states, w1, w2,
    topk_weights, topk_ids,
    sorted_token_ids, expert_ids, num_tokens_post_pad,
)

# BF16 down projection
out_down = invoke_cute_moe_down(
    moe_out, w3,
    topk_weights, topk_ids,
    sorted_token_ids, expert_ids, num_tokens_post_pad,
)
```

#### `moe_align` - MoE Token Alignment
Prepares sorted token indices for block-sparse MoE operations.
- **Input**: Expert assignments (topk_ids)
- **Output**: Sorted token IDs, expert block IDs, padded token count
- **Variants**: Standard and LoRA-aware versions

**Python API:**
```python
from kestrel_kernels.moe_align import moe_align_block_size

moe_align_block_size(
    topk_ids, num_experts, block_size,
    sorted_token_ids, expert_ids, num_tokens_post_pad,
    expert_map,  # optional for expert parallelism
)
```

#### `flash_attn` - Flash Attention (Prefill & Decode)
High-performance Flash Attention implementation for H100.

**Prefill kernels:**
- `FlashAttentionFwd` - Standard prefill attention
- Supports: Causal masking, variable-length sequences, paged KV cache, FP8 KV

**Decode kernels:**
- `FlashAttentionDecodeSm90` - Single-split decode
- `FlashAttentionDecodeSm90PersistentSplitFused` - Multi-split persistent decode (better for long sequences)
- Supports: Paged KV cache, FP8 KV, GQA/MQA

**Python API:**
```python
from kestrel_kernels.flash_attn.cute import flash_attn_func, flash_attn_varlen_func

# Fixed-length attention
out = flash_attn_func(q, k, v, causal=True)

# Variable-length attention
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True,
)
```

---

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

---

## Build Notes

- Default CUDA arch: SM90a (H100). Override with `-DKESTREL_KERNELS_CUDA_ARCH=...` via `CMAKE_ARGS`.
- Some kernels (topk, cute_moe, moe_align, flash_attn decode) are precompiled during wheel build and require an H100 GPU.
- Other kernels (flash_attn prefill/backward) may JIT-compile at runtime if not precompiled.

## Building a Wheel

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

## Installing from GitHub Release

```bash
pip install https://github.com/m87-labs/kestrel/releases/download/kestrel-kernels-nightly/kestrel_kernels-0.1.0-cp310-cp310-linux_x86_64.whl
```

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens:
   - PyPI: Account Settings → API tokens → Add API token
   - TestPyPI: Same process on test.pypi.org

### Platform Support

- **OS**: Linux only (H100s run on Linux servers)
- **CPU**: x86_64 (works on both Intel and AMD - same wheel)
- **GPU**: H100 (SM90) required at runtime
- **Python**: 3.10, 3.11, 3.12, 3.13

### Building Wheels (requires H100 GPU)

Wheels must be built on a machine with an H100 GPU. We build separate wheels for each Python version.

```bash
# On H100 machine (e.g., p1)
cd ~/code/kestrel/kestrel-kernels

# Install all Python versions
~/.local/bin/uv python install 3.10 3.11 3.12 3.13

# Build wheels for all Python versions
mkdir -p dist-all
for pyver in 3.10 3.11 3.12 3.13; do
    echo "Building for Python $pyver..."
    TORCH_CUDA_ARCH_LIST=9.0a CUDACXX=/usr/local/cuda/bin/nvcc \
        ~/.local/bin/uv run --python $pyver python -m build --wheel
    mv dist/*.whl dist-all/
done

# Verify wheels were created
ls -la dist-all/
# Should show:
#   kestrel_kernels-0.1.0-cp310-cp310-linux_x86_64.whl
#   kestrel_kernels-0.1.0-cp311-cp311-linux_x86_64.whl
#   kestrel_kernels-0.1.0-cp312-cp312-linux_x86_64.whl
#   kestrel_kernels-0.1.0-cp313-cp313-linux_x86_64.whl
```

### Test Upload (TestPyPI)

Always test on TestPyPI first:

```bash
# Install twine
~/.local/bin/uv pip install twine

# Upload to TestPyPI
~/.local/bin/uv run twine upload --repository testpypi dist-all/*

# Test installation in a fresh environment
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ kestrel-kernels
```

### Production Upload (PyPI)

```bash
~/.local/bin/uv run twine upload dist-all/*
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
~/.local/bin/uv run twine upload dist-all/*
```

### Versioning

Update version in `pyproject.toml` before each release:

```toml
[project]
version = "0.1.0"  # Increment for each release
```

### Quick Release Script

For convenience, here's a complete release script:

```bash
#!/bin/bash
set -e

VERSION=${1:?Usage: $0 <version>}
cd ~/code/kestrel/kestrel-kernels

# Clean previous builds
rm -rf dist dist-all build *.egg-info
mkdir -p dist-all

# Build for all Python versions
for pyver in 3.10 3.11 3.12 3.13; do
    echo "=== Building for Python $pyver ==="
    TORCH_CUDA_ARCH_LIST=9.0a CUDACXX=/usr/local/cuda/bin/nvcc \
        ~/.local/bin/uv run --python $pyver python -m build --wheel
    mv dist/*.whl dist-all/
    rm -rf dist build *.egg-info
done

echo "=== Built wheels ==="
ls -la dist-all/

echo "=== Upload to PyPI? (ctrl-c to cancel) ==="
read -p "Press enter to upload..."
~/.local/bin/uv run twine upload dist-all/*
```
