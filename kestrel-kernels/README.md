# kestrel-kernels

Precompiled CUDA kernels for [Kestrel](https://github.com/m87-labs/kestrel), a high-performance inference engine for [Moondream](https://moondream.ai), the world's most efficient vision-language model.

These kernels are optimized for NVIDIA H100 (SM90) and distributed as precompiled shared libraries for fast installation without CUDA compilation.

## Kernel Library

### CUDA Kernels (compiled via CMake)

These kernels are implemented in CUDA C++ and compiled during wheel build.

#### `activation` - GELU Residual Activation
Computes `GELU(h) * (g + 1)` fused gated activation used in MoE expert layers. The input tensor is split in half: `h` passes through GELU, `g` acts as a gate with +1 bias.

| Tokens | CUDA | PyTorch (eager) | Compile | vs PyTorch |
|--------|------|-----------------|---------|------------|
| 1 | 3.8 us | 64 us | 63 us | **17x** |
| 64 | 2.9 us | 49 us | 69 us | **17x** |
| 740 | 3.5 us | 49 us | 68 us | **14x** |
| 1024 | 3.9 us | 49 us | 68 us | **13x** |
| 2048 | 5.1 us | 49 us | 68 us | **10x** |

PyTorch eager launches separate kernels for slice, erf, multiply, and add, with intermediate tensors hitting global memory. Our kernel fuses everything into a single pass. torch.compile is slower than eager here, likely because the dynamic `x[:, :hidden]` slicing prevents effective fusion.

#### `fused_linear_residual` - Linear + Bias + Residual
Fused `out = x @ W.T + bias + residual` using cuBLASLt epilogues.

| Crops | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|-------|--------|------|-----------------|------------|
| 1 | 729 | 9.0 us | 24 us | **2.7x** |
| 2 | 1458 | 12 us | 24 us | **2.0x** |
| 4 | 2916 | 16 us | 29 us | **1.8x** |
| 8 | 5832 | 46 us | 50 us | **1.1x** |
| 13 | 9477 | 44 us | 77 us | **1.7x** |

cuBLASLt epilogues fuse bias addition and residual into the matmul, avoiding extra kernel launches and memory traffic.

#### `fused_mlp` - Fused MLP with cuBLASLt
Fused `out = residual + gelu(x @ W1.T + b1) @ W2.T + b2` using cuBLASLt epilogues.

| Crops | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|-------|--------|------|-----------------|------------|
| 1 | 729 | 43 us | 56 us | **1.3x** |
| 2 | 1458 | 72 us | 89 us | **1.2x** |
| 4 | 2916 | 97 us | 124 us | **1.3x** |
| 8 | 5832 | 214 us | 259 us | **1.2x** |
| 13 | 9477 | 283 us | 379 us | **1.3x** |

MLP is matmul-dominated so the speedup is modest. The gain comes from fusing GELU and residual add into cuBLASLt epilogues.

#### `kv_cache_write` - KV Cache Write with FP8 Quantization
Writes BF16 key/value tensors to FP8 paged KV cache with quantization.

| Tokens | Kestrel | vLLM | PyTorch (eager) | vs vLLM | vs PyTorch |
|--------|---------|------|-----------------|---------|------------|
| 1 | 3.7 us | 4.9 us | 67 us | **1.3x** | **18x** |
| 8 | 3.5 us | 4.8 us | 35 us | **1.4x** | **10x** |
| 64 | 3.7 us | 4.8 us | 35 us | **1.3x** | **9x** |
| 256 | 4.1 us | 4.8 us | 36 us | **1.2x** | **9x** |
| 1024 | 8.6 us | 9.7 us | 51 us | **1.1x** | **6x** |
| 4096 | 31 us | 46 us | 124 us | **1.5x** | **4x** |

Fused K/V processing and optimized vectorization provide 1.1-1.5x speedup over vLLM's implementation.

#### `layernorm_cuda` - Fast LayerNorm Forward
Optimized LayerNorm forward pass for common hidden dimensions.

**Vision Encoder (N=1152):**

| Crops | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|-------|--------|------|-----------------|------------|
| 1 | 729 | 3.9 us | 8.4 us | **2.2x** |
| 2 | 1458 | 4.2 us | 8.4 us | **2.0x** |
| 4 | 2916 | 5.5 us | 10 us | **1.8x** |
| 8 | 5832 | 8.3 us | 18 us | **2.1x** |
| 13 | 9477 | 18 us | 28 us | **1.6x** |

**Text Decoder (N=2048):**

| Context | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|---------|--------|------|-----------------|------------|
| decode | 1 | 4.2 us | 8.4 us | **2.0x** |
| prefill | 740 | 3.7 us | 8.4 us | **2.3x** |

Specialized kernels for N=1152 and N=2048 use 4 rows/block with warp-only reductions, avoiding shared memory overhead. Two epilogue strategies trade register pressure vs memory bandwidth.

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

