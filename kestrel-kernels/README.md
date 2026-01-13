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
Sums the weighted outputs from top-k MoE experts back into a single hidden state per token. Computes `out[t] = sum(expert_outputs[t, 0:k])` where each token selects k=8 experts.

| Context | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|---------|--------|------|-----------------|------------|
| decode | 1 | 3.0 us | 5.6 us | **1.9x** |
| batch 4 | 4 | 3.0 us | 5.4 us | **1.8x** |
| batch 16 | 16 | 2.9 us | 5.3 us | **1.8x** |
| prefill | 740 | 5.5 us | 10 us | **1.9x** |
| long | 1024 | 10 us | 15 us | **1.5x** |

Vectorized 16-byte loads (8 bf16 at once), fully unrolled k=8 reduction. FP32 accumulation provides better numerical stability than bf16 accumulation. Note: vLLM has a similar kernel, but only supports topk=2,3,4 and falls back to PyTorch for topk=8.

#### `rotary_embedding` - Rotary Position Embedding
Applies rotary position embedding to query and key tensors (n_heads=32, head_dim=64).

| Context | Tokens | Kestrel | vLLM | PyTorch (eager) | vs vLLM | vs PyTorch |
|---------|--------|---------|------|-----------------|---------|------------|
| decode | 1 | 3.3 us | 4.9 us | 118 us | **1.5x** | **36x** |
| batch 4 | 4 | 3.1 us | 4.5 us | 117 us | **1.5x** | **38x** |
| batch 16 | 16 | 3.1 us | 4.7 us | 117 us | **1.5x** | **38x** |
| prefill | 740 | 5.0 us | 8.0 us | 119 us | **1.6x** | **24x** |

Vectorized bfloat162 pair processing, shared memory caching of cos/sin values, FP32 math for numerical stability. Split-head kernel for decode increases SM utilization on small batch sizes.

#### `fp8_quant` - FP8 Quantization
Converts BF16 tensors to FP8 (e4m3fn) with per-row dynamic scale computation. Used for quantizing MoE activations before FP8 GEMM.

| Context | Rows | CUDA | PyTorch (eager) | vs PyTorch |
|---------|------|------|-----------------|------------|
| decode | 8 | 3.1 us | 53 us | **17x** |
| batch 4 | 32 | 3.1 us | 52 us | **17x** |
| batch 16 | 128 | 3.1 us | 52 us | **17x** |
| prefill | 5920 | 6.6 us | 67 us | **10x** |

Two kernel variants: warp-per-row for large batches (better SM utilization), block-per-row for small batches. Vectorized 16-byte loads/stores, fused absmax reduction.

#### `tau_tail` - TAU Attention Scaling
Applies per-head TAU scaling to Q and V in packed QKV. Computes `scale = tanh(tok_linear) + tau_pos_table[position]` then scales each head: `Q *= scale_q`, `V *= scale_v`.

| Context | Tokens | CUDA | PyTorch (eager) | vs PyTorch |
|---------|--------|------|-----------------|------------|
| decode | 1 | 4.6 us | 45 us | **10x** |
| batch 4 | 4 | 4.4 us | 46 us | **10x** |
| batch 16 | 16 | 9.0 us | 88 us | **10x** |
| prefill | 740 | 6.5 us | 63 us | **10x** |

---

### CuTe DSL Kernels (precompiled for wheel distribution)

These kernels are written in NVIDIA CuTe DSL (Python) and precompiled to `.so` files during wheel build. The kernel source templates are excluded from wheel distribution.

#### `topk` - Bitonic Top-K Selection
GPU top-k selection using bitonic sort network with optional fused softmax.

| Context | Tokens | Kestrel | Quack | PyTorch (eager) | vs Quack | vs PyTorch |
|---------|--------|---------|-------|-----------------|----------|------------|
| decode | 1 | 23 us | 29 us | 17 us | **1.3x** | 0.8x |
| batch 16 | 16 | 22 us | 27 us | 17 us | **1.2x** | 0.8x |
| prefill | 740 | 22 us | 28 us | 17 us | **1.2x** | 0.7x |

Note: Currently slower than PyTorch for N=64, k=8. PyTorch uses radix-based QuickSelect which is more efficient for small N. Algorithm should be revisited.

**Python API:**
```python
from kestrel_kernels.topk import topk_fwd

values, indices = topk_fwd(scores, k=8, softmax=True)
```

#### `cute_moe` - MoE Matrix Multiplications
Grouped GEMM kernels for Mixture-of-Experts layers, written in CuTe DSL for H100 (SM90). Supports BF16 and FP8 (W8A8) precision with both warp-level and WGMMA variants, automatically selected based on batch size.

**FP8 W8A8 Full MoE Layer** (up + activation + down + sum, E=64, k=8, with CUDA Graphs):

| Context | Tokens | Kestrel | vLLM (Triton) | vs vLLM |
|---------|--------|---------|---------------|---------|
| decode | 1 | 37 us | 56 us | **1.50x** |
| batch 4 | 4 | 87 us | 108 us | **1.23x** |
| batch 16 | 16 | 148 us | 171 us | **1.16x** |
| prefill | 740 | 250 us | 485 us | **1.94x** |

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
Prepares sorted token indices for block-sparse MoE operations. Given topk_ids, outputs sorted token IDs grouped by expert for block-sparse matmul.

| Context | Tokens | Kestrel | vLLM | vs vLLM |
|---------|--------|---------|------|---------|
| decode | 1 | 6.7 us | 9.8 us | **1.5x** |
| batch 4 | 4 | 6.5 us | 9.8 us | **1.5x** |
| batch 16 | 16 | 7.0 us | 10 us | **1.4x** |
| prefill | 740 | 12 us | 9.2 us | 0.8x |
| long | 1024 | 12 us | 9.5 us | 0.8x |

Uses optimized single-CTA shared-memory histogram for decode (numel < 1024). Prefill path needs optimization.

**Python API:**
```python
from kestrel_kernels.moe_align import moe_align_block_size

moe_align_block_size(
    topk_ids, num_experts, block_size,
    sorted_token_ids, expert_ids, num_tokens_post_pad,
    expert_map,  # optional for expert parallelism
)
```

#### `gelu_residual` - GELU Residual Activation (CuTe DSL)
CuTe DSL implementation of GELU residual activation for BF16. Computes `GELU(h) * (g + 1)` fused gated activation used in MoE expert layers. Uses vectorized memory access and streaming stores.

| Context | Rows | CuTe | CUDA | PyTorch | vs CUDA | vs PyTorch |
|---------|------|------|------|---------|---------|------------|
| decode | 8 | 2.3 us | 2.5 us | 7.5 us | **1.10x** | **3.3x** |
| batch 4 | 32 | 2.4 us | 3.0 us | 8.6 us | **1.24x** | **3.6x** |
| batch 16 | 128 | 2.6 us | 2.9 us | 8.9 us | **1.09x** | **3.4x** |
| prefill | 5920 | 9.9 us | 11.2 us | 55.9 us | **1.14x** | **5.6x** |

#### `fp8_quant_cute` - FP8 Quantization (CuTe DSL)
CuTe DSL implementation of FP8 row-wise quantization. Converts BF16 tensors to FP8 (e4m3fn) with per-row dynamic scaling.

**hidden=1024** (MoE down projection input):

| Context | Rows | CuTe | CUDA | vs CUDA |
|---------|------|------|------|---------|
| decode | 8 | 2.5 us | 2.7 us | **1.09x** |
| batch 4 | 32 | 2.8 us | 3.0 us | **1.07x** |
| batch 16 | 128 | 2.8 us | 3.0 us | **1.08x** |
| prefill | 5920 | 5.3 us | 6.6 us | **1.23x** |

**hidden=2048** (MoE up projection input):

| Context | Rows | CuTe | CUDA | vs CUDA |
|---------|------|------|------|---------|
| decode | 8 | 2.6 us | 2.7 us | **1.02x** |
| batch 4 | 32 | 2.9 us | 3.0 us | **1.04x** |
| batch 16 | 128 | 2.9 us | 3.0 us | **1.04x** |
| prefill | 5920 | 8.2 us | 10.7 us | **1.31x** |

#### `flash_attn` - Flash Attention (Prefill & Decode)
Flash Attention kernels written in CuTe DSL, with a dedicated decode path optimized for paged FP8 KV cache. 1.3-2.5x faster than FlashInfer on typical Moondream workloads.

- FP8 KV cache with per-tensor scaling
- Paged KV (page_size=1) for fine-grained memory management
- CUDA graph compatible
- Causal and prefix-LM masking, variable-length sequences, GQA/MQA

**FP8 KV Paged Decode** (with CUDA Graphs):

| Batch | KV Len | Kestrel | FlashInfer | vs FlashInfer |
|-------|--------|---------|------------|---------------|
| 1 | 740 | 9.6 us | 12.9 us | **1.34x** |
| 1 | 1024 | 8.7 us | 13.1 us | **1.50x** |
| 4 | 740 | 17.1 us | 23.9 us | **1.40x** |
| 8 | 512 | 10.0 us | 25.2 us | **2.51x** |
| 16 | 256 | 9.6 us | 17.6 us | **1.83x** |
| 32 | 128 | 11.8 us | 26.5 us | **2.24x** |

**FP8 KV Paged Prefill**:

| Seq Len | Kestrel | FlashInfer | vs FlashInfer |
|---------|---------|------------|---------------|
| 740 | 19.9 us | 47.6 us | **2.40x** |
| 1024 | 27.3 us | 58.9 us | **2.16x** |

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

