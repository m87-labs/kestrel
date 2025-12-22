#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <cstdint>

namespace kestrel {
namespace {

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

// This file implements a fast bf16-only LayerNorm forward kernel family.
//
// High-level perf tricks used here (H100 / bf16 inference focused):
//   1) **Vectorized global memory**: load/store 8 bf16 values at a time via `int4`
//      (16B). This requires 16B alignment and `N % 8 == 0`.
//   2) **Warp reductions** for mean/variance for common "vision/text width" cases
//      to avoid shared memory + block-wide reduction overhead.
//   3) **Specialization for N=1152 and N=2048**: map 1 warp to 1 row and process
//      4 rows per block (128 threads). This increases parallelism vs 1-row-per-block
//      and tends to balance overheads well for our target shapes.
//   4) **Two epilogue strategies** for the rows4 kernel:
//        - `...rows4_kernel`: caches the loaded `x` vectors in registers and reuses
//          them for the output pass (fewer global reads, higher register pressure).
//        - `...rows4_reloadx_kernel`: reloads `x` for the output pass (more global
//          reads, much lower register pressure -> higher occupancy / more latency hiding).
//      Which one wins depends on shape and device; we keep both and choose in Python.

#if defined(__CUDA_ARCH__)
  #define KESTREL_LDG(arg) __ldg(arg)
#else
  #define KESTREL_LDG(arg) *(arg)
#endif

inline bool is_16byte_aligned_host(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

struct SumSq {
  float sum;
  float sumsq;
};

struct SumSqOp {
  __device__ __forceinline__ SumSq operator()(SumSq a, SumSq b) const {
    return SumSq{a.sum + b.sum, a.sumsq + b.sumsq};
  }
};

__device__ __forceinline__ float warp_allreduce_sum(float x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

template <int N>
__global__ void layernorm_bias_bf16_vec8_rows4_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int M,
    float eps) {
  constexpr int kVecElems = 16 / sizeof(__nv_bfloat16);  // 8 bf16 values per int4
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = 4;
  constexpr int kNumVec = N / kVecElems;
  constexpr int kIters = (kNumVec + kWarpSize - 1) / kWarpSize;
  static_assert(N % kVecElems == 0, "N must be a multiple of 8");
  static_assert(kRowsPerBlock * kWarpSize == 128, "kernel assumes 128 threads/block");

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= M) return;

  const int4* x_vec = reinterpret_cast<const int4*>(x + static_cast<int64_t>(row) * N);
  int4* out_vec = reinterpret_cast<int4*>(out + static_cast<int64_t>(row) * N);
  const int4* w_vec = reinterpret_cast<const int4*>(weight);
  const int4* b_vec = reinterpret_cast<const int4*>(bias);

  int4 x_frag[kIters];

  float sum = 0.0f;
  float sumsq = 0.0f;

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int vec_idx = lane + it * kWarpSize;
    int4 v = {0, 0, 0, 0};
    if (vec_idx < kNumVec) {
      v = KESTREL_LDG(&x_vec[vec_idx]);
      const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        const float xf = __bfloat162float(vp[j]);
        sum += xf;
        sumsq += xf * xf;
      }
    }
    x_frag[it] = v;
  }

  sum = warp_allreduce_sum(sum);
  sumsq = warp_allreduce_sum(sumsq);

  float mean = sum / static_cast<float>(N);
  float ex2 = sumsq / static_cast<float>(N);
  float var = ex2 - mean * mean;
  var = (var > 0.0f) ? var : 0.0f;
  float inv_std = rsqrtf(var + eps);

  mean = __shfl_sync(0xffffffff, mean, 0);
  inv_std = __shfl_sync(0xffffffff, inv_std, 0);

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int vec_idx = lane + it * kWarpSize;
    if (vec_idx < kNumVec) {
      const int4 xv = x_frag[it];
      const int4 wv = KESTREL_LDG(&w_vec[vec_idx]);
      const int4 bv = KESTREL_LDG(&b_vec[vec_idx]);

      int4 ov;
      const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
      const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
      const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
      __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);

#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        const float xf = __bfloat162float(xp[j]);
        const float wf = __bfloat162float(wp[j]);
        const float bf = __bfloat162float(bp[j]);
        const float yf = (xf - mean) * inv_std * wf + bf;
        op[j] = __float2bfloat16_rn(yf);
      }
      out_vec[vec_idx] = ov;
    }
  }
}

template <int N>
__global__ void layernorm_bias_bf16_vec8_rows4_reloadx_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int M,
    float eps) {
  constexpr int kVecElems = 16 / sizeof(__nv_bfloat16);  // 8 bf16 values per int4
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = 4;
  constexpr int kNumVec = N / kVecElems;
  constexpr int kIters = (kNumVec + kWarpSize - 1) / kWarpSize;
  static_assert(N % kVecElems == 0, "N must be a multiple of 8");
  static_assert(kRowsPerBlock * kWarpSize == 128, "kernel assumes 128 threads/block");

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= M) return;

  const int4* x_vec =
      reinterpret_cast<const int4*>(x + static_cast<int64_t>(row) * N);
  int4* out_vec = reinterpret_cast<int4*>(out + static_cast<int64_t>(row) * N);
  const int4* w_vec = reinterpret_cast<const int4*>(weight);
  const int4* b_vec = reinterpret_cast<const int4*>(bias);

  float sum = 0.0f;
  float sumsq = 0.0f;

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int vec_idx = lane + it * kWarpSize;
    if (vec_idx < kNumVec) {
      const int4 v = KESTREL_LDG(&x_vec[vec_idx]);
      const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        const float xf = __bfloat162float(vp[j]);
        sum += xf;
        sumsq += xf * xf;
      }
    }
  }

  sum = warp_allreduce_sum(sum);
  sumsq = warp_allreduce_sum(sumsq);

  float mean = sum / static_cast<float>(N);
  float ex2 = sumsq / static_cast<float>(N);
  float var = ex2 - mean * mean;
  var = (var > 0.0f) ? var : 0.0f;
  float inv_std = rsqrtf(var + eps);

  mean = __shfl_sync(0xffffffff, mean, 0);
  inv_std = __shfl_sync(0xffffffff, inv_std, 0);

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int vec_idx = lane + it * kWarpSize;
    if (vec_idx < kNumVec) {
      const int4 xv = KESTREL_LDG(&x_vec[vec_idx]);
      const int4 wv = KESTREL_LDG(&w_vec[vec_idx]);
      const int4 bv = KESTREL_LDG(&b_vec[vec_idx]);

      int4 ov;
      const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
      const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
      const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
      __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);

#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        const float xf = __bfloat162float(xp[j]);
        const float wf = __bfloat162float(wp[j]);
        const float bf = __bfloat162float(bp[j]);
        const float yf = (xf - mean) * inv_std * wf + bf;
        op[j] = __float2bfloat16_rn(yf);
      }
      out_vec[vec_idx] = ov;
    }
  }
}

template <int BLOCK_THREADS>
__global__ void layernorm_bias_bf16_vec8_warp_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int N,
    float eps) {
  constexpr int kVecElems = 16 / sizeof(__nv_bfloat16);
  static_assert(BLOCK_THREADS == 32, "warp kernel must use 1 warp");

  const int row = static_cast<int>(blockIdx.x);
  const int num_vec = N / kVecElems;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  const __nv_bfloat16* x_row = x + static_cast<int64_t>(row) * N;
  __nv_bfloat16* out_row = out + static_cast<int64_t>(row) * N;

  const int4* x_vec = reinterpret_cast<const int4*>(x_row);
  int4* out_vec = reinterpret_cast<int4*>(out_row);
  const int4* w_vec = reinterpret_cast<const int4*>(weight);
  const int4* b_vec = reinterpret_cast<const int4*>(bias);

  float sum = 0.0f;
  float sumsq = 0.0f;

  for (int i = lane; i < num_vec; i += 32) {
    const int4 v = KESTREL_LDG(&x_vec[i]);
    const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      const float xf = __bfloat162float(vp[j]);
      sum += xf;
      sumsq += xf * xf;
    }
  }

  sum = warp_allreduce_sum(sum);
  sumsq = warp_allreduce_sum(sumsq);

  float mean = sum / static_cast<float>(N);
  float ex2 = sumsq / static_cast<float>(N);
  float var = ex2 - mean * mean;
  var = (var > 0.0f) ? var : 0.0f;
  float inv_std = rsqrtf(var + eps);

  mean = __shfl_sync(0xffffffff, mean, 0);
  inv_std = __shfl_sync(0xffffffff, inv_std, 0);

  for (int i = lane; i < num_vec; i += 32) {
    const int4 xv = KESTREL_LDG(&x_vec[i]);
    const int4 wv = KESTREL_LDG(&w_vec[i]);
    const int4 bv = KESTREL_LDG(&b_vec[i]);

    int4 ov;
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);

#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      const float xf = __bfloat162float(xp[j]);
      const float wf = __bfloat162float(wp[j]);
      const float bf = __bfloat162float(bp[j]);
      const float yf = (xf - mean) * inv_std * wf + bf;
      op[j] = __float2bfloat16_rn(yf);
    }
    out_vec[i] = ov;
  }
}

template <int BLOCK_THREADS>
__global__ void layernorm_bias_bf16_vec8_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int N,
    float eps) {
  constexpr int kVecElems = 16 / sizeof(__nv_bfloat16);  // 8 bf16 values per int4

  const int row = static_cast<int>(blockIdx.x);
  const int num_vec = N / kVecElems;

  const __nv_bfloat16* x_row = x + static_cast<int64_t>(row) * N;
  __nv_bfloat16* out_row = out + static_cast<int64_t>(row) * N;

  const int4* x_vec = reinterpret_cast<const int4*>(x_row);
  int4* out_vec = reinterpret_cast<int4*>(out_row);
  const int4* w_vec = reinterpret_cast<const int4*>(weight);
  const int4* b_vec = reinterpret_cast<const int4*>(bias);

  extern __shared__ int4 s_x_vec[];

  SumSq local{0.0f, 0.0f};
  for (int i = threadIdx.x; i < num_vec; i += BLOCK_THREADS) {
    const int4 v = KESTREL_LDG(&x_vec[i]);
    s_x_vec[i] = v;
    const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      const float xf = __bfloat162float(vp[j]);
      local.sum += xf;
      local.sumsq += xf * xf;
    }
  }

  using BlockReduce = cub::BlockReduce<SumSq, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float s_mean;
  __shared__ float s_inv_std;

  const SumSq total = BlockReduce(temp_storage).Reduce(local, SumSqOp{});

  if (threadIdx.x == 0) {
    const float mean = total.sum / static_cast<float>(N);
    const float ex2 = total.sumsq / static_cast<float>(N);
    float var = ex2 - mean * mean;
    var = (var > 0.0f) ? var : 0.0f;
    s_mean = mean;
    s_inv_std = rsqrtf(var + eps);
  }
  __syncthreads();

  const float mean = s_mean;
  const float inv_std = s_inv_std;

  for (int i = threadIdx.x; i < num_vec; i += BLOCK_THREADS) {
    const int4 xv = s_x_vec[i];
    const int4 wv = KESTREL_LDG(&w_vec[i]);
    const int4 bv = KESTREL_LDG(&b_vec[i]);

    int4 ov;
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);

#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      const float xf = __bfloat162float(xp[j]);
      const float wf = __bfloat162float(wp[j]);
      const float bf = __bfloat162float(bp[j]);
      const float yf = (xf - mean) * inv_std * wf + bf;
      op[j] = __float2bfloat16_rn(yf);
    }
    out_vec[i] = ov;
  }
}

void check_inputs(const torch::Tensor& out,
                  const torch::Tensor& x,
                  const torch::Tensor& weight,
                  const torch::Tensor& bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16,
              "weight must be bf16");
  TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16,
              "bias must be bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M,N)");
  TORCH_CHECK(out.dim() == 2, "out must be 2D (M,N)");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D (N,)");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D (N,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");

  const int64_t N = x.size(1);
  TORCH_CHECK(weight.size(0) == N, "weight must have shape (N,)");
  TORCH_CHECK(bias.size(0) == N, "bias must have shape (N,)");

  constexpr int kVecElems = 16 / sizeof(__nv_bfloat16);  // 8
  TORCH_CHECK((N % kVecElems) == 0, "N must be a multiple of 8 for vec8");

  const void* x_ptr = x.data_ptr<at::BFloat16>();
  const void* out_ptr = out.data_ptr<at::BFloat16>();
  const void* w_ptr = weight.data_ptr<at::BFloat16>();
  const void* b_ptr = bias.data_ptr<at::BFloat16>();
  TORCH_CHECK(is_16byte_aligned_host(x_ptr), "x must be 16-byte aligned");
  TORCH_CHECK(is_16byte_aligned_host(out_ptr), "out must be 16-byte aligned");
  TORCH_CHECK(is_16byte_aligned_host(w_ptr), "weight must be 16-byte aligned");
  TORCH_CHECK(is_16byte_aligned_host(b_ptr), "bias must be 16-byte aligned");
  TORCH_CHECK(((N * sizeof(__nv_bfloat16)) % 16) == 0,
              "row stride must be 16-byte aligned");
}

}  // namespace

void layernorm_bias_cuda_impl(torch::Tensor& out,
                              torch::Tensor& x,
                              torch::Tensor& weight,
                              torch::Tensor& bias,
                              double eps,
                              bool reload_x_rows4) {
  check_inputs(out, x, weight, bias);
  if (x.numel() == 0) return;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  TORCH_CHECK(N <= 8192, "N too large for this kernel: ", N);

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_ptr =
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  const auto* b_ptr =
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>());

  const int threads = (N <= 1536) ? 128 : ((N <= 3072) ? 256 : 512);
  const size_t smem_bytes = static_cast<size_t>(N) * sizeof(__nv_bfloat16);

  dim3 grid(static_cast<unsigned int>(M), 1u, 1u);
  if (M <= 32) {
    layernorm_bias_bf16_vec8_warp_kernel<32>
        <<<grid, 32, 0, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                  static_cast<int>(N),
                                  static_cast<float>(eps));
  } else if (N == 1152) {
    const unsigned int grid_x =
        static_cast<unsigned int>((M + 4 - 1) / 4);
    if (reload_x_rows4) {
      layernorm_bias_bf16_vec8_rows4_reloadx_kernel<1152>
          <<<dim3(grid_x, 1u, 1u), 128, 0, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                                     static_cast<int>(M),
                                                     static_cast<float>(eps));
    } else {
      layernorm_bias_bf16_vec8_rows4_kernel<1152>
          <<<dim3(grid_x, 1u, 1u), 128, 0, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                                     static_cast<int>(M),
                                                     static_cast<float>(eps));
    }
  } else if (N == 2048) {
    const unsigned int grid_x =
        static_cast<unsigned int>((M + 4 - 1) / 4);
    if (reload_x_rows4) {
      layernorm_bias_bf16_vec8_rows4_reloadx_kernel<2048>
          <<<dim3(grid_x, 1u, 1u), 128, 0, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                                     static_cast<int>(M),
                                                     static_cast<float>(eps));
    } else {
      layernorm_bias_bf16_vec8_rows4_kernel<2048>
          <<<dim3(grid_x, 1u, 1u), 128, 0, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                                     static_cast<int>(M),
                                                     static_cast<float>(eps));
    }
  } else if (threads == 128) {
    layernorm_bias_bf16_vec8_kernel<128>
        <<<grid, 128, smem_bytes, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                            static_cast<int>(N),
                                            static_cast<float>(eps));
  } else if (threads == 256) {
    layernorm_bias_bf16_vec8_kernel<256>
        <<<grid, 256, smem_bytes, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                            static_cast<int>(N),
                                            static_cast<float>(eps));
  } else {
    layernorm_bias_bf16_vec8_kernel<512>
        <<<grid, 512, smem_bytes, stream>>>(out_ptr, x_ptr, w_ptr, b_ptr,
                                            static_cast<int>(N),
                                            static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layernorm_bias_cuda(torch::Tensor& out,
                         torch::Tensor& x,
                         torch::Tensor& weight,
                         torch::Tensor& bias,
                         double eps) {
  layernorm_bias_cuda_impl(out, x, weight, bias, eps, /*reload_x_rows4=*/false);
}

void layernorm_bias_reload_cuda(torch::Tensor& out,
                                torch::Tensor& x,
                                torch::Tensor& weight,
                                torch::Tensor& bias,
                                double eps) {
  layernorm_bias_cuda_impl(out, x, weight, bias, eps, /*reload_x_rows4=*/true);
}

}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm_bias_cuda",
        &kestrel::layernorm_bias_cuda,
        "LayerNorm forward (bf16) with weight+bias (CUDA)");
  m.def("layernorm_bias_reload_cuda",
        &kestrel::layernorm_bias_reload_cuda,
        "LayerNorm forward (bf16) with weight+bias (CUDA, reload-x rows4 variant)");
}
