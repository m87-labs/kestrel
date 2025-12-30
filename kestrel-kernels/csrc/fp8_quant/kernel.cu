#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>

#include <cmath>

namespace kestrel {

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

constexpr float kFp8E4M3Max = 448.0f;
constexpr float kMinScale = 1e-6f;
constexpr int kWarpSize = 32;
constexpr int kDefaultThreads = 256;

__device__ __forceinline__ float warp_reduce_max(float v) {
  // Butterfly-reduce within a warp.
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

template <int kThreads>
__device__ __forceinline__ float block_reduce_max(float v) {
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp;
  return BlockReduce(temp).Reduce(v, cub::Max());
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <int kVecSize>
__device__ __forceinline__ void store_fp8_vector(__nv_fp8_e4m3* dst, const uint8_t* bytes);

template <>
__device__ __forceinline__ void store_fp8_vector<16>(__nv_fp8_e4m3* dst, const uint8_t* bytes) {
  // Store 16 bytes at once.
  union alignas(16) Pack {
    uint8_t b[16];
    uint4 u4;
  } pack;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    pack.b[i] = bytes[i];
  }
  *reinterpret_cast<uint4*>(dst) = pack.u4;
}

template <>
__device__ __forceinline__ void store_fp8_vector<8>(__nv_fp8_e4m3* dst, const uint8_t* bytes) {
  union alignas(8) Pack {
    uint8_t b[8];
    uint2 u2;
  } pack;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pack.b[i] = bytes[i];
  }
  *reinterpret_cast<uint2*>(dst) = pack.u2;
}

template <>
__device__ __forceinline__ void store_fp8_vector<4>(__nv_fp8_e4m3* dst, const uint8_t* bytes) {
  union alignas(4) Pack {
    uint8_t b[4];
    uint32_t u32;
  } pack;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pack.b[i] = bytes[i];
  }
  *reinterpret_cast<uint32_t*>(dst) = pack.u32;
}

template <int kVecSize>
__global__ void fp8_e4m3fn_rowwise_quant_small_batch_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    __nv_fp8_e4m3* __restrict__ output_fp8,   // [M, K] (stored into uint8)
    float* __restrict__ output_scale,         // [M]
    int64_t hidden_dim,
    int64_t num_rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_rows) {
    return;
  }

  const int tid = threadIdx.x;
  const int64_t row_offset = row * hidden_dim;
  const int64_t num_vecs = hidden_dim / kVecSize;

  // Pass-1: per-row absmax (vectorized).
  float thread_max = 0.0f;
  for (int64_t vec = tid; vec < num_vecs; vec += blockDim.x) {
    // Load kVecSize bf16 values.
    const int64_t col0 = vec * kVecSize;
    // Use 16-byte aligned loads when possible. kVecSize is in {4, 8, 16},
    // so the largest load is 32 bytes (two int4).
    if constexpr (kVecSize == 16) {
      const int4* in4 = reinterpret_cast<const int4*>(input + row_offset + col0);
      int4 x0 = in4[0];
      int4 x1 = in4[1];
      const __nv_bfloat16* v0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
      const __nv_bfloat16* v1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        thread_max = fmaxf(thread_max, fabsf(bf16_to_float(v0[i])));
        thread_max = fmaxf(thread_max, fabsf(bf16_to_float(v1[i])));
      }
    } else if constexpr (kVecSize == 8) {
      const int4 x = *reinterpret_cast<const int4*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        thread_max = fmaxf(thread_max, fabsf(bf16_to_float(v[i])));
      }
    } else if constexpr (kVecSize == 4) {
      const int2 x = *reinterpret_cast<const int2*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        thread_max = fmaxf(thread_max, fabsf(bf16_to_float(v[i])));
      }
    }
  }

  float row_max = block_reduce_max<kDefaultThreads>(thread_max);

  __shared__ float row_scale;
  if (tid == 0) {
    float scale = row_max / kFp8E4M3Max;
    if (scale < kMinScale) {
      scale = kMinScale;
    }
    row_scale = scale;
    output_scale[row] = scale;
  }
  __syncthreads();

  const float inv_scale = 1.0f / row_scale;

  // Pass-2: quantize and store (vectorized).
  for (int64_t vec = tid; vec < num_vecs; vec += blockDim.x) {
    const int64_t col0 = vec * kVecSize;
    uint8_t out_bytes[kVecSize];

    if constexpr (kVecSize == 16) {
      const int4* in4 = reinterpret_cast<const int4*>(input + row_offset + col0);
      int4 x0 = in4[0];
      int4 x1 = in4[1];
      const __nv_bfloat16* v0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
      const __nv_bfloat16* v1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a0 = bf16_to_float(v0[i]) * inv_scale;
        float a1 = bf16_to_float(v1[i]) * inv_scale;
        a0 = fminf(fmaxf(a0, -kFp8E4M3Max), kFp8E4M3Max);
        a1 = fminf(fmaxf(a1, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q0 = static_cast<__nv_fp8_e4m3>(a0);
        __nv_fp8_e4m3 q1 = static_cast<__nv_fp8_e4m3>(a1);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q0);
        out_bytes[i + 8] = *reinterpret_cast<const uint8_t*>(&q1);
      }
      store_fp8_vector<16>(output_fp8 + row_offset + col0, out_bytes);
    } else if constexpr (kVecSize == 8) {
      const int4 x = *reinterpret_cast<const int4*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a = bf16_to_float(v[i]) * inv_scale;
        a = fminf(fmaxf(a, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(a);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q);
      }
      store_fp8_vector<8>(output_fp8 + row_offset + col0, out_bytes);
    } else if constexpr (kVecSize == 4) {
      const int2 x = *reinterpret_cast<const int2*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float a = bf16_to_float(v[i]) * inv_scale;
        a = fminf(fmaxf(a, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(a);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q);
      }
      store_fp8_vector<4>(output_fp8 + row_offset + col0, out_bytes);
    }
  }
}

template <int kTokensPerCTA, int kVecSize>
__global__ void fp8_e4m3fn_rowwise_quant_warp_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    __nv_fp8_e4m3* __restrict__ output_fp8,   // [M, K]
    float* __restrict__ output_scale,         // [M]
    int64_t hidden_dim,
    int64_t num_rows) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0..kTokensPerCTA-1
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0..31
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kTokensPerCTA + warp_id;
  if (row >= num_rows) {
    return;
  }

  const int64_t row_offset = row * hidden_dim;
  const int64_t num_vecs = hidden_dim / kVecSize;

  // Pass-1: warp absmax.
  float lane_max = 0.0f;
  for (int64_t vec = lane_id; vec < num_vecs; vec += kWarpSize) {
    const int64_t col0 = vec * kVecSize;
    if constexpr (kVecSize == 16) {
      const int4* in4 = reinterpret_cast<const int4*>(input + row_offset + col0);
      int4 x0 = in4[0];
      int4 x1 = in4[1];
      const __nv_bfloat16* v0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
      const __nv_bfloat16* v1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        lane_max = fmaxf(lane_max, fabsf(bf16_to_float(v0[i])));
        lane_max = fmaxf(lane_max, fabsf(bf16_to_float(v1[i])));
      }
    } else if constexpr (kVecSize == 8) {
      const int4 x = *reinterpret_cast<const int4*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        lane_max = fmaxf(lane_max, fabsf(bf16_to_float(v[i])));
      }
    } else if constexpr (kVecSize == 4) {
      const int2 x = *reinterpret_cast<const int2*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        lane_max = fmaxf(lane_max, fabsf(bf16_to_float(v[i])));
      }
    }
  }

  const float row_max = warp_reduce_max(lane_max);
  const float scale = fmaxf(row_max / kFp8E4M3Max, kMinScale);
  const float inv_scale = 1.0f / scale;

  if (lane_id == 0) {
    output_scale[row] = scale;
  }

  // Pass-2: quantize.
  for (int64_t vec = lane_id; vec < num_vecs; vec += kWarpSize) {
    const int64_t col0 = vec * kVecSize;
    uint8_t out_bytes[kVecSize];
    if constexpr (kVecSize == 16) {
      const int4* in4 = reinterpret_cast<const int4*>(input + row_offset + col0);
      int4 x0 = in4[0];
      int4 x1 = in4[1];
      const __nv_bfloat16* v0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
      const __nv_bfloat16* v1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a0 = bf16_to_float(v0[i]) * inv_scale;
        float a1 = bf16_to_float(v1[i]) * inv_scale;
        a0 = fminf(fmaxf(a0, -kFp8E4M3Max), kFp8E4M3Max);
        a1 = fminf(fmaxf(a1, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q0 = static_cast<__nv_fp8_e4m3>(a0);
        __nv_fp8_e4m3 q1 = static_cast<__nv_fp8_e4m3>(a1);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q0);
        out_bytes[i + 8] = *reinterpret_cast<const uint8_t*>(&q1);
      }
      store_fp8_vector<16>(output_fp8 + row_offset + col0, out_bytes);
    } else if constexpr (kVecSize == 8) {
      const int4 x = *reinterpret_cast<const int4*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a = bf16_to_float(v[i]) * inv_scale;
        a = fminf(fmaxf(a, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(a);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q);
      }
      store_fp8_vector<8>(output_fp8 + row_offset + col0, out_bytes);
    } else if constexpr (kVecSize == 4) {
      const int2 x = *reinterpret_cast<const int2*>(input + row_offset + col0);
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&x);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float a = bf16_to_float(v[i]) * inv_scale;
        a = fminf(fmaxf(a, -kFp8E4M3Max), kFp8E4M3Max);
        __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(a);
        out_bytes[i] = *reinterpret_cast<const uint8_t*>(&q);
      }
      store_fp8_vector<4>(output_fp8 + row_offset + col0, out_bytes);
    }
  }
}

static void check_inputs(const torch::Tensor& input, const torch::Tensor& output_bits,
                         const torch::Tensor& output_scale) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(output_bits.is_cuda(), "output_bits must be CUDA");
  TORCH_CHECK(output_scale.is_cuda(), "output_scale must be CUDA");

  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
  TORCH_CHECK(output_bits.scalar_type() == at::ScalarType::Byte, "output_bits must be uint8");
  TORCH_CHECK(output_scale.scalar_type() == at::ScalarType::Float, "output_scale must be float32");

  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_bits.is_contiguous(), "output_bits must be contiguous");
  TORCH_CHECK(output_scale.is_contiguous(), "output_scale must be contiguous");

  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(output_bits.dim() == 2, "output_bits must be 2D [M, K]");
  TORCH_CHECK(output_scale.dim() == 1, "output_scale must be 1D [M]");

  TORCH_CHECK(output_bits.sizes() == input.sizes(), "output_bits must match input shape");
  TORCH_CHECK(output_scale.size(0) == input.size(0), "output_scale must have length M");

  TORCH_CHECK(input.stride(1) == 1, "input must be contiguous in the last dim");
  TORCH_CHECK(output_bits.stride(1) == 1, "output_bits must be contiguous in the last dim");
  TORCH_CHECK(
      (input.size(1) % 4) == 0,
      "hidden dimension must be divisible by 4 for vectorized FP8 quantization");
}

void fp8_e4m3fn_rowwise_quant_cuda(torch::Tensor& output_bits, torch::Tensor& output_scale,
                                  const torch::Tensor& input) {
  check_inputs(input, output_bits, output_scale);
  if (input.numel() == 0) {
    return;
  }

  const int64_t num_rows = input.size(0);
  const int64_t hidden_dim = input.size(1);

  constexpr int threads = kDefaultThreads;

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto* in_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output_bits.data_ptr<uint8_t>());
  auto* scale_ptr = output_scale.data_ptr<float>();

  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  constexpr int tokens_per_cta = 8;
  const bool use_warp_kernel = (num_rows >= static_cast<int64_t>(sm_count) * 2 * tokens_per_cta);

  if ((hidden_dim % 16) == 0) {
    if (use_warp_kernel) {
      dim3 grid((num_rows + tokens_per_cta - 1) / tokens_per_cta);
      dim3 block(tokens_per_cta * kWarpSize);
      fp8_e4m3fn_rowwise_quant_warp_kernel<tokens_per_cta, 16><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    } else {
      dim3 grid(num_rows);
      dim3 block(threads);
      fp8_e4m3fn_rowwise_quant_small_batch_kernel<16><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    }
  } else if ((hidden_dim % 8) == 0) {
    if (use_warp_kernel) {
      dim3 grid((num_rows + tokens_per_cta - 1) / tokens_per_cta);
      dim3 block(tokens_per_cta * kWarpSize);
      fp8_e4m3fn_rowwise_quant_warp_kernel<tokens_per_cta, 8><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    } else {
      dim3 grid(num_rows);
      dim3 block(threads);
      fp8_e4m3fn_rowwise_quant_small_batch_kernel<8><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    }
  } else {
    if (use_warp_kernel) {
      dim3 grid((num_rows + tokens_per_cta - 1) / tokens_per_cta);
      dim3 block(tokens_per_cta * kWarpSize);
      fp8_e4m3fn_rowwise_quant_warp_kernel<tokens_per_cta, 4><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    } else {
      dim3 grid(num_rows);
      dim3 block(threads);
      fp8_e4m3fn_rowwise_quant_small_batch_kernel<4><<<grid, block, 0, stream>>>(
          in_ptr, out_ptr, scale_ptr, hidden_dim, num_rows);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_e4m3fn_rowwise_quant_cuda", &kestrel::fp8_e4m3fn_rowwise_quant_cuda,
        "Row-wise FP8(E4M3FN) quantization (CUDA)");
}
