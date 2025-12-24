#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace kestrel {
namespace kv_cache {

static_assert(sizeof(at::Half) == sizeof(__half), "Half size mismatch");
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

template <typename T, int VEC_SIZE>
struct __align__(VEC_SIZE * sizeof(T)) vec_n_t {
  T val[VEC_SIZE];
};

template <typename OutT, typename InT>
struct CopyNoScaleOp {
  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    dst = static_cast<OutT>(src);
  }
};

__device__ __forceinline__ uint8_t float_to_fp8_e4m3fn_bits(const float x) {
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

template <typename InT>
__device__ __forceinline__ float to_float(const InT x) {
  return static_cast<float>(x);
}
template <>
__device__ __forceinline__ float to_float<__half>(const __half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename OutT, typename InT>
struct CopyFp8E4M3Op {
  float scale;
  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    // Store fp8(x / scale); dequantize later with fp8 * scale.
    const float x = to_float(src);
    dst = static_cast<OutT>(float_to_fp8_e4m3fn_bits(x / scale));
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename ScalarOp>
__device__ inline void vectorize_with_alignment(const InT* in, OutT* out,
                                                int len, int tid, int stride,
                                                ScalarOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  const bool can_vec =
      ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    const int num_vec = len / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    const auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);
    for (int i = tid; i < num_vec; i += stride) {
      const vin_t src = v_in[i];
      vout_t dst;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        scalar_op(dst.val[j], src.val[j]);
      }
      v_out[i] = dst;
    }
    return;
  }

  const int misalignment_offset = static_cast<int>(addr & (WIDTH - 1));
  const int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = (alignment_bytes & (WIDTH - 1)) / sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  const int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  const auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);
  for (int i = tid; i < num_vec; i += stride) {
    const vin_t src = v_in[i];
    vout_t dst;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      scalar_op(dst.val[j], src.val[j]);
    }
    v_out[i] = dst;
  }

  const int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <typename InT, typename OutT, typename ScalarOp>
__device__ inline void vectorize_best_effort(const InT* in, OutT* out, int len,
                                             int tid, int stride,
                                             ScalarOp&& scalar_op) {
  // Heuristic: pick a vector width that keeps at least one vector element per
  // thread (i.e., num_vec >= stride) when possible. This improves occupancy and
  // parallelism for small head sizes (e.g., head_size=64 with stride=32).
  if constexpr (sizeof(InT) == 2) {
    const int ratio = len / stride;
    if (ratio >= 8) {
      vectorize_with_alignment<8>(in, out, len, tid, stride, scalar_op);
    } else if (ratio >= 4) {
      vectorize_with_alignment<4>(in, out, len, tid, stride, scalar_op);
    } else if (ratio >= 2) {
      vectorize_with_alignment<2>(in, out, len, tid, stride, scalar_op);
    } else {
      vectorize_with_alignment<1>(in, out, len, tid, stride, scalar_op);
    }
  } else if constexpr (sizeof(InT) == 4) {
    const int ratio = len / stride;
    if (ratio >= 4) {
      vectorize_with_alignment<4>(in, out, len, tid, stride, scalar_op);
    } else if (ratio >= 2) {
      vectorize_with_alignment<2>(in, out, len, tid, stride, scalar_op);
    } else {
      vectorize_with_alignment<1>(in, out, len, tid, stride, scalar_op);
    }
  } else {
    vectorize_with_alignment<1>(in, out, len, tid, stride, scalar_op);
  }
}

template <typename scalar_t, typename cache_t, bool kFp8>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND (see below)
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const int heads_per_block, const int head_chunk_size,
    const int threads_per_head, const float* k_scale_ptr,
    const float* v_scale_ptr) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int head_block = static_cast<int>(blockIdx.y);
  const int head_start = head_block * heads_per_block;
  const int head_end = min(head_start + heads_per_block, num_heads);
  const int local_heads = head_end - head_start;
  if (local_heads <= 0) {
    return;
  }
  const int head_chunk_idx = static_cast<int>(blockIdx.z);
  const int head_chunk_start = head_chunk_idx * head_chunk_size;
  if (head_chunk_start >= head_size) {
    return;
  }
  const int head_chunk_len = min(head_chunk_size, head_size - head_chunk_start);

  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  const bool is_contiguous_heads = (head_stride == head_size);

  if constexpr (kFp8) {
    const float k_scale = *k_scale_ptr;
    const float v_scale = *v_scale_ptr;
    CopyFp8E4M3Op<cache_t, scalar_t> k_op{k_scale};
    CopyFp8E4M3Op<cache_t, scalar_t> v_op{v_scale};
    if (is_contiguous_heads) {
      // NHD: [num_blocks, block_size, num_heads, head_size]
      const int n_elems = local_heads * head_size;
      const int64_t head_offset = static_cast<int64_t>(head_start) * head_size;
      vectorize_best_effort(key_src + head_offset, key_dst + head_offset,
                            n_elems, threadIdx.x, blockDim.x, k_op);
      vectorize_best_effort(value_src + head_offset, value_dst + head_offset,
                            n_elems, threadIdx.x, blockDim.x, v_op);
    } else {
      // HND backing storage; heads are strided, but each head segment is
      // contiguous.
      const bool use_half_warp = (threads_per_head == 16);
      const int lane = use_half_warp ? (threadIdx.x & 15) : (threadIdx.x & 31);
      const int warp_id = use_half_warp ? (threadIdx.x >> 4) : (threadIdx.x >> 5);
      const int warps_per_block =
          use_half_warp ? (blockDim.x >> 4) : (blockDim.x >> 5);
      for (int head = head_start + warp_id; head < head_end;
           head += warps_per_block) {
        const scalar_t* __restrict__ k_src_h =
            key_src + head * head_size + head_chunk_start;
        const scalar_t* __restrict__ v_src_h =
            value_src + head * head_size + head_chunk_start;
        cache_t* __restrict__ k_dst_h =
            key_dst + static_cast<int64_t>(head) * head_stride + head_chunk_start;
        cache_t* __restrict__ v_dst_h =
            value_dst + static_cast<int64_t>(head) * head_stride + head_chunk_start;
        vectorize_best_effort(k_src_h, k_dst_h, head_chunk_len, lane,
                              threads_per_head, k_op);
        vectorize_best_effort(v_src_h, v_dst_h, head_chunk_len, lane,
                              threads_per_head, v_op);
      }
    }
  } else {
    CopyNoScaleOp<cache_t, scalar_t> op{};
    if (is_contiguous_heads) {
      const int n_elems = local_heads * head_size;
      const int64_t head_offset = static_cast<int64_t>(head_start) * head_size;
      vectorize_best_effort(key_src + head_offset, key_dst + head_offset,
                            n_elems, threadIdx.x, blockDim.x, op);
      vectorize_best_effort(value_src + head_offset, value_dst + head_offset,
                            n_elems, threadIdx.x, blockDim.x, op);
    } else {
      const bool use_half_warp = (threads_per_head == 16);
      const int lane = use_half_warp ? (threadIdx.x & 15) : (threadIdx.x & 31);
      const int warp_id = use_half_warp ? (threadIdx.x >> 4) : (threadIdx.x >> 5);
      const int warps_per_block =
          use_half_warp ? (blockDim.x >> 4) : (blockDim.x >> 5);
      for (int head = head_start + warp_id; head < head_end;
           head += warps_per_block) {
        const scalar_t* __restrict__ k_src_h =
            key_src + head * head_size + head_chunk_start;
        const scalar_t* __restrict__ v_src_h =
            value_src + head * head_size + head_chunk_start;
        cache_t* __restrict__ k_dst_h =
            key_dst + static_cast<int64_t>(head) * head_stride + head_chunk_start;
        cache_t* __restrict__ v_dst_h =
            value_dst + static_cast<int64_t>(head) * head_stride + head_chunk_start;
        vectorize_best_effort(k_src_h, k_dst_h, head_chunk_len, lane,
                              threads_per_head, op);
        vectorize_best_effort(v_src_h, v_dst_h, head_chunk_len, lane,
                              threads_per_head, op);
      }
    }
  }
}

static void check_inputs(const torch::Tensor& key, const torch::Tensor& value,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& slot_mapping,
                         const std::string& kv_cache_dtype,
                         const torch::Tensor& k_scale,
                         const torch::Tensor& v_scale) {
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
  TORCH_CHECK(k_scale.is_cuda(), "k_scale must be CUDA");
  TORCH_CHECK(v_scale.is_cuda(), "v_scale must be CUDA");

  TORCH_CHECK(key.device() == value.device(),
              "key and value must be on the same device");
  TORCH_CHECK(key_cache.device() == key.device(),
              "key_cache and key must be on the same device");
  TORCH_CHECK(value_cache.device() == key.device(),
              "value_cache and key must be on the same device");
  TORCH_CHECK(slot_mapping.device() == key.device(),
              "slot_mapping and key must be on the same device");
  TORCH_CHECK(k_scale.device() == key.device(),
              "k_scale and key must be on the same device");
  TORCH_CHECK(v_scale.device() == key.device(),
              "v_scale and key must be on the same device");

  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");

  TORCH_CHECK(key.dim() == 3, "key must have shape [T, H, D]");
  TORCH_CHECK(value.dim() == 3, "value must have shape [T, H, D]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must have shape [B, block_size, H, D]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must have shape [B, block_size, H, D]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must have shape [T]");

  TORCH_CHECK(key.sizes() == value.sizes(), "key and value must match shape");
  TORCH_CHECK(slot_mapping.size(0) == key.size(0),
              "slot_mapping and key must have matching T");
  TORCH_CHECK(key_cache.size(2) == key.size(1),
              "key_cache H must match key H");
  TORCH_CHECK(value_cache.size(2) == key.size(1),
              "value_cache H must match key H");
  TORCH_CHECK(key_cache.size(3) == key.size(2),
              "key_cache D must match key D");
  TORCH_CHECK(value_cache.size(3) == key.size(2),
              "value_cache D must match key D");

  TORCH_CHECK(key.stride(2) == 1, "key last dimension must be contiguous");
  TORCH_CHECK(value.stride(2) == 1, "value last dimension must be contiguous");
  TORCH_CHECK(key.stride(1) == key.size(2),
              "key head dimension must be contiguous");
  TORCH_CHECK(value.stride(1) == value.size(2),
              "value head dimension must be contiguous");
  TORCH_CHECK(key.stride(0) >= key.size(1) * key.size(2),
              "key stride(0) must be >= H*D");
  TORCH_CHECK(value.stride(0) >= value.size(1) * value.size(2),
              "value stride(0) must be >= H*D");

  TORCH_CHECK(key.scalar_type() == at::ScalarType::Half ||
                  key.scalar_type() == at::ScalarType::BFloat16,
              "key must be float16 or bfloat16");
  TORCH_CHECK(value.scalar_type() == key.scalar_type(),
              "value dtype must match key dtype");

  TORCH_CHECK(k_scale.scalar_type() == at::ScalarType::Float,
              "k_scale must be float32");
  TORCH_CHECK(v_scale.scalar_type() == at::ScalarType::Float,
              "v_scale must be float32");
  TORCH_CHECK(k_scale.numel() == 1, "k_scale must be a scalar tensor");
  TORCH_CHECK(v_scale.numel() == 1, "v_scale must be a scalar tensor");

  const bool is_fp8 =
      (kv_cache_dtype == "fp8") || (kv_cache_dtype == "fp8_e4m3");
  const int64_t head_stride = key_cache.stride(2);
  if (!is_fp8 && head_stride != key.size(2)) {
    TORCH_CHECK(static_cast<int64_t>(key.size(1)) * key.size(2) >= 32,
                "HND kv_cache layout requires num_heads * head_size >= 32");
  }
  if (is_fp8) {
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Byte,
                "fp8 kv-cache requires key_cache uint8 view");
    TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::Byte,
                "fp8 kv-cache requires value_cache uint8 view");
  } else {
    TORCH_CHECK(kv_cache_dtype == "auto",
                "Unsupported kv_cache_dtype: ", kv_cache_dtype);
    TORCH_CHECK(key_cache.scalar_type() == key.scalar_type(),
                "key_cache dtype must match key dtype for kv_cache_dtype=auto");
    TORCH_CHECK(value_cache.scalar_type() == key.scalar_type(),
                "value_cache dtype must match key dtype for kv_cache_dtype=auto");
  }

  TORCH_CHECK(key_cache.size(1) > 0, "block_size must be positive");
}

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale) {
  check_inputs(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
               k_scale, v_scale);
  if (slot_mapping.numel() == 0) {
    return;
  }

  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));

  const int64_t key_stride = key.stride(0);
  const int64_t value_stride = value.stride(0);
  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride = key_cache.stride(1);
  const int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0),
              "key_cache and value_cache must have matching stride(0)");

  const c10::cuda::CUDAGuard device_guard(key.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool is_fp8 =
      (kv_cache_dtype == "fp8") || (kv_cache_dtype == "fp8_e4m3");
  int head_blocks = 1;
  int heads_per_block = num_heads;
  int sm_count = 0;
  int threads_per_head = 32;
  if (is_fp8) {
    const auto* props = at::cuda::getDeviceProperties(key.device().index());
    sm_count = props ? props->multiProcessorCount : 0;
    int min_head_blocks = 1;
    if (sm_count > 0 && num_heads > 1) {
      // For small T (decode), split heads across more blocks to increase
      // parallelism. This is especially important for head_size=64 HND caches
      // where the baseline launch has num_tokens blocks (often < #SM).
      const int target_blocks = sm_count;
      min_head_blocks =
          std::min(num_heads, std::max(1, (target_blocks + num_tokens - 1) /
                                              std::max(1, num_tokens)));
    }
    head_blocks = min_head_blocks;
    heads_per_block = (num_heads + head_blocks - 1) / head_blocks;
    if (head_stride != head_size && head_size >= 64) {
      threads_per_head = 16;
    }
  }

  int head_chunks = 1;
  int head_chunk_size = head_size;
  if (is_fp8 && head_stride != head_size && sm_count > 0) {
    const int total_blocks = num_tokens * head_blocks;
    if (total_blocks < sm_count) {
      const int min_chunk = (total_blocks * 2 < sm_count) ? 16 : 32;
      const int max_chunks = (head_size + min_chunk - 1) / min_chunk;
      const int desired_chunks = (sm_count + total_blocks - 1) / total_blocks;
      head_chunks = std::min(max_chunks, std::max(1, desired_chunks));
      head_chunk_size = (head_size + head_chunks - 1) / head_chunks;
    }
  }

  dim3 grid(num_tokens, head_blocks, head_chunks);
  int fp8_block_threads = std::max(32, heads_per_block * threads_per_head);
  fp8_block_threads = ((fp8_block_threads + 31) / 32) * 32;
  fp8_block_threads = std::min(512, fp8_block_threads);
  dim3 block(is_fp8 ? fp8_block_threads : std::min(num_heads * head_size, 512));

  if (key.scalar_type() == at::ScalarType::BFloat16) {
    if (is_fp8) {
      reshape_and_cache_flash_kernel<__nv_bfloat16, uint8_t, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  key.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(
                  value.data_ptr<at::BFloat16>()),
              reinterpret_cast<uint8_t*>(key_cache.data_ptr<uint8_t>()),
              reinterpret_cast<uint8_t*>(value_cache.data_ptr<uint8_t>()),
              slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
              head_stride, key_stride, value_stride, num_heads, head_size,
              block_size, heads_per_block, head_chunk_size, threads_per_head,
              k_scale.data_ptr<float>(),
              v_scale.data_ptr<float>());
    } else {
      reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(
                  key.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(
                  value.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(
                  key_cache.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(
                  value_cache.data_ptr<at::BFloat16>()),
              slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
              head_stride, key_stride, value_stride, num_heads, head_size,
              block_size, heads_per_block, head_chunk_size, threads_per_head,
              k_scale.data_ptr<float>(),
              v_scale.data_ptr<float>());
    }
  } else if (key.scalar_type() == at::ScalarType::Half) {
    if (is_fp8) {
      reshape_and_cache_flash_kernel<__half, uint8_t, true><<<grid, block, 0,
                                                            stream>>>(
          reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
          reinterpret_cast<uint8_t*>(key_cache.data_ptr<uint8_t>()),
          reinterpret_cast<uint8_t*>(value_cache.data_ptr<uint8_t>()),
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride, head_stride,
          key_stride, value_stride, num_heads, head_size, block_size,
          heads_per_block, head_chunk_size, threads_per_head,
          k_scale.data_ptr<float>(),
          v_scale.data_ptr<float>());
    } else {
      reshape_and_cache_flash_kernel<__half, __half, false><<<grid, block, 0,
                                                            stream>>>(
          reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(key_cache.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(value_cache.data_ptr<at::Half>()),
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride, head_stride,
          key_stride, value_stride, num_heads, head_size, block_size,
          heads_per_block, head_chunk_size, threads_per_head,
          k_scale.data_ptr<float>(),
          v_scale.data_ptr<float>());
    }
  } else {
    TORCH_CHECK(false, "Unsupported key dtype: ", key.scalar_type());
  }
}

}  // namespace kv_cache
}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reshape_and_cache_flash", &kestrel::kv_cache::reshape_and_cache_flash,
        "reshape_and_cache_flash (CUDA)");
}
