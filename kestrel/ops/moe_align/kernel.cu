#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <optional>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

#define VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)        \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::UInt16, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt32, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt64, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                  \
                     VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(__VA_ARGS__))

namespace kestrel {
namespace moe {

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, int32_t model_offset, int32_t inactive_expert_id,
    int32_t topk_num, int32_t* token_mask, bool has_expert_map) {
  extern __shared__ int32_t shared_counts[];

  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  if (blockIdx.x % 2) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = static_cast<int>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  const size_t fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int32_t fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t max_num_m_blocks,
    int32_t inactive_expert_id, int32_t model_offset, int32_t topk_num,
    int32_t* token_mask, bool has_expert_map) {
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    total_tokens_post_pad[model_offset] = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t* __restrict__ token_mask,
    int32_t model_offset, int32_t topk_num, bool has_expert_map) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }

    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int32_t rank_post_pad = atomicAdd(
          &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
          i;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t topk_num, bool has_expert_map) {
  _moe_align_block_size(topk_ids, sorted_token_ids, expert_ids,
                        total_tokens_post_pad, expert_map, num_experts,
                        padded_num_experts, experts_per_warp, block_size, numel,
                        cumsum, max_num_tokens_padded,
                        CEILDIV(max_num_tokens_padded, block_size), 0, 0,
                        topk_num, nullptr, has_expert_map);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  _count_and_sort_expert_tokens(topk_ids, sorted_token_ids, cumsum_buffer,
                                expert_map, numel, num_experts,
                                max_num_tokens_padded, nullptr, 0, topk_num,
                                has_expert_map);
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t topk_num,
    bool has_expert_map) {
  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), 0, 0, topk_num, nullptr,
      has_expert_map);
}

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor expert_map) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard device_guard(topk_ids.device());

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  const bool has_expert_map = expert_map.numel() > 0;
  torch::Tensor expert_map_tensor = has_expert_map ? expert_map
                                                   : torch::empty({0}, options_int);

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        const bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t local_threads = std::max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((local_threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);
          constexpr int32_t fill_threads = 256;
          auto small_kernel =
              moe_align_block_size_small_batch_expert_kernel<scalar_t,
                                                            fill_threads>;
          small_kernel<<<1, fill_threads + local_threads, shared_mem_size,
                         stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map_tensor.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        } else {
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);
          auto align_kernel = moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map_tensor.data_ptr<int32_t>(), num_experts,
              padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
              sorted_token_ids.size(0), topk_ids.size(1), has_expert_map);

          const int block_threads = std::min(256, threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          dim3 gridDims(1, actual_blocks);

          auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(),
              expert_map_tensor.data_ptr<int32_t>(), topk_ids.numel(),
              num_experts, sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        }
      });
}

}  // namespace moe
}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &kestrel::moe::moe_align_block_size,
        "moe_align_block_size (CUDA)");
}
