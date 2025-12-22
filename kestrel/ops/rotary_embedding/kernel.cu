#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace kestrel {
namespace pos_encoding {

#if defined(__CUDA_ARCH__)
  #define KESTREL_LDG(arg) __ldg(arg)
#else
  #define KESTREL_LDG(arg) *(arg)
#endif

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

// Assumptions (by design):
// - GPT-NeoX rotary embedding only.
// - bf16 only.
// - positions: contiguous int64, shape [B, S].
// - query/key: contiguous bf16, shape [B, S, heads, head_size].
// - cos_sin_cache: contiguous fp32, shape [max_pos, rot_dim] with vLLM layout
//   (cos first, sin second; both length rot_dim/2).
// - rot_dim % 4 == 0 (so the 2-offset-per-thread kernel covers rot offsets in pairs).

__device__ __forceinline__ void apply_rotary_neox_fp32(
    __nv_bfloat16* __restrict__ vec, const int rot_offset, const int embed_dim,
    const float cos, const float sin) {
  const int x_index = rot_offset;
  const int y_index = embed_dim + rot_offset;

  const float x = __bfloat162float(vec[x_index]);
  const float y = __bfloat162float(vec[y_index]);

  // Match vixtral-train semantics: fp32 math, then cast once to bf16.
  const float out_x =
      __fsub_rn(__fmul_rn(x, cos), __fmul_rn(y, sin));  // x*cos - y*sin
  const float out_y =
      __fadd_rn(__fmul_rn(y, cos), __fmul_rn(x, sin));  // y*cos + x*sin

  vec[x_index] = __float2bfloat16_rn(out_x);
  vec[y_index] = __float2bfloat16_rn(out_y);
}

// Vectorized version processing two consecutive rotary offsets (rot0, rot0+1).
// Assumes rot0 is even.
__device__ __forceinline__ void apply_rotary_neox_fp32_pair(
    __nv_bfloat16* __restrict__ vec, const int rot0, const int embed_dim,
    const float cos0, const float sin0, const float cos1, const float sin1) {
  const int x_index = rot0;
  const int y_index = embed_dim + rot0;

  const auto* __restrict__ x_ptr =
      reinterpret_cast<const __nv_bfloat162*>(vec + x_index);
  const auto* __restrict__ y_ptr =
      reinterpret_cast<const __nv_bfloat162*>(vec + y_index);

  const float2 x2 = __bfloat1622float2(*x_ptr);  // {x0, x1}
  const float2 y2 = __bfloat1622float2(*y_ptr);  // {y0, y1}

  const float out_x0 =
      __fsub_rn(__fmul_rn(x2.x, cos0), __fmul_rn(y2.x, sin0));
  const float out_x1 =
      __fsub_rn(__fmul_rn(x2.y, cos1), __fmul_rn(y2.y, sin1));
  const float out_y0 =
      __fadd_rn(__fmul_rn(y2.x, cos0), __fmul_rn(x2.x, sin0));
  const float out_y1 =
      __fadd_rn(__fmul_rn(y2.y, cos1), __fmul_rn(x2.y, sin1));

  auto* __restrict__ out_x_ptr =
      reinterpret_cast<__nv_bfloat162*>(vec + x_index);
  auto* __restrict__ out_y_ptr =
      reinterpret_cast<__nv_bfloat162*>(vec + y_index);
  *out_x_ptr = __floats2bfloat162_rn(out_x0, out_x1);
  *out_y_ptr = __floats2bfloat162_rn(out_y0, out_y1);
}

__global__ void rotary_embedding_neox_fp32_kernel(
    const int64_t* __restrict__ positions, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, const float* __restrict__ cos_sin_cache,
    const int num_heads,
    const int num_kv_heads, const int head_size, const int rot_dim) {
  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];

  const int embed_dim = rot_dim / 2;
  const int embed_dim2 = embed_dim / 2;

  extern __shared__ float sh_cache[];
  const float* __restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;
  for (int i = threadIdx.x; i < rot_dim; i += blockDim.x) {
    sh_cache[i] = KESTREL_LDG(&cache_ptr[i]);
  }
  __syncthreads();
  const float* __restrict__ cos_ptr = sh_cache;
  const float* __restrict__ sin_ptr = sh_cache + embed_dim;

  const int query_hidden_size = num_heads * head_size;
  const int key_hidden_size = num_kv_heads * head_size;
  __nv_bfloat16* __restrict__ q_token = query + token_idx * query_hidden_size;
  __nv_bfloat16* __restrict__ k_token = key + token_idx * key_hidden_size;

  const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  const int max_n2 = max_heads * embed_dim2;

  for (int i = threadIdx.x; i < max_n2; i += blockDim.x) {
    const int head_idx = i / embed_dim2;
    const int offset2 = i - head_idx * embed_dim2;
    const int rot0 = offset2 * 2;

    // Shared for q/k.
    const float cos0 = cos_ptr[rot0];
    const float sin0 = sin_ptr[rot0];
    const float cos1 = cos_ptr[rot0 + 1];
    const float sin1 = sin_ptr[rot0 + 1];

    if (head_idx < num_heads) {
      __nv_bfloat16* __restrict__ q_head = q_token + head_idx * head_size;
      apply_rotary_neox_fp32_pair(q_head, rot0, embed_dim, cos0, sin0, cos1,
                                  sin1);
    }
    if (head_idx < num_kv_heads) {
      __nv_bfloat16* __restrict__ k_head = k_token + head_idx * head_size;
      apply_rotary_neox_fp32_pair(k_head, rot0, embed_dim, cos0, sin0, cos1,
                                  sin1);
    }
  }
}

// Split-head variant of the 2-offset-per-thread kernel. This increases kernel
// occupancy for small token counts (decode) by launching multiple blocks per
// token, each operating on a disjoint range of heads.
__global__ void rotary_embedding_neox_fp32_split_heads_kernel(
    const int64_t* __restrict__ positions, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, const float* __restrict__ cos_sin_cache,
    const int num_heads,
    const int num_kv_heads, const int head_size, const int rot_dim,
    const int heads_per_block) {
    const int token_idx = blockIdx.x;
  const int head_group = blockIdx.y;
  const int64_t pos = positions[token_idx];

  const int embed_dim = rot_dim / 2;
  const int embed_dim2 = embed_dim / 2;

  extern __shared__ float sh_cache[];
  const float* __restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;
  for (int i = threadIdx.x; i < rot_dim; i += blockDim.x) {
    sh_cache[i] = KESTREL_LDG(&cache_ptr[i]);
  }
  __syncthreads();
  const float* __restrict__ cos_ptr = sh_cache;
  const float* __restrict__ sin_ptr = sh_cache + embed_dim;

  const int query_hidden_size = num_heads * head_size;
  const int key_hidden_size = num_kv_heads * head_size;
  __nv_bfloat16* __restrict__ q_token = query + token_idx * query_hidden_size;
  __nv_bfloat16* __restrict__ k_token = key + token_idx * key_hidden_size;

  const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  const int head_start = head_group * heads_per_block;
  if (head_start >= max_heads) {
    return;
  }
  const int head_end =
      head_start + heads_per_block < max_heads ? head_start + heads_per_block
                                               : max_heads;

  const int start_i = head_start * embed_dim2;
  const int end_i = head_end * embed_dim2;
  for (int i = threadIdx.x + start_i; i < end_i; i += blockDim.x) {
    const int head_idx = i / embed_dim2;
    const int offset2 = i - head_idx * embed_dim2;
    const int rot0 = offset2 * 2;

    const float cos0 = cos_ptr[rot0];
    const float sin0 = sin_ptr[rot0];
    const float cos1 = cos_ptr[rot0 + 1];
    const float sin1 = sin_ptr[rot0 + 1];

    if (head_idx < num_heads) {
      __nv_bfloat16* __restrict__ q_head = q_token + head_idx * head_size;
      apply_rotary_neox_fp32_pair(q_head, rot0, embed_dim, cos0, sin0, cos1,
                                  sin1);
    }
    if (head_idx < num_kv_heads) {
      __nv_bfloat16* __restrict__ k_head = k_token + head_idx * head_size;
      apply_rotary_neox_fp32_pair(k_head, rot0, embed_dim, cos0, sin0, cos1,
                                  sin1);
    }
  }
}

static void check_inputs(const torch::Tensor& positions,
                         const torch::Tensor& query, const torch::Tensor& key,
                         const torch::Tensor& cos_sin_cache,
                         const int64_t head_size) {
  TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA");

  TORCH_CHECK(positions.device() == query.device(),
              "positions and query must be on the same device");
  TORCH_CHECK(key.device() == query.device(),
              "key and query must be on the same device");
  TORCH_CHECK(cos_sin_cache.device() == query.device(),
              "cos_sin_cache and query must be on the same device");

  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache must be contiguous");

  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::BFloat16,
              "query must be bfloat16");
  TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16,
              "key must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float,
              "cos_sin_cache must be float32");

  TORCH_CHECK(positions.dim() == 2, "positions must have shape [B, S]");
  TORCH_CHECK(query.dim() == 4, "query must have shape [B, S, H, D]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [B, S, H, D]");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "cos_sin_cache must have shape [max_pos, rot_dim]");

  TORCH_CHECK(query.size(0) == positions.size(0) &&
                  query.size(1) == positions.size(1),
              "query and positions must have matching [B, S]");
  TORCH_CHECK(key.size(0) == positions.size(0) &&
                  key.size(1) == positions.size(1),
              "key and positions must have matching [B, S]");

  TORCH_CHECK(head_size > 0, "head_size must be positive");
  TORCH_CHECK(query.size(-1) == head_size,
              "query head_size must match head_size argument");
  TORCH_CHECK(key.size(-1) == head_size,
              "key head_size must match head_size argument");

  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  TORCH_CHECK((rot_dim % 4) == 0, "rot_dim must be divisible by 4");
  TORCH_CHECK(rot_dim <= head_size, "rot_dim must be <= head_size");
}

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache) {
  check_inputs(positions, query, key, cos_sin_cache, head_size);
  if (positions.numel() == 0) {
    return;
  }

  const int64_t num_tokens = positions.numel();
  const int num_heads = static_cast<int>(query.size(2));
  const int num_kv_heads = static_cast<int>(key.size(2));
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));

  const int embed_dim = rot_dim / 2;
  const int embed_dim2 = embed_dim / 2;
  const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  const int work_pair = max_heads * embed_dim2;

  // Heuristic: our 2-offset-per-thread kernel tends to win for very small token
  // counts (decode), and for very large token counts (high batching). The
  // 2-offset-per-thread kernel is also better for prefill sizes after
  // vectorization + shared cos/sin caching.
  const bool maybe_use_split_heads_kernel =
      (num_tokens <= 64) && (max_heads > 1);

  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int split = 1;
  int heads_per_block = max_heads;
  if (maybe_use_split_heads_kernel) {
    const cudaDeviceProp* __restrict__ props =
        at::cuda::getCurrentDeviceProperties();
    const int sm_count = props->multiProcessorCount;

    const int split_target =
        static_cast<int>((sm_count + num_tokens - 1) / num_tokens);
    split = std::min(std::max(split_target, 1), max_heads);
    heads_per_block = (max_heads + split - 1) / split;
  }

  const bool use_split_heads_kernel =
      maybe_use_split_heads_kernel && (split > 1);

  int threads = 32;
  int work = work_pair;
  if (use_split_heads_kernel) {
    work = heads_per_block * embed_dim2;
  }
  while (threads < work) {
    threads <<= 1;
  }
  threads = std::min(threads, 256);

  dim3 grid(static_cast<unsigned int>(num_tokens));
  const int shared_bytes = rot_dim * static_cast<int>(sizeof(float));
  if (use_split_heads_kernel) {
    dim3 split_grid(static_cast<unsigned int>(num_tokens),
                    static_cast<unsigned int>(split));
    rotary_embedding_neox_fp32_split_heads_kernel<<<split_grid, threads,
                                                   shared_bytes, stream>>>(
        positions.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        cos_sin_cache.data_ptr<float>(),
        num_heads, num_kv_heads, static_cast<int>(head_size), rot_dim,
        heads_per_block);
  } else {
    rotary_embedding_neox_fp32_kernel<<<grid, threads, shared_bytes, stream>>>(
        positions.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        cos_sin_cache.data_ptr<float>(),
        num_heads, num_kv_heads, static_cast<int>(head_size), rot_dim);
  }
}

}  // namespace pos_encoding
}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &kestrel::pos_encoding::rotary_embedding,
        "rotary_embedding (CUDA)");
}
