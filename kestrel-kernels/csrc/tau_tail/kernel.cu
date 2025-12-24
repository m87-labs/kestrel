#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace kestrel {
namespace {

static_assert(sizeof(at::Half) == sizeof(__half), "Half size mismatch");
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

__device__ __forceinline__ __half2 half2_broadcast(__half x) {
  return __halves2half2(x, x);
}

__device__ __forceinline__ __nv_bfloat162 bf162_broadcast(__nv_bfloat16 x) {
  return __halves2bfloat162(x, x);
}

__device__ __forceinline__ int4 load_int4(const void* ptr) {
  return *reinterpret_cast<const int4*>(ptr);
}

__device__ __forceinline__ void store_int4(void* ptr, const int4& v) {
  *reinterpret_cast<int4*>(ptr) = v;
}

__global__ void tau_tail_apply_fp16_kernel(
    __half* __restrict__ qkv_out,
    const __half* __restrict__ tok_qv_lin,
    const __half* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int heads_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t head_idx =
      static_cast<int64_t>(blockIdx.y) * heads_per_block + warp_id;
  if (head_idx >= n_heads) return;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __half2float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const float scale_q_f = tanhf(tok_q_f) + tau_pos_f;
  const float scale_v_f = tanhf(tok_v_f) + tau_pos_f;

  const __half scale_q = __float2half_rn(scale_q_f);
  const __half scale_v = __float2half_rn(scale_v_f);
  const __half2 scale_q2 = half2_broadcast(scale_q);
  const __half2 scale_v2 = half2_broadcast(scale_v);

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;
  auto* q2_ptr = reinterpret_cast<__half2*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__half2*>(qkv_out + v_base);

  for (int i = lane; i < vecs; i += kWarpSize) {
    q2_ptr[i] = __hmul2(q2_ptr[i], scale_q2);
    v2_ptr[i] = __hmul2(v2_ptr[i], scale_v2);
  }
}

__global__ void tau_tail_apply_bf16_kernel(
    __nv_bfloat16* __restrict__ qkv_out,
    const __nv_bfloat16* __restrict__ tok_qv_lin,
    const __nv_bfloat16* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int heads_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t head_idx =
      static_cast<int64_t>(blockIdx.y) * heads_per_block + warp_id;
  if (head_idx >= n_heads) return;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __bfloat162float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __bfloat162float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f = __bfloat162float(
      tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const float scale_q_f = tanhf(tok_q_f) + tau_pos_f;
  const float scale_v_f = tanhf(tok_v_f) + tau_pos_f;

  const __nv_bfloat16 scale_q = __float2bfloat16_rn(scale_q_f);
  const __nv_bfloat16 scale_v = __float2bfloat16_rn(scale_v_f);
  const __nv_bfloat162 scale_q2 = bf162_broadcast(scale_q);
  const __nv_bfloat162 scale_v2 = bf162_broadcast(scale_v);

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;
  auto* q2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + v_base);

  for (int i = lane; i < vecs; i += kWarpSize) {
    q2_ptr[i] = __hmul2(q2_ptr[i], scale_q2);
    v2_ptr[i] = __hmul2(v2_ptr[i], scale_v2);
  }
}

__global__ void tau_tail_apply_fp16_vec16_kernel(
    __half* __restrict__ qkv_out,
    const __half* __restrict__ tok_qv_lin,
    const __half* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int group_id = lane / 8;      // 0..3
  const int lane8 = lane & 7;         // 0..7
  const int heads_per_warp = 4;
  const int heads_per_block =
      (static_cast<int>(blockDim.x) / kWarpSize) * heads_per_warp;

  const int64_t head_idx =
      static_cast<int64_t>(blockIdx.y) * heads_per_block +
      static_cast<int64_t>(warp_id) * heads_per_warp + group_id;
  if (head_idx >= n_heads) return;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __half2float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const __half2 scale_q2 =
      half2_broadcast(__float2half_rn(tanhf(tok_q_f) + tau_pos_f));
  const __half2 scale_v2 =
      half2_broadcast(__float2half_rn(tanhf(tok_v_f) + tau_pos_f));

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;  // number of half2
  auto* q2_ptr = reinterpret_cast<__half2*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__half2*>(qkv_out + v_base);

  // Each group of 8 lanes handles one head. Each lane handles 4 half2 (= 16
  // bytes).
  for (int i = lane8 * 4; i + 3 < vecs; i += 32) {
    int4 q4 = load_int4(q2_ptr + i);
    int4 v4 = load_int4(v2_ptr + i);
    auto* qh2 = reinterpret_cast<__half2*>(&q4);
    auto* vh2 = reinterpret_cast<__half2*>(&v4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qh2[j] = __hmul2(qh2[j], scale_q2);
      vh2[j] = __hmul2(vh2[j], scale_v2);
    }
    store_int4(q2_ptr + i, q4);
    store_int4(v2_ptr + i, v4);
  }

  // Tail (should be empty for head_dim multiple of 64).
  for (int i = (vecs & ~31) + lane8; i < vecs; i += 8) {
    q2_ptr[i] = __hmul2(q2_ptr[i], scale_q2);
    v2_ptr[i] = __hmul2(v2_ptr[i], scale_v2);
  }
}

__global__ void tau_tail_apply_bf16_vec16_kernel(
    __nv_bfloat16* __restrict__ qkv_out,
    const __nv_bfloat16* __restrict__ tok_qv_lin,
    const __nv_bfloat16* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int group_id = lane / 8;      // 0..3
  const int lane8 = lane & 7;         // 0..7
  const int heads_per_warp = 4;
  const int heads_per_block =
      (static_cast<int>(blockDim.x) / kWarpSize) * heads_per_warp;

  const int64_t head_idx =
      static_cast<int64_t>(blockIdx.y) * heads_per_block +
      static_cast<int64_t>(warp_id) * heads_per_warp + group_id;
  if (head_idx >= n_heads) return;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __bfloat162float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __bfloat162float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f = __bfloat162float(
      tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const __nv_bfloat162 scale_q2 = bf162_broadcast(
      __float2bfloat16_rn(tanhf(tok_q_f) + tau_pos_f));
  const __nv_bfloat162 scale_v2 = bf162_broadcast(
      __float2bfloat16_rn(tanhf(tok_v_f) + tau_pos_f));

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;  // number of bf162
  auto* q2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + v_base);

  for (int i = lane8 * 4; i + 3 < vecs; i += 32) {
    int4 q4 = load_int4(q2_ptr + i);
    int4 v4 = load_int4(v2_ptr + i);
    auto* qb2 = reinterpret_cast<__nv_bfloat162*>(&q4);
    auto* vb2 = reinterpret_cast<__nv_bfloat162*>(&v4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qb2[j] = __hmul2(qb2[j], scale_q2);
      vb2[j] = __hmul2(vb2[j], scale_v2);
    }
    store_int4(q2_ptr + i, q4);
    store_int4(v2_ptr + i, v4);
  }

  for (int i = (vecs & ~31) + lane8; i < vecs; i += 8) {
    q2_ptr[i] = __hmul2(q2_ptr[i], scale_q2);
    v2_ptr[i] = __hmul2(v2_ptr[i], scale_v2);
  }
}

__global__ void tau_tail_apply_fp16_halfwarp_kernel(
    __half* __restrict__ qkv_out,
    const __half* __restrict__ tok_qv_lin,
    const __half* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int heads_per_block = (static_cast<int>(blockDim.x) / kWarpSize) * 2;
  const int64_t head_pair_base =
      static_cast<int64_t>(blockIdx.y) * heads_per_block +
      static_cast<int64_t>(warp_id) * 2;
  const int head_offset = (lane >= 16) ? 1 : 0;
  const int64_t head_idx = head_pair_base + head_offset;
  if (head_idx >= n_heads) return;

  const int lane16 = lane & 15;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __half2float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f =
      __half2float(tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const __half2 scale_q2 =
      half2_broadcast(__float2half_rn(tanhf(tok_q_f) + tau_pos_f));
  const __half2 scale_v2 =
      half2_broadcast(__float2half_rn(tanhf(tok_v_f) + tau_pos_f));

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;
  auto* q2_ptr = reinterpret_cast<__half2*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__half2*>(qkv_out + v_base);

  int i0 = lane16;
  while (i0 < vecs) {
    const int i1 = i0 + 16;

    __half2 q0 = q2_ptr[i0];
    __half2 v0 = v2_ptr[i0];
    __half2 q1{};
    __half2 v1{};
    if (i1 < vecs) {
      q1 = q2_ptr[i1];
      v1 = v2_ptr[i1];
    }

    q0 = __hmul2(q0, scale_q2);
    v0 = __hmul2(v0, scale_v2);
    q2_ptr[i0] = q0;
    v2_ptr[i0] = v0;

    if (i1 < vecs) {
      q1 = __hmul2(q1, scale_q2);
      v1 = __hmul2(v1, scale_v2);
      q2_ptr[i1] = q1;
      v2_ptr[i1] = v1;
    }
    i0 += 32;
  }
}

__global__ void tau_tail_apply_bf16_halfwarp_kernel(
    __nv_bfloat16* __restrict__ qkv_out,
    const __nv_bfloat16* __restrict__ tok_qv_lin,
    const __nv_bfloat16* __restrict__ tau_pos_table,
    const int64_t* __restrict__ position_ids,
    int64_t qkv_dim,
    int64_t n_heads,
    int64_t head_dim) {
  constexpr int kWarpSize = 32;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);

  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int heads_per_block = (static_cast<int>(blockDim.x) / kWarpSize) * 2;
  const int64_t head_pair_base =
      static_cast<int64_t>(blockIdx.y) * heads_per_block +
      static_cast<int64_t>(warp_id) * 2;
  const int head_offset = (lane >= 16) ? 1 : 0;
  const int64_t head_idx = head_pair_base + head_offset;
  if (head_idx >= n_heads) return;

  const int lane16 = lane & 15;

  const int64_t q_dim = qkv_dim / 3;
  const int64_t tok_stride = 2 * n_heads;

  const int64_t pos = position_ids[token_idx];
  const float tau_pos_f =
      __bfloat162float(tau_pos_table[pos * n_heads + head_idx]);
  const float tok_q_f =
      __bfloat162float(tok_qv_lin[token_idx * tok_stride + head_idx]);
  const float tok_v_f = __bfloat162float(
      tok_qv_lin[token_idx * tok_stride + n_heads + head_idx]);

  const __nv_bfloat162 scale_q2 = bf162_broadcast(
      __float2bfloat16_rn(tanhf(tok_q_f) + tau_pos_f));
  const __nv_bfloat162 scale_v2 = bf162_broadcast(
      __float2bfloat16_rn(tanhf(tok_v_f) + tau_pos_f));

  const int64_t token_base = token_idx * qkv_dim;
  const int64_t q_base = token_base + head_idx * head_dim;
  const int64_t v_base = token_base + 2 * q_dim + head_idx * head_dim;

  const int64_t vecs = head_dim / 2;
  auto* q2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + q_base);
  auto* v2_ptr = reinterpret_cast<__nv_bfloat162*>(qkv_out + v_base);

  int i0 = lane16;
  while (i0 < vecs) {
    const int i1 = i0 + 16;

    __nv_bfloat162 q0 = q2_ptr[i0];
    __nv_bfloat162 v0 = v2_ptr[i0];
    __nv_bfloat162 q1{};
    __nv_bfloat162 v1{};
    if (i1 < vecs) {
      q1 = q2_ptr[i1];
      v1 = v2_ptr[i1];
    }

    q0 = __hmul2(q0, scale_q2);
    v0 = __hmul2(v0, scale_v2);
    q2_ptr[i0] = q0;
    v2_ptr[i0] = v0;

    if (i1 < vecs) {
      q1 = __hmul2(q1, scale_q2);
      v1 = __hmul2(v1, scale_v2);
      q2_ptr[i1] = q1;
      v2_ptr[i1] = v1;
    }
    i0 += 32;
  }
}

void check_inputs(const torch::Tensor& qkv_out,
                  const torch::Tensor& tok_qv_lin,
                  const torch::Tensor& tau_pos_table,
                  const torch::Tensor& position_ids) {
  TORCH_CHECK(qkv_out.is_cuda(), "qkv_out must be CUDA");
  TORCH_CHECK(tok_qv_lin.is_cuda(), "tok_qv_lin must be CUDA");
  TORCH_CHECK(tau_pos_table.is_cuda(), "tau_pos_table must be CUDA");
  TORCH_CHECK(position_ids.is_cuda(), "position_ids must be CUDA");

  TORCH_CHECK(qkv_out.is_contiguous(), "qkv_out must be contiguous");
  TORCH_CHECK(tok_qv_lin.is_contiguous(), "tok_qv_lin must be contiguous");
  TORCH_CHECK(tau_pos_table.is_contiguous(), "tau_pos_table must be contiguous");
  TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");

  TORCH_CHECK(qkv_out.dim() == 3, "qkv_out must be rank-3 (B,S,C)");
  TORCH_CHECK(tok_qv_lin.dim() == 3, "tok_qv_lin must be rank-3 (B,S,2H)");
  TORCH_CHECK(tau_pos_table.dim() == 2, "tau_pos_table must be rank-2 (P,H)");
  TORCH_CHECK(position_ids.dim() == 2, "position_ids must be rank-2 (B,S)");

  TORCH_CHECK(qkv_out.size(0) == tok_qv_lin.size(0) &&
                  qkv_out.size(1) == tok_qv_lin.size(1),
              "qkv_out and tok_qv_lin must match in (B,S)");
  TORCH_CHECK(qkv_out.size(0) == position_ids.size(0) &&
                  qkv_out.size(1) == position_ids.size(1),
              "qkv_out and position_ids must match in (B,S)");

  TORCH_CHECK(qkv_out.scalar_type() == tok_qv_lin.scalar_type() &&
                  qkv_out.scalar_type() == tau_pos_table.scalar_type(),
              "qkv_out/tok_qv_lin/tau_pos_table must have the same dtype");
  TORCH_CHECK(qkv_out.scalar_type() == at::ScalarType::Half ||
                  qkv_out.scalar_type() == at::ScalarType::BFloat16,
              "qkv_out must be fp16 or bf16");

  TORCH_CHECK(position_ids.scalar_type() == at::ScalarType::Long,
              "position_ids must be int64");

  TORCH_CHECK((qkv_out.size(-1) % 3) == 0,
              "qkv_out last dim must be divisible by 3 (no GQA)");
  TORCH_CHECK((tok_qv_lin.size(-1) % 2) == 0,
              "tok_qv_lin last dim must be divisible by 2");

  const int64_t qkv_dim = qkv_out.size(-1);
  const int64_t q_dim = qkv_dim / 3;
  const int64_t n_heads = tok_qv_lin.size(-1) / 2;
  TORCH_CHECK(q_dim % n_heads == 0, "q_dim must be divisible by n_heads");
  const int64_t head_dim = q_dim / n_heads;
  TORCH_CHECK(head_dim * n_heads == q_dim, "invalid head_dim");
  TORCH_CHECK((head_dim % 2) == 0, "head_dim must be even");
  TORCH_CHECK(qkv_dim == 3 * q_dim, "qkv_out must be QKV packed (Q=K=V)");

  TORCH_CHECK(tau_pos_table.size(1) == n_heads,
              "tau_pos_table second dim must match n_heads");
}

void tau_tail_apply_cuda(torch::Tensor& qkv_out,
                         const torch::Tensor& tok_qv_lin,
                         const torch::Tensor& tau_pos_table,
                         const torch::Tensor& position_ids) {
  check_inputs(qkv_out, tok_qv_lin, tau_pos_table, position_ids);
  if (qkv_out.numel() == 0) {
    return;
  }

  const int64_t qkv_dim = qkv_out.size(-1);
  const int64_t n_heads = tok_qv_lin.size(-1) / 2;
  const int64_t q_dim = qkv_dim / 3;
  const int64_t head_dim = q_dim / n_heads;
  const int64_t num_tokens = qkv_out.size(0) * qkv_out.size(1);

  const c10::cuda::CUDAGuard device_guard(qkv_out.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kThreads = 256;
  const bool use_vec16 = ((head_dim % 64) == 0);
  const bool use_halfwarp = (!use_vec16 && (num_tokens >= 256));
  dim3 grid(static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(
                use_vec16 ? ((n_heads + 31) / 32)
                          : (use_halfwarp ? ((n_heads + 15) / 16)
                                          : ((n_heads + 7) / 8))),
            1);

  if (qkv_out.scalar_type() == at::ScalarType::Half) {
    auto* qkv_ptr = reinterpret_cast<__half*>(qkv_out.data_ptr<at::Half>());
    const auto* tok_ptr =
        reinterpret_cast<const __half*>(tok_qv_lin.data_ptr<at::Half>());
    const auto* tau_ptr = reinterpret_cast<const __half*>(
        tau_pos_table.data_ptr<at::Half>());
    const auto* pos_ptr = position_ids.data_ptr<int64_t>();
    if (use_vec16) {
      tau_tail_apply_fp16_vec16_kernel<<<grid, kThreads, 0, stream>>>(
          qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
    } else if (use_halfwarp) {
      tau_tail_apply_fp16_halfwarp_kernel<<<grid, kThreads, 0, stream>>>(
          qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
    } else {
      tau_tail_apply_fp16_kernel<<<grid, kThreads, 0, stream>>>(
          qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto* qkv_ptr =
      reinterpret_cast<__nv_bfloat16*>(qkv_out.data_ptr<at::BFloat16>());
  const auto* tok_ptr =
      reinterpret_cast<const __nv_bfloat16*>(tok_qv_lin.data_ptr<at::BFloat16>());
  const auto* tau_ptr = reinterpret_cast<const __nv_bfloat16*>(
      tau_pos_table.data_ptr<at::BFloat16>());
  const auto* pos_ptr = position_ids.data_ptr<int64_t>();
  if (use_vec16) {
    tau_tail_apply_bf16_vec16_kernel<<<grid, kThreads, 0, stream>>>(
        qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
  } else if (use_halfwarp) {
    tau_tail_apply_bf16_halfwarp_kernel<<<grid, kThreads, 0, stream>>>(
        qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
  } else {
    tau_tail_apply_bf16_kernel<<<grid, kThreads, 0, stream>>>(
        qkv_ptr, tok_ptr, tau_ptr, pos_ptr, qkv_dim, n_heads, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tau_tail_apply_cuda", &kestrel::tau_tail_apply_cuda,
        "Tau tail fused tanh+gather+Q/V scaling (CUDA)");
}

