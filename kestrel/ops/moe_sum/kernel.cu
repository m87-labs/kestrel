#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace kestrel {
namespace moe {

#if defined(__CUDA_ARCH__)
  #define KESTREL_LDG(arg) __ldg(arg)
#else
  #define KESTREL_LDG(arg) *(arg)
#endif

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

// Fast path rationale (why this beats torch.sum(dim=1)):
// - Special-cases the exact production case: topk=8, bf16, contiguous, H % 8 == 0.
// - Vectorizes over H with 16B int4 loads/stores (8 bf16 at a time) for fewer memory ops.
// - Fully unrolls the reduction over K=8 and accumulates in registers (no generic reduction machinery).
// PyTorch's reduction kernel is more general (arbitrary K/strides/dtypes), so it carries extra indexing
// and control overhead and may not hit the same vectorized memory access pattern for this layout.
__global__ void moe_sum_bf16x8_kernel(__nv_bfloat16* __restrict__ out,
                                      const __nv_bfloat16* __restrict__ input,
                                      const int d) {
  constexpr int TOPK = 8;
  constexpr int VEC_ELEMS = 8;

  const int64_t token_idx = blockIdx.x;
  const int num_vecs = d / VEC_ELEMS;

  const int4* __restrict__ input_vec =
      reinterpret_cast<const int4*>(input + token_idx * TOPK * d);
  int4* __restrict__ out_vec = reinterpret_cast<int4*>(out + token_idx * d);

  for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
    float2 acc0 = {0.f, 0.f};
    float2 acc1 = {0.f, 0.f};
    float2 acc2 = {0.f, 0.f};
    float2 acc3 = {0.f, 0.f};

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int4 v = KESTREL_LDG(&input_vec[k * num_vecs + vec_idx]);
      const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
      const float2 f0 = __bfloat1622float2(pairs[0]);
      const float2 f1 = __bfloat1622float2(pairs[1]);
      const float2 f2 = __bfloat1622float2(pairs[2]);
      const float2 f3 = __bfloat1622float2(pairs[3]);
      acc0.x += f0.x;
      acc0.y += f0.y;
      acc1.x += f1.x;
      acc1.y += f1.y;
      acc2.x += f2.x;
      acc2.y += f2.y;
      acc3.x += f3.x;
      acc3.y += f3.y;
    }

    int4 out_v;
    auto* out_pairs = reinterpret_cast<__nv_bfloat162*>(&out_v);
    out_pairs[0] = __float22bfloat162_rn(acc0);
    out_pairs[1] = __float22bfloat162_rn(acc1);
    out_pairs[2] = __float22bfloat162_rn(acc2);
    out_pairs[3] = __float22bfloat162_rn(acc3);
    out_vec[vec_idx] = out_v;
  }
}

static void check_inputs(const torch::Tensor& input, const torch::Tensor& output) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output must have the same dtype");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
                  input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "unsupported dtype");
  TORCH_CHECK(input.dim() == 3, "input must have shape [num_tokens, top_k, hidden]");
  TORCH_CHECK(output.dim() == 2, "output must have shape [num_tokens, hidden]");
  TORCH_CHECK(input.size(0) == output.size(0),
              "output num_tokens must match input");
  TORCH_CHECK(input.size(-1) == output.size(-1),
              "output hidden must match input hidden");
}

void moe_sum(torch::Tensor& input, torch::Tensor& output) {
  check_inputs(input, output);
  if (input.numel() == 0) {
    return;
  }

  const int hidden_size = static_cast<int>(input.size(-1));
  const int topk = static_cast<int>(input.size(1));
  const int64_t num_tokens = input.size(0);

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(static_cast<unsigned int>(num_tokens));

  if (topk == 8 && input.scalar_type() == at::ScalarType::BFloat16 &&
      (hidden_size % 8) == 0) {
    dim3 bf16_block(std::min(256, hidden_size / 8));
    auto* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    const auto* in_ptr =
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    moe_sum_bf16x8_kernel<<<grid, bf16_block, 0, stream>>>(out_ptr, in_ptr,
                                                          hidden_size);
  } else {
    at::sum_out(output, input, 1);
  }
}

}  // namespace moe
}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sum", &kestrel::moe::moe_sum, "moe_sum (CUDA)");
}
