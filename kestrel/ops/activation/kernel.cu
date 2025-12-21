#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace kestrel {

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "BFloat16 size mismatch");

#if defined(__CUDA_ARCH__)
  #define KESTREL_LDG(arg) __ldg(arg)
#else
  #define KESTREL_LDG(arg) *(arg)
#endif

__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

inline bool is_16byte_aligned_host(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

inline int next_pow2_int(int v) {
  v = v - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

inline int pick_split(int64_t num_tokens) {
  if (num_tokens <= 32) {
    return 4;
  }
  if (num_tokens <= 64) {
    return 2;
  }
  return 1;
}
__device__ __forceinline__ __nv_bfloat16 gelu_bf16(__nv_bfloat16 x) {
  const float f = __bfloat162float(x);
  constexpr float kAlpha = 0.7071067811865476f;
  const float y = f * 0.5f * (1.0f + ::erff(f * kAlpha));
  return __float2bfloat16_rn(y);
}

__device__ __forceinline__ __nv_bfloat16 bf16_one() {
  __nv_bfloat16_raw raw;
  raw.x = 0x3f80;  // bf16 encoding for 1.0f
  return __nv_bfloat16(raw);
}

__device__ __forceinline__ __nv_bfloat162 bf162_one() {
  const __nv_bfloat16 one = bf16_one();
  return __halves2bfloat162(one, one);
}

__device__ __forceinline__ __nv_bfloat16 add_one_bf16(__nv_bfloat16 x) {
  return __hadd(x, bf16_one());
}

__device__ __forceinline__ __nv_bfloat16 mul_bf16(__nv_bfloat16 a,
                                                  __nv_bfloat16 b) {
  return __hmul(a, b);
}

__device__ __forceinline__ __nv_bfloat16 gelu_residual_direct(
    __nv_bfloat16 x, __nv_bfloat16 y) {
  return mul_bf16(gelu_bf16(x), add_one_bf16(y));
}

__global__ void gelu_residual_kernel(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ input,
                                     const int d, const int num_vecs,
                                     const int vecs_per_block) {
  constexpr int VEC_SIZE = 16 / sizeof(__nv_bfloat16);
  const int64_t token_idx = blockIdx.x;
  const int vec_start = blockIdx.y * vecs_per_block;
  const int vec_end = min(vec_start + vecs_per_block, num_vecs);
  const __nv_bfloat16* x_ptr = input + token_idx * 2 * d;
  const __nv_bfloat16* y_ptr = x_ptr + d;
  __nv_bfloat16* out_ptr = out + token_idx * d;

  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= VEC_SIZE) {
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const __nv_bfloat162 one2 = bf162_one();

    for (int i = vec_start + threadIdx.x; i < vec_end; i += blockDim.x) {
      int4 x = KESTREL_LDG(&x_vec[i]);
      int4 y = KESTREL_LDG(&y_vec[i]);
      int4 r;
      auto* xp = reinterpret_cast<__nv_bfloat16*>(&x);
      auto* yp = reinterpret_cast<__nv_bfloat16*>(&y);
      auto* rp = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j += 2) {
        const __nv_bfloat16 gelu0 = gelu_bf16(xp[j]);
        const __nv_bfloat16 gelu1 = gelu_bf16(xp[j + 1]);
        const __nv_bfloat162 gelu2 = __halves2bfloat162(gelu0, gelu1);
        const __nv_bfloat162 y2 =
            *reinterpret_cast<const __nv_bfloat162*>(&yp[j]);
        const __nv_bfloat162 g2 = __hadd2(y2, one2);
        const __nv_bfloat162 out2 = __hmul2(gelu2, g2);
        *reinterpret_cast<__nv_bfloat162*>(&rp[j]) = out2;
      }
      out_vec[i] = r;
    }

    if (vec_end == num_vecs) {
      const int scalar_start = num_vecs * VEC_SIZE;
      for (int i = scalar_start + threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = gelu_residual_direct(KESTREL_LDG(&x_ptr[i]),
                                          KESTREL_LDG(&y_ptr[i]));
      }
    }
  } else {
    const int elem_start = vec_start * VEC_SIZE;
    const int elem_end = min(vec_end * VEC_SIZE, d);
    for (int64_t idx = elem_start + threadIdx.x; idx < elem_end;
         idx += blockDim.x) {
      const __nv_bfloat16 x = KESTREL_LDG(&x_ptr[idx]);
      const __nv_bfloat16 y = KESTREL_LDG(&y_ptr[idx]);
      out_ptr[idx] = gelu_residual_direct(x, y);
    }
  }
}

void check_inputs(const torch::Tensor& out, const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "input must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
              "out must be bfloat16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dims");
  TORCH_CHECK(out.dim() == input.dim(), "out must match input rank");
  TORCH_CHECK(input.size(-1) == out.size(-1) * 2,
              "input last dim must be 2x out last dim");
  const int64_t d = input.size(-1) / 2;
  TORCH_CHECK((d % 8) == 0,
              "hidden dimension must be a multiple of 8 for bf16 vectorization");
  const void* in_ptr = input.data_ptr<at::BFloat16>();
  const void* out_ptr = out.data_ptr<at::BFloat16>();
  TORCH_CHECK(is_16byte_aligned_host(in_ptr),
              "input must be 16-byte aligned");
  TORCH_CHECK(is_16byte_aligned_host(out_ptr),
              "out must be 16-byte aligned");
}

void gelu_residual_cuda(torch::Tensor& out, torch::Tensor& input) {
  check_inputs(out, input);
  if (input.numel() == 0) {
    return;
  }

  const int64_t d = input.size(-1) / 2;
  const int64_t num_tokens = input.numel() / input.size(-1);
  const int vec_elems = 16 / sizeof(__nv_bfloat16);

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* in_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());

  const bool aligned = is_16byte_aligned_host(in_ptr) &&
                       is_16byte_aligned_host(out_ptr) &&
                       ((d * sizeof(__nv_bfloat16)) % 16 == 0) &&
                       (d % vec_elems == 0);
  int split = aligned ? pick_split(num_tokens) : 1;
  int num_vecs = aligned ? static_cast<int>(d / vec_elems) : 0;
  if (aligned && split > num_vecs) {
    split = num_vecs;
  }
  const int vecs_per_block =
      aligned ? (num_vecs + split - 1) / split : 0;
  const int vecs_per_thread_target = (num_tokens <= 32) ? 2 : 1;
  int threads = aligned
                    ? next_pow2_int(std::max(1, (vecs_per_block + vecs_per_thread_target - 1) /
                                                  vecs_per_thread_target))
                    : static_cast<int>(std::min<int64_t>(d, 1024));
  threads = aligned ? std::max(threads, 32) : threads;
  threads = aligned ? std::min(threads, 128) : threads;
  dim3 grid(num_tokens, split, 1);
  gelu_residual_kernel<<<grid, threads, 0, stream>>>(
      out_ptr, in_ptr, static_cast<int>(d), num_vecs, vecs_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_residual_cuda", &kestrel::gelu_residual_cuda,
        "GELU(h)*(g+1) residual activation (CUDA)");
}
