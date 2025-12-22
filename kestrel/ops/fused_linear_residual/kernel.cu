#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace kestrel {
namespace {

#define KESTREL_CHECK_CUBLASLT(cmd)                                             \
  do {                                                                          \
    cublasStatus_t _s = (cmd);                                                  \
    TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, "cublasLt error: ", (int)_s);      \
  } while (0)

struct LtState {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 128ull << 20;  // 128 MiB
  int device = -1;
  std::mutex mu;
};

LtState& lt_state() {
  static LtState s;
  return s;
}

void ensure_lt(int device) {
  auto& s = lt_state();
  std::lock_guard<std::mutex> lock(s.mu);
  if (s.handle == nullptr || s.device != device) {
    if (s.workspace != nullptr) {
      TORCH_CHECK(cudaFree(s.workspace) == cudaSuccess, "cudaFree(workspace) failed");
      s.workspace = nullptr;
    }
    if (s.handle != nullptr) {
      KESTREL_CHECK_CUBLASLT(cublasLtDestroy(s.handle));
      s.handle = nullptr;
    }
    KESTREL_CHECK_CUBLASLT(cublasLtCreate(&s.handle));
    s.device = device;
  }
  if (s.workspace == nullptr) {
    TORCH_CHECK(cudaMalloc(&s.workspace, s.workspace_bytes) == cudaSuccess,
                "cudaMalloc(workspace) failed");
  }
}

struct MatmulKey {
  int device;
  int dtype;     // cudaDataType_t
  int beta_tag;  // 0 if beta==0 else 1
  uint64_t bias_ptr;
  int m;
  int n;
  int k;
};

struct MatmulKeyHash {
  size_t operator()(const MatmulKey& x) const noexcept {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(x.device));
    mix(static_cast<uint64_t>(x.dtype));
    mix(static_cast<uint64_t>(x.beta_tag));
    mix(static_cast<uint64_t>(x.bias_ptr));
    mix(static_cast<uint64_t>(x.m));
    mix(static_cast<uint64_t>(x.n));
    mix(static_cast<uint64_t>(x.k));
    return h;
  }
};

inline bool operator==(const MatmulKey& a, const MatmulKey& b) {
  return a.device == b.device && a.dtype == b.dtype && a.beta_tag == b.beta_tag &&
         a.bias_ptr == b.bias_ptr && a.m == b.m && a.n == b.n && a.k == b.k;
}

struct MatmulPlan {
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulAlgo_t algo{};

  MatmulPlan() = default;
  MatmulPlan(const MatmulPlan&) = delete;
  MatmulPlan& operator=(const MatmulPlan&) = delete;

  ~MatmulPlan() {
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
  }
};

struct PrefGuard {
  cublasLtMatmulPreference_t pref = nullptr;
  ~PrefGuard() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
  }
};

std::mutex plan_mu;
std::unordered_map<MatmulKey, std::unique_ptr<MatmulPlan>, MatmulKeyHash>
    plan_cache;

MatmulPlan* get_plan(cublasLtHandle_t handle,
                     cudaDataType_t dtype,
                     cublasLtEpilogue_t epilogue,
                     const void* bias,
                     float beta,
                     int m,
                     int n,
                     int k,
                     size_t workspace_bytes,
                     int device) {
  const int beta_tag = (beta == 0.0f) ? 0 : 1;
  const MatmulKey key{
      device,
      static_cast<int>(dtype),
      beta_tag,
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bias)),
      m,
      n,
      k,
  };

  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto it = plan_cache.find(key);
    if (it != plan_cache.end()) return it->second.get();
  }

  auto plan = std::make_unique<MatmulPlan>();

  const cublasOperation_t opA = CUBLAS_OP_N;
  const cublasOperation_t opB = CUBLAS_OP_N;
  const cublasLtOrder_t orderA = CUBLASLT_ORDER_ROW;
  const cublasLtOrder_t orderB = CUBLASLT_ORDER_COL;
  const cublasLtOrder_t orderC = CUBLASLT_ORDER_COL;
  const cublasLtOrder_t orderD = CUBLASLT_ORDER_COL;

  const int lda = k;  // row-major A: cols
  const int ldb = k;  // col-major B: rows
  const int ldc = m;  // col-major C: rows
  const int ldd = m;  // col-major D: rows

  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;

  KESTREL_CHECK_CUBLASLT(
      cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, scale_type));
  KESTREL_CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  KESTREL_CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  KESTREL_CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  TORCH_CHECK(bias != nullptr, "bias must be non-null for epilogue ", (int)epilogue);
  KESTREL_CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

  const int a_rows = m;
  const int a_cols = k;
  const int b_rows = k;
  const int b_cols = n;

  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan->a_layout, dtype, a_rows, a_cols, lda));
  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));

  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan->b_layout, dtype, b_rows, b_cols, ldb));
  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB)));

  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan->c_layout, dtype, m, n, ldc));
  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderC, sizeof(orderC)));

  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan->d_layout, dtype, m, n, ldd));
  KESTREL_CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan->d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderD, sizeof(orderD)));

  PrefGuard pref;
  KESTREL_CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref.pref));
  KESTREL_CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
      pref.pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes)));

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      handle,
      plan->matmul_desc,
      plan->a_layout,
      plan->b_layout,
      plan->c_layout,
      plan->d_layout,
      pref.pref,
      1,
      &heur,
      &returned);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS && returned > 0,
              "cublasLtMatmulAlgoGetHeuristic failed: status=",
              (int)st,
              " returned=",
              returned,
              " epilogue=",
              (int)epilogue,
              " dtype=",
              (int)dtype,
              " m=",
              m,
              " n=",
              n,
              " k=",
              k);

  plan->algo = heur.algo;

  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto [it, inserted] = plan_cache.emplace(key, std::move(plan));
    if (!inserted) return it->second.get();
    return it->second.get();
  }
}

void lt_matmul(const void* A,
               const void* B,
               const void* C,
               void* D,
               int m,
               int n,
               int k,
               cudaDataType_t dtype,
               cublasLtEpilogue_t epilogue,
               const void* bias,
               float alpha,
               float beta,
               cudaStream_t stream,
               int device) {
  ensure_lt(device);
  auto& s = lt_state();
  MatmulPlan* plan = get_plan(
      s.handle, dtype, epilogue, bias, beta, m, n, k, s.workspace_bytes, device);

  cublasStatus_t st = cublasLtMatmul(
      s.handle,
      plan->matmul_desc,
      &alpha,
      A,
      plan->a_layout,
      B,
      plan->b_layout,
      &beta,
      C,
      plan->c_layout,
      D,
      plan->d_layout,
      &plan->algo,
      s.workspace,
      s.workspace_bytes,
      stream);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
              "cublasLtMatmul failed: status=",
              (int)st,
              " epilogue=",
              (int)epilogue,
              " dtype=",
              (int)dtype,
              " m=",
              m,
              " n=",
              n,
              " k=",
              k,
              " beta=",
              beta);
}

void check_inputs(const torch::Tensor& out,
                  const torch::Tensor& x,
                  const torch::Tensor& w,
                  const torch::Tensor& b,
                  const torch::Tensor& residual) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "w must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");

  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, in_dim)");
  TORCH_CHECK(w.dim() == 2, "w must be 2D (out_dim, in_dim)");
  TORCH_CHECK(b.dim() == 1, "b must be 1D (out_dim,)");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2D (M, out_dim)");
  TORCH_CHECK(out.dim() == 2, "out must be 2D (M, out_dim)");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w must have same dtype");
  TORCH_CHECK(x.scalar_type() == b.scalar_type(), "x and b must have same dtype");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(),
              "x and residual must have same dtype");
  TORCH_CHECK(x.scalar_type() == out.scalar_type(), "x and out must have same dtype");

  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16,
              "Only float16/bfloat16 are supported");

  const auto M = x.size(0);
  const auto in_dim = x.size(1);
  const auto out_dim = w.size(0);
  TORCH_CHECK(w.size(1) == in_dim, "w must have shape (out_dim, in_dim)");
  TORCH_CHECK(b.size(0) == out_dim, "b must have shape (out_dim,)");
  TORCH_CHECK(out.size(0) == M && out.size(1) == out_dim, "out must have shape (M, out_dim)");
  TORCH_CHECK(residual.size(0) == M && residual.size(1) == out_dim,
              "residual must have shape (M, out_dim)");
}

}  // namespace

void fused_linear_bias_residual_cuda(torch::Tensor& out,
                                    torch::Tensor& x,
                                    torch::Tensor& w,
                                    torch::Tensor& b,
                                    torch::Tensor& residual) {
  check_inputs(out, x, w, b, residual);

  const c10::cuda::CUDAGuard device_guard(x.device());
  int device = -1;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int M = static_cast<int>(x.size(0));
  const int in_dim = static_cast<int>(x.size(1));
  const int out_dim = static_cast<int>(w.size(0));

  const cudaDataType_t dtype =
      (x.scalar_type() == at::ScalarType::BFloat16) ? CUDA_R_16BF : CUDA_R_16F;

  const float alpha = 1.0f;
  const float beta = 1.0f;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;

  // Compute out^T = w @ x^T + b + residual^T (all column-major descriptors),
  // so bias length matches out_dim ("D rows" requirement).
  lt_matmul(
      /*A=*/w.data_ptr(),
      /*B=*/x.data_ptr(),
      /*C=*/residual.data_ptr(),
      /*D=*/out.data_ptr(),
      /*m=*/out_dim,
      /*n=*/M,
      /*k=*/in_dim,
      /*dtype=*/dtype,
      /*epilogue=*/epilogue,
      /*bias=*/b.data_ptr(),
      /*alpha=*/alpha,
      /*beta=*/beta,
      /*stream=*/stream,
      /*device=*/device);
}

}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_linear_bias_residual_cuda",
        &kestrel::fused_linear_bias_residual_cuda,
        "Fused linear+bias+residual add (CUDA)");
}

