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
  int dtype;  // cudaDataType_t
  int epilogue;  // cublasLtEpilogue_t
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
    mix(static_cast<uint64_t>(x.epilogue));
    mix(static_cast<uint64_t>(x.beta_tag));
    mix(static_cast<uint64_t>(x.bias_ptr));
    mix(static_cast<uint64_t>(x.m));
    mix(static_cast<uint64_t>(x.n));
    mix(static_cast<uint64_t>(x.k));
    return h;
  }
};

inline bool operator==(const MatmulKey& a, const MatmulKey& b) {
  return a.device == b.device && a.dtype == b.dtype && a.epilogue == b.epilogue &&
         a.beta_tag == b.beta_tag && a.bias_ptr == b.bias_ptr && a.m == b.m &&
         a.n == b.n && a.k == b.k;
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

std::mutex plan_mu;
std::unordered_map<MatmulKey, std::unique_ptr<MatmulPlan>, MatmulKeyHash>
    plan_cache;

struct PrefGuard {
  cublasLtMatmulPreference_t pref = nullptr;
  ~PrefGuard() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
  }
};

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
      static_cast<int>(epilogue),
      beta_tag,
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bias)),
      m,
      n,
      k,
  };

  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto it = plan_cache.find(key);
    if (it != plan_cache.end()) {
      return it->second.get();
    }
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

  // Bias epilogues require a bias pointer for heuristic selection, and also for
  // the matmul itself. We bake it into the plan to avoid mutating shared state
  // at call time (thread-safety).
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
    if (!inserted) {
      return it->second.get();
    }
    return it->second.get();
  }
}

void lt_matmul(
    const void* A,
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
  TORCH_CHECK(
      st == CUBLAS_STATUS_SUCCESS,
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
                  const torch::Tensor& hidden,
                  const torch::Tensor& x,
                  const torch::Tensor& w1,
                  const torch::Tensor& b1,
                  const torch::Tensor& w2,
                  const torch::Tensor& b2,
                  const torch::Tensor& residual) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w1.is_cuda(), "w1 must be CUDA");
  TORCH_CHECK(w2.is_cuda(), "w2 must be CUDA");
  TORCH_CHECK(b1.is_cuda(), "b1 must be CUDA");
  TORCH_CHECK(b2.is_cuda(), "b2 must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");

  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, in_dim)");
  TORCH_CHECK(w1.dim() == 2, "w1 must be 2D (hidden_dim, in_dim)");
  TORCH_CHECK(w2.dim() == 2, "w2 must be 2D (out_dim, hidden_dim)");
  TORCH_CHECK(b1.dim() == 1, "b1 must be 1D (hidden_dim,)");
  TORCH_CHECK(b2.dim() == 1, "b2 must be 1D (out_dim,)");
  TORCH_CHECK(out.dim() == 2, "out must be 2D (M, out_dim)");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D (M, hidden_dim)");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2D (M, out_dim)");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w1.is_contiguous(), "w1 must be contiguous");
  TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
  TORCH_CHECK(b1.is_contiguous(), "b1 must be contiguous");
  TORCH_CHECK(b2.is_contiguous(), "b2 must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");

  TORCH_CHECK(x.scalar_type() == w1.scalar_type(), "x and w1 must have same dtype");
  TORCH_CHECK(x.scalar_type() == w2.scalar_type(), "x and w2 must have same dtype");
  TORCH_CHECK(x.scalar_type() == b1.scalar_type(), "x and b1 must have same dtype");
  TORCH_CHECK(x.scalar_type() == b2.scalar_type(), "x and b2 must have same dtype");
  TORCH_CHECK(x.scalar_type() == out.scalar_type(), "x and out must have same dtype");
  TORCH_CHECK(x.scalar_type() == hidden.scalar_type(), "x and hidden must have same dtype");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(), "x and residual must have same dtype");

  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16,
              "Only float16/bfloat16 are supported");

  const auto M = x.size(0);
  const auto in_dim = x.size(1);
  const auto hidden_dim = w1.size(0);
  TORCH_CHECK(w1.size(1) == in_dim, "w1 must have shape (hidden_dim, in_dim)");
  TORCH_CHECK(b1.size(0) == hidden_dim, "b1 must have shape (hidden_dim,)");
  TORCH_CHECK(hidden.size(0) == M && hidden.size(1) == hidden_dim,
              "hidden must have shape (M, hidden_dim)");

  const auto out_dim = w2.size(0);
  TORCH_CHECK(w2.size(1) == hidden_dim, "w2 must have shape (out_dim, hidden_dim)");
  TORCH_CHECK(b2.size(0) == out_dim, "b2 must have shape (out_dim,)");
  TORCH_CHECK(out.size(0) == M && out.size(1) == out_dim, "out must have shape (M, out_dim)");
  TORCH_CHECK(residual.size(0) == M && residual.size(1) == out_dim,
              "residual must have shape (M, out_dim)");
}

}  // namespace

void fused_mlp_gelu_bias_residual_cuda(torch::Tensor& out,
                                      torch::Tensor& hidden,
                                      torch::Tensor& x,
                                      torch::Tensor& w1,
                                      torch::Tensor& b1,
                                      torch::Tensor& w2,
                                      torch::Tensor& b2,
                                      torch::Tensor& residual) {
  check_inputs(out, hidden, x, w1, b1, w2, b2, residual);

  const c10::cuda::CUDAGuard device_guard(x.device());
  int device = -1;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int m = static_cast<int>(x.size(0));
  const int in_dim = static_cast<int>(x.size(1));
  const int hidden_dim = static_cast<int>(w1.size(0));
  const int out_dim = static_cast<int>(w2.size(0));

  const cudaDataType_t dtype =
      (x.scalar_type() == at::ScalarType::BFloat16) ? CUDA_R_16BF : CUDA_R_16F;

  // FC1: hidden = GELU( x @ w1^T + b1 )
  const cublasLtEpilogue_t fc1_epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // We compute hidden^T = w1 @ x^T in column-major, so bias matches hidden_dim.
    // Layouts:
    // - A (w1): row-major [hidden_dim, in_dim]
    // - B (x^T): column-major [in_dim, M] backed by row-major x [M, in_dim]
    // - D (hidden^T): column-major [hidden_dim, M] backed by row-major hidden [M, hidden_dim]
    lt_matmul(
        /*A=*/w1.data_ptr(),
        /*B=*/x.data_ptr(),
        /*C=*/hidden.data_ptr(),
        /*D=*/hidden.data_ptr(),
        /*m=*/hidden_dim,
        /*n=*/m,
        /*k=*/in_dim,
        /*dtype=*/dtype,
        /*epilogue=*/fc1_epilogue,
        /*bias=*/b1.data_ptr(),
        /*alpha=*/alpha,
        /*beta=*/beta,
        /*stream=*/stream,
        /*device=*/device);
  }

  // FC2: out = hidden @ w2^T + b2 + residual
  const cublasLtEpilogue_t fc2_epilogue = CUBLASLT_EPILOGUE_BIAS;
  {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    // We compute out^T = w2 @ hidden^T in column-major, so bias matches out_dim.
    // Layouts:
    // - A (w2): row-major [out_dim, hidden_dim]
    // - B (hidden^T): column-major [hidden_dim, M] backed by row-major hidden [M, hidden_dim]
    // - C (residual^T): column-major [out_dim, M] backed by row-major residual [M, out_dim]
    // - D (out^T): column-major [out_dim, M] backed by row-major out [M, out_dim]
    lt_matmul(
        /*A=*/w2.data_ptr(),
        /*B=*/hidden.data_ptr(),
        /*C=*/residual.data_ptr(),
        /*D=*/out.data_ptr(),
        /*m=*/out_dim,
        /*n=*/m,
        /*k=*/hidden_dim,
        /*dtype=*/dtype,
        /*epilogue=*/fc2_epilogue,
        /*bias=*/b2.data_ptr(),
        /*alpha=*/alpha,
        /*beta=*/beta,
        /*stream=*/stream,
        /*device=*/device);
  }
}

}  // namespace kestrel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_mlp_gelu_bias_residual_cuda",
        &kestrel::fused_mlp_gelu_bias_residual_cuda,
        "Fused MLP: fc1+gelu+bias then fc2+bias+residual (CUDA)");
}
