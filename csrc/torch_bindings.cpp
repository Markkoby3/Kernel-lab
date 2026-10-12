// Registers the CUDA kernels as PyTorch operators: torch.ops.kernel_lab.*.
//
// Ops are registered with TORCH_LIBRARY (not pybind11 functions) so they work
// with torch.compile and can be called from TorchScript and C++. Outputs are
// passed in ("Tensor!" = mutated) so callers control allocation, as vLLM does.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kernel_lab.h"

namespace {

kernel_lab::DType to_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return kernel_lab::DType::kFloat32;
    case at::kHalf:
      return kernel_lab::DType::kFloat16;
    case at::kBFloat16:
      return kernel_lab::DType::kBFloat16;
    default:
      TORCH_CHECK(false, "kernel_lab: unsupported dtype ", t);
  }
}

void check_cuda(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, "kernel_lab: ", what, " failed: ", cudaGetErrorString(err));
}

void rms_norm_impl(at::Tensor& out, const at::Tensor& x, at::Tensor* residual,
                   const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
  TORCH_CHECK(x.dim() >= 1 && x.stride(-1) == 1, "x must have a contiguous last dimension");
  TORCH_CHECK(out.is_contiguous() && weight.is_contiguous(), "out and weight must be contiguous");
  const int64_t hidden = x.size(-1);
  TORCH_CHECK(weight.numel() == hidden, "weight size ", weight.numel(), " != hidden ", hidden);
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == out.scalar_type(),
              "x, weight and out must share a dtype");

  // Rows of x may be strided (e.g. a slice); view everything as [rows, hidden].
  const at::Tensor x2 = x.dim() == 2 ? x : x.reshape({-1, hidden});
  TORCH_CHECK(x2.stride(-1) == 1, "x cannot be viewed as rows with a contiguous last dim");
  void* res_ptr = nullptr;
  if (residual != nullptr) {
    TORCH_CHECK(residual->is_contiguous() && residual->sizes() == x.sizes() &&
                    residual->scalar_type() == x.scalar_type(),
                "residual must be contiguous, same shape and dtype as x");
    res_ptr = residual->data_ptr();
  }

  const c10::cuda::CUDAGuard guard(x.device());
  check_cuda(kernel_lab::launch_rms_norm(to_dtype(x.scalar_type()), out.data_ptr(), x2.data_ptr(),
                                         res_ptr, weight.data_ptr(), static_cast<int>(x2.size(0)),
                                         static_cast<int>(hidden), x2.stride(0),
                                         static_cast<float>(eps),
                                         at::cuda::getCurrentCUDAStream()),
             "rms_norm");
}

void rms_norm(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps) {
  rms_norm_impl(out, x, nullptr, weight, eps);
}

void fused_add_rms_norm(at::Tensor& out, const at::Tensor& x, at::Tensor& residual,
                        const at::Tensor& weight, double eps) {
  rms_norm_impl(out, x, &residual, weight, eps);
}

void paged_decode_attention(at::Tensor& out, const at::Tensor& q, const at::Tensor& k_cache,
                            const at::Tensor& v_cache, const at::Tensor& block_table,
                            const at::Tensor& seq_lens, double scale) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "tensors must be on CUDA");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4, "q is [batch, heads, dim]; caches are 4-D");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && k_cache.is_contiguous() &&
                  v_cache.is_contiguous(),
              "q, out and both caches must be contiguous");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shapes differ");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt,
              "block_table and seq_lens must be int32");
  TORCH_CHECK(block_table.stride(-1) == 1 && seq_lens.is_contiguous(),
              "block_table rows and seq_lens must be contiguous");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == out.scalar_type(),
              "q, caches and out must share a dtype");

  const int batch = static_cast<int>(q.size(0));
  const int num_q_heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int page_size = static_cast<int>(k_cache.size(1));
  const int num_kv_heads = static_cast<int>(k_cache.size(2));
  TORCH_CHECK(k_cache.size(3) == head_dim, "cache head_dim differs from q");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be a multiple of num_kv_heads");

  const c10::cuda::CUDAGuard guard(q.device());
  check_cuda(kernel_lab::launch_paged_decode_attention(
                 to_dtype(q.scalar_type()), out.data_ptr(), q.data_ptr(), k_cache.data_ptr(),
                 v_cache.data_ptr(), block_table.data_ptr<int32_t>(),
                 seq_lens.data_ptr<int32_t>(), batch, num_q_heads, num_kv_heads, head_dim,
                 page_size, block_table.stride(0), static_cast<float>(scale),
                 at::cuda::getCurrentCUDAStream()),
             "paged_decode_attention (head_dim must be 64/128/256, group 1/2/4/8)");
}

}  // namespace

TORCH_LIBRARY(kernel_lab, m) {
  m.def("rms_norm(Tensor! out, Tensor x, Tensor weight, float eps) -> ()");
  m.def("fused_add_rms_norm(Tensor! out, Tensor x, Tensor! residual, Tensor weight, float eps) -> ()");
  m.def(
      "paged_decode_attention(Tensor! out, Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor block_table, Tensor seq_lens, float scale) -> ()");
}

TORCH_LIBRARY_IMPL(kernel_lab, CUDA, m) {
  m.impl("rms_norm", &rms_norm);
  m.impl("fused_add_rms_norm", &fused_add_rms_norm);
  m.impl("paged_decode_attention", &paged_decode_attention);
}

// Importing kernel_lab._C runs the registrations above.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
