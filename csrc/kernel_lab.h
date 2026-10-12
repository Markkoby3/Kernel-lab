// Launchers for the CUDA kernels. This header has no PyTorch dependency, so the
// kernels can be compiled and checked with plain nvcc (see the CI cuda job);
// torch_bindings.cpp adapts them to PyTorch tensors.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace kernel_lab {

enum class DType { kFloat32, kFloat16, kBFloat16 };

// RMSNorm over rows of length `hidden`. With `residual` non-null this is the fused
// pre-norm step: residual <- x + residual (in place), out <- rmsnorm(residual).
// `residual`, `out` and `weight` are contiguous; `x` rows are `x_row_stride` apart.
cudaError_t launch_rms_norm(DType dtype, void* out, const void* x, void* residual,
                            const void* weight, int rows, int hidden, int64_t x_row_stride,
                            float eps, cudaStream_t stream);

// Single-token decode attention over a paged KV cache (NHD layout).
//   q, out       [batch, num_q_heads, head_dim]                      contiguous
//   k/v_cache    [num_pages, page_size, num_kv_heads, head_dim]      contiguous
//   block_table  [batch, block_table_stride] int32
//   seq_lens     [batch] int32, every entry >= 1
// Supported: head_dim in {64, 128, 256}; num_q_heads / num_kv_heads in {1, 2, 4, 8}.
cudaError_t launch_paged_decode_attention(DType dtype, void* out, const void* q,
                                          const void* k_cache, const void* v_cache,
                                          const int32_t* block_table, const int32_t* seq_lens,
                                          int batch, int num_q_heads, int num_kv_heads,
                                          int head_dim, int page_size,
                                          int64_t block_table_stride, float scale,
                                          cudaStream_t stream);

}  // namespace kernel_lab
