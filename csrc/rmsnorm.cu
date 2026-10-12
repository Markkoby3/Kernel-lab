// RMSNorm and fused residual-add + RMSNorm.
//
// One thread block per row. Threads stride over the row in 16-byte packs
// (8 fp16/bf16 or 4 fp32 values), accumulate the sum of squares in fp32, and
// reduce it across the block with warp shuffles. A second pass scales and
// writes the output. That second pass re-reads data this block touched
// microseconds earlier, so it is served from L2 rather than DRAM; DRAM traffic
// stays at one read and one write per tensor.
#include <cstdint>

#include "common.cuh"
#include "kernel_lab.h"

namespace kernel_lab {
namespace {

template <typename T, int VEC, bool HAS_RESIDUAL>
__global__ void rms_norm_kernel(T* __restrict__ out, const T* __restrict__ x,
                                T* __restrict__ residual, const T* __restrict__ weight, int hidden,
                                int64_t x_row_stride, float eps) {
  using P = Pack<T, VEC>;
  const int64_t row = blockIdx.x;
  const P* x_row = reinterpret_cast<const P*>(x + row * x_row_stride);
  P* res_row = HAS_RESIDUAL ? reinterpret_cast<P*>(residual + row * hidden) : nullptr;
  P* out_row = reinterpret_cast<P*>(out + row * hidden);
  const P* w = reinterpret_cast<const P*>(weight);
  const int packs = hidden / VEC;

  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    P a = x_row[i];
    if constexpr (HAS_RESIDUAL) {
      const P r = res_row[i];
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        // Round the sum to T: that rounded value is what the residual stream keeps.
        a.v[j] = Num<T>::from_float(Num<T>::to_float(a.v[j]) + Num<T>::to_float(r.v[j]));
      }
      res_row[i] = a;
    }
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float f = Num<T>::to_float(a.v[j]);
      sum_sq += f * f;
    }
  }

  const float rstd = rsqrtf(block_sum(sum_sq) / static_cast<float>(hidden) + eps);

  // Each thread re-reads exactly the packs it wrote above, so there is no race.
  const P* src = HAS_RESIDUAL ? res_row : x_row;
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    const P a = src[i];
    const P g = w[i];
    P o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      o.v[j] = Num<T>::from_float(Num<T>::to_float(a.v[j]) * rstd * Num<T>::to_float(g.v[j]));
    }
    out_row[i] = o;
  }
}

bool aligned(const void* p, int bytes) {
  return reinterpret_cast<uintptr_t>(p) % bytes == 0;
}

template <typename T>
cudaError_t launch_typed(void* out, const void* x, void* residual, const void* weight, int rows,
                         int hidden, int64_t x_row_stride, float eps, cudaStream_t stream) {
  constexpr int VEC = 16 / sizeof(T);
  const bool vectorize = hidden % VEC == 0 && x_row_stride % VEC == 0 && aligned(out, 16) &&
                         aligned(x, 16) && aligned(weight, 16) &&
                         (residual == nullptr || aligned(residual, 16));
  const int packs = vectorize ? hidden / VEC : hidden;
  // One pack per thread where possible, in whole warps, at most 1024 threads.
  const int threads = std::min(1024, std::max(32, (packs + 31) / 32 * 32));

  auto o = static_cast<T*>(out);
  auto xi = static_cast<const T*>(x);
  auto r = static_cast<T*>(residual);
  auto w = static_cast<const T*>(weight);
  if (vectorize) {
    if (residual) {
      rms_norm_kernel<T, VEC, true><<<rows, threads, 0, stream>>>(o, xi, r, w, hidden, x_row_stride, eps);
    } else {
      rms_norm_kernel<T, VEC, false><<<rows, threads, 0, stream>>>(o, xi, r, w, hidden, x_row_stride, eps);
    }
  } else {
    if (residual) {
      rms_norm_kernel<T, 1, true><<<rows, threads, 0, stream>>>(o, xi, r, w, hidden, x_row_stride, eps);
    } else {
      rms_norm_kernel<T, 1, false><<<rows, threads, 0, stream>>>(o, xi, r, w, hidden, x_row_stride, eps);
    }
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_rms_norm(DType dtype, void* out, const void* x, void* residual,
                            const void* weight, int rows, int hidden, int64_t x_row_stride,
                            float eps, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  switch (dtype) {
    case DType::kFloat32:
      return launch_typed<float>(out, x, residual, weight, rows, hidden, x_row_stride, eps, stream);
    case DType::kFloat16:
      return launch_typed<__half>(out, x, residual, weight, rows, hidden, x_row_stride, eps, stream);
    case DType::kBFloat16:
      return launch_typed<__nv_bfloat16>(out, x, residual, weight, rows, hidden, x_row_stride, eps,
                                         stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kernel_lab
