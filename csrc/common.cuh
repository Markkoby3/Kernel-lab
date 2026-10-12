// Shared device helpers: dtype conversion, packed vector loads, warp/block reductions.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace kernel_lab {

// Explicit conversions. PyTorch builds define __CUDA_NO_HALF_CONVERSIONS__, so
// implicit half<->float casts are unavailable; these work either way.
template <typename T>
struct Num;

template <>
struct Num<float> {
  static __device__ __forceinline__ float to_float(float x) { return x; }
  static __device__ __forceinline__ float from_float(float x) { return x; }
};

template <>
struct Num<__half> {
  static __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
  static __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }
};

template <>
struct Num<__nv_bfloat16> {
  static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float x) {
    return __float2bfloat16_rn(x);
  }
};

// N elements loaded or stored as one aligned transaction (16 bytes at most), so
// a warp reads whole 128-byte cache lines instead of many 2-byte pieces.
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T v[N];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// Sum across the block; every thread receives the result. Call once per kernel
// (it uses static shared memory).
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float warp_partials[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  v = warp_sum(v);
  if (lane == 0) warp_partials[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float w = lane < num_warps ? warp_partials[lane] : 0.0f;
    w = warp_sum(w);
    if (lane == 0) total = w;
  }
  __syncthreads();
  return total;
}

}  // namespace kernel_lab
