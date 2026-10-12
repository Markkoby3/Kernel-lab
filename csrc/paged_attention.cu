// Paged-KV decode attention in CUDA: a straightforward warp-per-token design.
//
// One thread block per (sequence, KV head), with 4 warps. Each warp walks every
// 4th token of the sequence. For a token, the 32 lanes each load a contiguous
// slice of K (head_dim / 32 elements, one vector load), form partial dot products
// with every query head in the GQA group, and finish them with a warp shuffle
// reduction. Each lane then updates an online softmax (running max, normaliser
// and output accumulator) for its slice of V. The 4 warps' partial results are
// merged once at the end through shared memory with the log-sum-exp rule.
//
// This is a readable baseline, not a tuned kernel. It uses CUDA cores rather
// than tensor cores, keeps one token in flight per warp, and has no split-KV, so
// small batches leave SMs idle. The Triton kernel (split-KV, tl.dot on tensor
// cores) and FlashInfer show what those choices are worth; docs/design.md
// discusses the gap.
#include <cfloat>
#include <cstdint>

#include "common.cuh"
#include "kernel_lab.h"

namespace kernel_lab {
namespace {

constexpr int kWarps = 4;

template <typename T, int HEAD_DIM, int GROUP>
__global__ void __launch_bounds__(kWarps * 32)
    paged_decode_kernel(T* __restrict__ out, const T* __restrict__ q,
                        const T* __restrict__ k_cache, const T* __restrict__ v_cache,
                        const int32_t* __restrict__ block_table,
                        const int32_t* __restrict__ seq_lens, int num_kv_heads, int page_size,
                        int64_t block_table_stride, float scale) {
  constexpr int PER_LANE = HEAD_DIM / 32;
  using P = Pack<T, PER_LANE>;

  const int b = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int seq_len = seq_lens[b];
  const int num_q_heads = num_kv_heads * GROUP;

  // Query slices for every head in the group, pre-scaled.
  float qv[GROUP][PER_LANE];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const P pq = reinterpret_cast<const P*>(
        q + (static_cast<int64_t>(b) * num_q_heads + kv_head * GROUP + g) * HEAD_DIM)[lane];
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) qv[g][i] = Num<T>::to_float(pq.v[i]) * scale;
  }

  float m[GROUP], l[GROUP], acc[GROUP][PER_LANE];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    m[g] = -FLT_MAX;
    l[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) acc[g][i] = 0.0f;
  }

  const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * HEAD_DIM;
  const int64_t page_stride = token_stride * page_size;
  const int32_t* pages = block_table + b * block_table_stride;

  for (int t = warp; t < seq_len; t += kWarps) {
    const int64_t offset = static_cast<int64_t>(pages[t / page_size]) * page_stride +
                           (t % page_size) * token_stride +
                           static_cast<int64_t>(kv_head) * HEAD_DIM;
    const P pk = reinterpret_cast<const P*>(k_cache + offset)[lane];
    const P pv = reinterpret_cast<const P*>(v_cache + offset)[lane];

#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) dot += qv[g][i] * Num<T>::to_float(pk.v[i]);
      const float score = warp_sum(dot);  // every lane now holds the full score

      // Score and softmax update in one pass, so no per-group score array stays live.
      const float m_new = fmaxf(m[g], score);
      const float alpha = __expf(m[g] - m_new);
      const float p = __expf(score - m_new);
      l[g] = l[g] * alpha + p;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        acc[g][i] = acc[g][i] * alpha + p * Num<T>::to_float(pv.v[i]);
      }
      m[g] = m_new;
    }
  }

  // Merge the warps: out = sum_w e^(m_w - M) acc_w / sum_w e^(m_w - M) l_w.
  __shared__ float s_m[kWarps][GROUP];
  __shared__ float s_l[kWarps][GROUP];
  __shared__ float s_acc[kWarps][GROUP][HEAD_DIM];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    if (lane == 0) {
      s_m[warp][g] = m[g];
      s_l[warp][g] = l[g];
    }
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) s_acc[warp][g][lane * PER_LANE + i] = acc[g][i];
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < GROUP * HEAD_DIM; idx += blockDim.x) {
    const int g = idx / HEAD_DIM;
    const int d = idx % HEAD_DIM;
    float max_m = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) max_m = fmaxf(max_m, s_m[w][g]);
    float denom = 0.0f;
    float numer = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (s_l[w][g] > 0.0f) {  // a warp with no tokens contributes nothing
        const float c = __expf(s_m[w][g] - max_m);
        denom += s_l[w][g] * c;
        numer += s_acc[w][g][d] * c;
      }
    }
    out[(static_cast<int64_t>(b) * num_q_heads + kv_head * GROUP + g) * HEAD_DIM + d] =
        Num<T>::from_float(numer / denom);
  }
}

template <typename T, int HEAD_DIM, int GROUP>
cudaError_t launch_with(void* out, const void* q, const void* k_cache, const void* v_cache,
                        const int32_t* block_table, const int32_t* seq_lens, int batch,
                        int num_kv_heads, int page_size, int64_t block_table_stride, float scale,
                        cudaStream_t stream) {
  const dim3 grid(batch, num_kv_heads);
  paged_decode_kernel<T, HEAD_DIM, GROUP><<<grid, kWarps * 32, 0, stream>>>(
      static_cast<T*>(out), static_cast<const T*>(q), static_cast<const T*>(k_cache),
      static_cast<const T*>(v_cache), block_table, seq_lens, num_kv_heads, page_size,
      block_table_stride, scale);
  return cudaGetLastError();
}

template <typename T, int HEAD_DIM>
cudaError_t dispatch_group(int group, void* out, const void* q, const void* k, const void* v,
                           const int32_t* bt, const int32_t* lens, int batch, int num_kv_heads,
                           int page_size, int64_t bt_stride, float scale, cudaStream_t stream) {
  switch (group) {
    case 1:
      return launch_with<T, HEAD_DIM, 1>(out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    case 2:
      return launch_with<T, HEAD_DIM, 2>(out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    case 4:
      return launch_with<T, HEAD_DIM, 4>(out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    case 8:
      return launch_with<T, HEAD_DIM, 8>(out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <typename T>
cudaError_t dispatch_head_dim(int head_dim, int group, void* out, const void* q, const void* k,
                              const void* v, const int32_t* bt, const int32_t* lens, int batch,
                              int num_kv_heads, int page_size, int64_t bt_stride, float scale,
                              cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return dispatch_group<T, 64>(group, out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    case 128:
      return dispatch_group<T, 128>(group, out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    case 256:
      return dispatch_group<T, 256>(group, out, q, k, v, bt, lens, batch, num_kv_heads, page_size, bt_stride, scale, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t launch_paged_decode_attention(DType dtype, void* out, const void* q,
                                          const void* k_cache, const void* v_cache,
                                          const int32_t* block_table, const int32_t* seq_lens,
                                          int batch, int num_q_heads, int num_kv_heads,
                                          int head_dim, int page_size,
                                          int64_t block_table_stride, float scale,
                                          cudaStream_t stream) {
  if (batch == 0) return cudaSuccess;
  if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return cudaErrorInvalidValue;
  const int group = num_q_heads / num_kv_heads;
  switch (dtype) {
    case DType::kFloat32:
      return dispatch_head_dim<float>(head_dim, group, out, q, k_cache, v_cache, block_table, seq_lens, batch, num_kv_heads, page_size, block_table_stride, scale, stream);
    case DType::kFloat16:
      return dispatch_head_dim<__half>(head_dim, group, out, q, k_cache, v_cache, block_table, seq_lens, batch, num_kv_heads, page_size, block_table_stride, scale, stream);
    case DType::kBFloat16:
      return dispatch_head_dim<__nv_bfloat16>(head_dim, group, out, q, k_cache, v_cache, block_table, seq_lens, batch, num_kv_heads, page_size, block_table_stride, scale, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kernel_lab
