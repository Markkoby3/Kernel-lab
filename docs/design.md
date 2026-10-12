# Kernel design notes

## Both kernels are memory-bound, so bandwidth is the scoreboard

A kernel is compute-bound only if it does more FLOPs per byte of DRAM traffic
than the GPU's ridge point (peak FLOP/s divided by peak bytes/s). On an H100 SXM
that is 989 TFLOP/s / 3.35 TB/s ≈ **295 FLOPs per byte**.

| Kernel | FLOPs per byte (fp16) | Ridge on H100 | Bound by |
|---|---:|---:|---|
| Fused add + RMSNorm | ~0.6 | 295 | DRAM bandwidth |
| Decode attention, GQA group 4 (Llama-3-8B) | ~4 | 295 | DRAM bandwidth |

For decode attention, each cached K and V element (2 bytes) is used for one
multiply-add per query head sharing it. With 4 query heads per KV head that is
about 4 FLOPs per byte, roughly 70x below the ridge. Faster math cannot help.
The only levers are moving fewer bytes and keeping every memory channel busy.
That is why the benchmark reports **% of peak DRAM bandwidth**, not TFLOP/s.

A useful lower bound: Llama-3-8B decode at batch 32 with 4,096-token contexts
reads 537 MB of KV cache per layer step. At 3.35 TB/s that cannot take less than
**160 µs**, whatever the kernel does.

## Fused add + RMSNorm

In a pre-norm transformer every layer computes `residual = x + residual`, then
`rms_norm(residual)`. Done as two kernels, the sum is written to DRAM and read
straight back: 6 tensor-sized transfers. Fused, it is 4 (read x and residual,
write the new residual and the output), cutting traffic by a third.

- **Triton:** one program per row. A whole row (up to 16K elements) is a single
  block in registers, so the sum of squares is one `tl.sum`, with no second pass
  over memory. Warps scale with row width to keep per-thread register use flat.
- **CUDA:** one block per row, 16-byte vector loads (8 fp16 values per load), a
  warp-shuffle reduction, then a second pass that re-reads data the block
  touched microseconds earlier, served from L2 rather than DRAM. Unaligned or
  odd-width rows fall back to scalar loads.
- **Rounding:** the sum is rounded to the activation dtype *before*
  normalizing, because that rounded value is what the residual stream stores.
  Kernels and reference round at the same point, so the updated residual
  matches bit for bit and tests require exact equality.

## Paged decode attention

### Layout

KV cache: `[num_pages, page_size, num_kv_heads, head_dim]` (FlashInfer's NHD
layout). A block table maps each sequence to its pages in order. Tests scatter
pages in random order across the pool and fill unused pages with 1e4, so a
kernel that reads a page it should not produces a visibly wrong answer.

### Triton: split-KV with GQA packing

Grid: `(batch, num_kv_heads, num_splits)`.

1. **GQA packing.** A program owns one KV head and all `G` query heads that
   share it. Each K/V page is loaded once and used by all `G` heads; one program
   per query head would load it `G` times. The group is padded to 16 rows so
   `Q·Kᵀ` and `P·V` both run as `tl.dot` on tensor cores.
2. **Split-KV (flash-decoding).** At batch 1 with 8 KV heads there are only 8
   (sequence, head) pairs, while an H100 has 132 SMs. Splitting each sequence's
   pages into chunks multiplies the parallelism. `choose_num_splits` aims for
   two programs per SM.
3. **Online softmax** in base 2 (`exp2`, with log2(e) folded into the scale).
   Each chunk keeps a running max, normaliser and output, and writes its
   normalised output plus its log-sum-exp.
4. **Exact merge.** A second kernel combines chunks with
   `out = Σ 2^(lse_s − M)·o_s / Σ 2^(lse_s − M)`. Chunks past the end of a short
   sequence write `lse = −∞` and get zero weight. Tests check that results are
   identical for 1, 2, 3, 5 and 64 splits.
5. **No host sync.** Work per sequence is bounded by the block table's width (or
   `max_seq_len`), not by reading `seq_lens` back to the CPU every step.

### CUDA: warp per token (a baseline, on purpose)

Grid: `(batch, num_kv_heads)`, 4 warps per block. Each warp takes every 4th
token. Its 32 lanes each load a contiguous slice of K with one vector load,
compute partial dot products for every query head in the group, finish them with
a shuffle reduction, and update an online softmax for their slice of V. The 4
warps merge through shared memory at the end.

It is deliberately simpler than the Triton kernel, which makes the gap
measurable:

| Missing | Why it matters | Where it shows |
|---|---|---|
| Split-KV | Only `batch × kv_heads` blocks, so 8 blocks on 132 SMs at batch 1 | Small batches |
| Tensor cores | Dot products run on CUDA cores, one token per warp at a time | Large GQA groups |
| Async copies (`cp.async` / TMA) and pipelining | Each warp waits on every K/V load | Every shape |

Compiler report (`nvcc -Xptxas -v`, sm_80/sm_90): RMSNorm uses at most 32
registers and never spills. For attention, 68 of 72 instantiations are
spill-free. Four spill 4–12 bytes, mostly the fp32 variants the benchmarks do not
use. Fusing the score and softmax-update loops removed the rest.

### What FlashInfer does beyond this

FlashInfer's decode kernels add pipelined asynchronous loads of K/V into
shared memory, tensor-core GQA decode, a planning step that load-balances
uneven sequence lengths across SMs, and cascade/shared-prefix attention.
Load-balancing is the most interesting next step here. With uniform splits, one
long sequence in a batch of short ones sets the latency for the whole batch.
