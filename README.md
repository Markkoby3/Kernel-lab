# kernel-lab

[![CI](https://github.com/Markkoby3/kernel-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Markkoby3/kernel-lab/actions/workflows/ci.yml)

LLM inference kernels written twice, in **Triton** and in **CUDA C++**, each
checked against a PyTorch reference and measured against the roofline:

| Kernel | Triton | CUDA | Used in |
|---|---|---|---|
| Fused residual-add + RMSNorm | One program per row, whole row in registers | 16-byte vector loads, warp-shuffle reduction | Every pre-norm transformer layer |
| Paged-KV decode attention | Split-KV (flash-decoding), GQA packing, `tl.dot` on tensor cores, exact log-sum-exp merge | Warp-per-token baseline, online softmax, shared-memory merge | Every decode step in vLLM, SGLang and FlashInfer |

Both kernels are memory-bound (decode attention does ~4 FLOPs per byte of KV
cache, far below any GPU's ridge point), so the benchmark scores them as a
percentage of peak DRAM bandwidth. [Design notes](docs/design.md) explain each
choice and what FlashInfer does beyond it.

## Correctness without a GPU

Triton kernels run on the CPU through Triton's interpreter, so the full
correctness suite runs anywhere, including CI:

```bash
pip install torch triton numpy pytest
KERNEL_LAB_NO_CUDA=1 pip install -e . --no-build-isolation
pytest -q          # sets TRITON_INTERPRET=1 automatically when there is no GPU
```

The tests cover:

- **Model shapes:** multi-head, grouped-query and multi-query attention, plus fp32 and fp16 (bf16 on a GPU).
- **Edge cases:** sequence lengths at and around page boundaries, and 1 to 64 KV splits, including splits with no work.
- **Exactness:** the fused RMSNorm's residual output must match bit for bit.
- **Paging:** shuffled pages, with poisoned unused pages that fail the test if a kernel reads one.

The PyTorch references are themselves checked against independent paths: SDPA,
and PyTorch's built-in `rms_norm`.

The CUDA kernels compile in CI with `-Werror` for sm_80, sm_86, sm_89 and sm_90,
and their GPU tests run as part of the [GPU runbook](docs/gpu-runbook.md).

## Build the CUDA kernels

On a machine with an NVIDIA GPU and a CUDA toolkit:

```bash
pip install -e . --no-build-isolation        # builds kernel_lab._C
python -c "import torch, kernel_lab.cuda_ops; print(torch.ops.kernel_lab.paged_decode_attention)"
```

The kernels are registered with `TORCH_LIBRARY` as `torch.ops.kernel_lab.*`, so
they work inside `torch.compile(fullgraph=True)` graphs (tested). The kernel
sources in `csrc/` have no PyTorch dependency; `csrc/torch_bindings.cpp` is the
only file that includes PyTorch.

## Benchmarks

```bash
python -m kernel_lab.bench all --dtype float16 --out results/
```

The suites:

- **Fused add + RMSNorm:** 1 to 4,096 tokens at hidden sizes 4,096 and 8,192. It is compared with unfused PyTorch eager and `torch.compile`.
- **Paged decode attention:** Llama-3-8B attention (32 query heads, 8 KV heads, head dim 128, 16-token pages) at batch 1 to 64 and contexts of 512 and 4,096. It is compared with PyTorch (gather + SDPA) and FlashInfer when installed.

Each row reports time, achieved GB/s, **% of peak DRAM bandwidth**, speedup and
max error. Each implementation's output is checked before it is timed.

### Results

> **Pending a GPU run.** Numbers will be published here with the GPU, driver and
> date, and raw JSON in `results/`. To reproduce, follow the [GPU runbook](docs/gpu-runbook.md).

## Layout

```
kernel_lab/
  reference.py              PyTorch ground truth (fp32 math, model-faithful rounding)
  kv_cache.py               Paged KV cache builder: shuffled pages, poisoned spares
  triton_kernels/           rmsnorm.py, paged_attention.py
  cuda_ops.py               Python wrappers for torch.ops.kernel_lab.*
  bench/                    Benchmark CLI and roofline model (bytes, FLOPs, GPU peaks)
csrc/
  common.cuh                dtype conversion, vector packs, warp and block reductions
  rmsnorm.cu                Fused add + RMSNorm
  paged_attention.cu        Paged decode attention
  torch_bindings.cpp        TORCH_LIBRARY registration
tests/                      CPU (Triton interpreter) and GPU tests
docs/                       Design notes, GPU runbook
```

## Roadmap

- [ ] Publish H100 and A100 results, with Nsight Compute profiles
- [ ] CUDA decode kernel with split-KV and `cp.async` double buffering
- [ ] Load-balanced split planning for uneven sequence lengths
- [ ] FP8 KV cache with per-head scales

## License

MIT
