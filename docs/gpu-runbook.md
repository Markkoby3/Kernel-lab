# GPU runbook: correctness and benchmarks on a rented GPU

About 45 minutes of GPU time. The CPU tests prove the Triton kernels are
correct. This run proves the CUDA kernels are correct and measures everything.

## 1. Machine

Any NVIDIA GPU with compute capability 8.0 or newer (A100, A10G, L4, L40S,
H100, RTX 30/40 series) and a CUDA toolkit. Lambda, RunPod and Vast.ai images
with PyTorch preinstalled all work. An H100 or A100 gives the most comparable
numbers.

```bash
nvidia-smi
nvcc --version          # the CUDA toolkit; needed to build the CUDA kernels
```

## 2. Install

```bash
git clone https://github.com/Markkoby3/kernel-lab && cd kernel-lab
python -m venv .venv && . .venv/bin/activate
pip install torch triton numpy pytest     # or reuse the image's PyTorch
pip install -e . --no-build-isolation     # builds kernel_lab._C for this GPU
python -c "from kernel_lab import cuda_ops; print(cuda_ops.available())"   # True
```

## 3. Correctness

```bash
pytest -q
```

On a GPU the tests run natively, not in the interpreter. That adds bfloat16
coverage, the 4,096-token long-context case, every CUDA kernel variant, and
torch.compile compatibility. All tests must pass before any benchmark counts.

## 4. Benchmarks

```bash
python -m kernel_lab.bench all --dtype float16 --out results/$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_')
python -m kernel_lab.bench all --dtype bfloat16 --no-compile --out results/bf16
```

Optional, to compare against the state of the art:

```bash
pip install flashinfer-python    # match your torch/CUDA version, see flashinfer docs
python -m kernel_lab.bench decode --dtype float16 --out results/with-flashinfer
```

Every implementation is checked against the PyTorch baseline before it is
timed. A row marked "WRONG RESULT" means a bug, not a slow kernel.

## 5. Profile one kernel

```bash
ncu --set full -k regex:_decode_split_kernel -c 1 -o triton_decode \
  python -m kernel_lab.bench decode --no-compile
ncu --set full -k regex:paged_decode_kernel -c 1 -o cuda_decode \
  python -m kernel_lab.bench decode --no-compile
```

In Nsight Compute, check **DRAM throughput (% of peak)**, **achieved
occupancy**, and **warp stall reasons**. For the CUDA baseline, expect "long
scoreboard" stalls (waiting on loads), the cost of having no pipelining.

## 6. Publish

Copy `results/*/results.md` into the README's Results section with the GPU
name and date, commit the JSON files under `results/`, and shut the machine down.

## Sanity checks

- Achieved bandwidth above ~92% of peak means the byte count is wrong, or data
  came from L2 instead of DRAM. Re-check before publishing.
- The PyTorch baseline gathers pages into a padded tensor before SDPA, so it
  moves more bytes than the model counts. Its "% of peak" understates its raw
  memory throughput. Compare speedups, not baseline bandwidth.
- Run each suite twice. More than ~3% run-to-run difference means something
  else is using the GPU.
