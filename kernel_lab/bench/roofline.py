"""Roofline bookkeeping: bytes and FLOPs each kernel must move or compute, and GPU peaks.

The model counts the minimum DRAM traffic a perfect kernel would need: every
input read once, every output written once. Achieved bandwidth = bytes / time.
For a memory-bound kernel, "% of peak bandwidth" is the honest efficiency
number; a FLOP rate would look tiny and say nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUPeak:
    name: str
    dram_gbps: float  # spec-sheet DRAM bandwidth, GB/s
    fp16_tflops: float  # dense fp16/bf16 tensor-core throughput, fp32 accumulate

    @property
    def ridge_flops_per_byte(self) -> float:
        """Arithmetic intensity above which a kernel is compute-bound."""
        return self.fp16_tflops * 1e12 / (self.dram_gbps * 1e9)


# Spec-sheet numbers (dense, no sparsity). Real achievable bandwidth is typically
# 85-92% of these; pass --peak-gbps to use a measured value instead.
GPU_PEAKS = [
    # Names as torch.cuda.get_device_name() reports them; most specific first.
    GPUPeak("H100 80GB HBM3", 3350, 989),  # H100 SXM
    GPUPeak("H100 NVL", 3900, 835),
    GPUPeak("H100 PCIe", 2000, 756),
    GPUPeak("A100-SXM4-80GB", 2039, 312),
    GPUPeak("A100 80GB PCIe", 1935, 312),
    GPUPeak("A100", 1555, 312),  # 40GB SXM and PCIe
    GPUPeak("L40S", 864, 362),
    GPUPeak("L40", 864, 181),
    GPUPeak("L4", 300, 121),
    GPUPeak("A10G", 600, 70),
    GPUPeak("RTX 4090", 1008, 165),
]


def _words(name: str) -> set[str]:
    return set(name.lower().replace("-", " ").split())


def lookup(device_name: str) -> GPUPeak | None:
    """Match torch.cuda.get_device_name() against the table by whole words, so
    'L4' never matches an L40 and 'A100' matches only when nothing more specific does."""
    words = _words(device_name)
    for peak in GPU_PEAKS:
        if _words(peak.name) <= words:
            return peak
    return None


@dataclass(frozen=True)
class Work:
    bytes: float
    flops: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes


def rms_norm_work(rows: int, hidden: int, elem_bytes: int, fused_residual: bool) -> Work:
    # Read x, write out; fused also reads the residual and writes the new one.
    tensors = 4 if fused_residual else 2
    data = rows * hidden * elem_bytes * tensors + hidden * elem_bytes  # + weight
    flops = rows * hidden * (5 if fused_residual else 4)  # add, square, sum, scale, weight
    return Work(bytes=data, flops=flops)


def decode_attention_work(
    seq_lens: list[int], num_q_heads: int, num_kv_heads: int, head_dim: int, elem_bytes: int
) -> Work:
    tokens = sum(seq_lens)
    kv = 2 * tokens * num_kv_heads * head_dim * elem_bytes  # K and V, each read once
    q_out = 2 * len(seq_lens) * num_q_heads * head_dim * elem_bytes
    # Per query head and cached token: QK^T and PV each cost 2*head_dim FLOPs.
    flops = 4 * tokens * num_q_heads * head_dim
    return Work(bytes=kv + q_out, flops=flops)
