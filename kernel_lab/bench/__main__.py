"""Benchmark the kernels against PyTorch baselines and report roofline efficiency.

    python -m kernel_lab.bench all --dtype float16 --out results/

Every implementation is checked against the PyTorch baseline before it is timed;
a wrong result is reported as an error, never as a speed. Timing uses
triton.testing.do_bench (median of many runs, with L2 flushed between runs so
data is fetched from DRAM as it would be in a real model step).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kernel_lab import cuda_ops
from kernel_lab.bench.roofline import GPUPeak, Work, decode_attention_work, lookup, rms_norm_work
from kernel_lab.kv_cache import random_decode_problem

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class Row:
    kernel: str
    shape: str
    impl: str
    us: float | None
    gbps: float | None
    pct_peak_bw: float | None
    speedup: float | None
    max_abs_err: float | None
    note: str = ""


def bench_ms(fn: Callable[[], object]) -> float:
    import triton.testing

    return float(triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median"))


def measure(
    kernel: str,
    shape: str,
    impls: dict[str, Callable[[], torch.Tensor]],
    work: Work,
    peak: GPUPeak | None,
    baseline: str,
    atol: float,
) -> list[Row]:
    expected = impls[baseline]()
    rows: list[Row] = []
    base_us = None
    for name, fn in impls.items():
        try:
            out = fn()
            err = (out.float() - expected.float()).abs().max().item()
            if err > atol:
                rows.append(Row(kernel, shape, name, None, None, None, None, err, "WRONG RESULT, not timed"))
                continue
            us = bench_ms(fn) * 1e3
        except Exception as exc:  # unsupported shape, missing library, OOM
            rows.append(Row(kernel, shape, name, None, None, None, None, None, f"skipped: {exc}"[:120]))
            continue
        gbps = work.bytes / (us * 1e-6) / 1e9
        if name == baseline:
            base_us = us
        rows.append(
            Row(
                kernel,
                shape,
                name,
                round(us, 2),
                round(gbps, 1),
                round(100 * gbps / peak.dram_gbps, 1) if peak else None,
                None,
                err,
            )
        )
    for row in rows:
        if row.us and base_us:
            row.speedup = round(base_us / row.us, 2)
    return rows


# ------------------------------------------------------------------ RMSNorm


def rmsnorm_suite(dtype: torch.dtype, peak: GPUPeak | None, compile_baseline: bool) -> list[Row]:
    from kernel_lab import triton_kernels

    elem = torch.tensor([], dtype=dtype).element_size()
    rows: list[Row] = []
    for hidden in (4096, 8192):
        for tokens in (1, 32, 256, 4096):
            x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
            r = torch.randn_like(x)
            w = torch.randn(hidden, device="cuda", dtype=dtype)

            def eager(x=x, r=r, w=w):
                s = x + r  # unfused: separate add kernel writes the sum, norm reads it back
                return torch.nn.functional.rms_norm(s, (s.shape[-1],), w, eps=1e-6)

            impls: dict[str, Callable[[], torch.Tensor]] = {"torch eager": eager}
            if compile_baseline:
                compiled = torch.compile(eager)
                impls["torch.compile"] = compiled
            impls["triton"] = lambda x=x, r=r, w=w: triton_kernels.fused_add_rms_norm(x, r, w)[0]
            if cuda_ops.available():
                impls["cuda"] = lambda x=x, r=r, w=w: cuda_ops.fused_add_rms_norm(x, r, w)[0]
            work = rms_norm_work(tokens, hidden, elem, fused_residual=True)
            atol = 2e-2 if dtype != torch.float32 else 1e-4
            rows += measure(
                "fused_add_rms_norm", f"{tokens}x{hidden}", impls, work, peak, "torch eager", atol
            )
    return rows


# ------------------------------------------------------- decode attention


def torch_paged_decode(q, k_cache, v_cache, block_table, seq_lens, scale):
    """What plain PyTorch does: gather pages into padded K/V, mask, call SDPA (with GQA)."""
    batch, _, head_dim = q.shape
    page_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    max_len = block_table.shape[1] * page_size
    k = k_cache[block_table.long()].reshape(batch, max_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v_cache[block_table.long()].reshape(batch, max_len, num_kv_heads, head_dim).transpose(1, 2)
    mask = torch.arange(max_len, device=q.device)[None, :] < seq_lens[:, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2), k, v, attn_mask=mask[:, None, None, :], scale=scale, enable_gqa=True
    )
    return out.squeeze(2)


def flashinfer_decode(q, kv, num_q_heads, num_kv_heads, head_dim):
    """FlashInfer's paged decode (written against its 0.2-series API). The result is
    checked against the PyTorch baseline before timing, so an API mismatch shows up
    as an error or 'skipped', never as a wrong speed."""
    import flashinfer

    page_size = kv.page_size
    lens = kv.seq_lens.tolist()
    pages = [(n + page_size - 1) // page_size for n in lens]
    indptr = torch.tensor([0, *torch.tensor(pages).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    indices = torch.cat([kv.block_table[b, : pages[b]] for b in range(len(lens))]).to(torch.int32)
    last = torch.tensor(
        [n - (p - 1) * page_size for n, p in zip(lens, pages, strict=True)], dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr,
        indices,
        last,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q.dtype,
        kv_data_type=kv.k_cache.dtype,
    )
    return lambda: wrapper.run(q, (kv.k_cache, kv.v_cache))


def decode_suite(dtype: torch.dtype, peak: GPUPeak | None) -> list[Row]:
    from kernel_lab import triton_kernels

    # Llama-3-8B attention: 32 query heads, 8 KV heads, head_dim 128, 16-token pages.
    hq, hkv, d, page = 32, 8, 128, 16
    elem = torch.tensor([], dtype=dtype).element_size()
    rows: list[Row] = []
    for batch in (1, 8, 32, 64):
        for ctx in (512, 4096):
            lens = [ctx] * batch
            q, kv, _, _ = random_decode_problem(lens, hq, hkv, d, page, dtype=dtype, device="cuda")
            args = (q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens)
            scale = d**-0.5
            impls: dict[str, Callable[[], torch.Tensor]] = {
                "torch (gather + SDPA)": lambda a=args, s=scale: torch_paged_decode(*a, s),
                "triton (split-KV)": lambda a=args, n=ctx: triton_kernels.paged_decode_attention(
                    *a, max_seq_len=n
                ),
            }
            if cuda_ops.available():
                impls["cuda (warp per token)"] = lambda a=args: cuda_ops.paged_decode_attention(*a)
            try:
                impls["flashinfer"] = flashinfer_decode(q, kv, hq, hkv, d)
            except Exception as exc:
                rows.append(
                    Row(
                        "paged_decode",
                        f"b{batch} ctx{ctx}",
                        "flashinfer",
                        None,
                        None,
                        None,
                        None,
                        None,
                        f"skipped: {type(exc).__name__}: {exc}"[:120],
                    )
                )
            work = decode_attention_work(lens, hq, hkv, d, elem)
            atol = 2e-2 if dtype != torch.float32 else 1e-3
            rows += measure(
                "paged_decode", f"b{batch} ctx{ctx}", impls, work, peak, "torch (gather + SDPA)", atol
            )
            del q, kv, args
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------- report


def to_markdown(rows: list[Row], header: str) -> str:
    lines = [
        header,
        "",
        "| Kernel | Shape | Implementation | Time (µs) | GB/s | % peak BW | Speedup | Max err |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        fmt = lambda v, suffix="": "-" if v is None else f"{v}{suffix}"  # noqa: E731
        err = "-" if r.max_abs_err is None else f"{r.max_abs_err:.1e}"
        impl = r.impl if not r.note else f"{r.impl} ({r.note})"
        lines.append(
            f"| {r.kernel} | {r.shape} | {impl} | {fmt(r.us)} | {fmt(r.gbps)} | {fmt(r.pct_peak_bw, '%')} "
            f"| {fmt(r.speedup, 'x')} | {err} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kernel_lab.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("suite", choices=["rmsnorm", "decode", "all"])
    parser.add_argument("--dtype", choices=list(DTYPES), default="float16")
    parser.add_argument("--peak-gbps", type=float, help="override the spec-sheet DRAM bandwidth")
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    parser.add_argument("--out", type=Path, help="directory for results.json and results.md")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("No CUDA device. Benchmarks need an NVIDIA GPU; see docs/gpu-runbook.md.", file=sys.stderr)
        return 2

    name = torch.cuda.get_device_name()
    peak = lookup(name)
    if args.peak_gbps:
        peak = GPUPeak(name, args.peak_gbps, peak.fp16_tflops if peak else 0.0)
    dtype = DTYPES[args.dtype]
    print(
        f"GPU: {name}; peak DRAM: {peak.dram_gbps if peak else 'unknown'} GB/s; dtype: {args.dtype}",
        file=sys.stderr,
    )
    if not cuda_ops.available():
        print(f"CUDA kernels not benchmarked: {cuda_ops.why_unavailable()}", file=sys.stderr)

    rows: list[Row] = []
    if args.suite in ("rmsnorm", "all"):
        rows += rmsnorm_suite(dtype, peak, compile_baseline=not args.no_compile)
    if args.suite in ("decode", "all"):
        rows += decode_suite(dtype, peak)

    header = (
        f"### {name}, {args.dtype}\n\nPeak DRAM bandwidth used for %: "
        f"{peak.dram_gbps if peak else 'unknown'} GB/s. Speedup is vs the first PyTorch baseline per shape."
    )
    table = to_markdown(rows, header)
    print(table)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        meta = {
            "gpu": name,
            "dtype": args.dtype,
            "torch": torch.__version__,
            "peak_dram_gbps": peak.dram_gbps if peak else None,
        }
        (args.out / "results.json").write_text(
            json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2)
        )
        (args.out / "results.md").write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
