"""CUDA kernels vs the references. Skipped unless the extension is built and a GPU is present."""

from __future__ import annotations

import pytest
import torch

from kernel_lab import cuda_ops, reference
from kernel_lab.kv_cache import random_decode_problem
from tests.conftest import tolerance

pytestmark = pytest.mark.skipif(not cuda_ops.available(), reason=cuda_ops.why_unavailable())

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 64), (3, 100), (8, 4096), (2, 5, 8192)])
def test_rms_norm(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    torch.testing.assert_close(cuda_ops.rms_norm(x, w), reference.rms_norm(x, w), **tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 100), (16, 4096)])
def test_fused_add_rms_norm(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    r = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    out, res = cuda_ops.fused_add_rms_norm(x, r, w)
    ref_out, ref_res = reference.fused_add_rms_norm(x, r, w)
    torch.testing.assert_close(res, ref_res, atol=0, rtol=0)
    torch.testing.assert_close(out, ref_out, **tolerance(dtype))


def test_rms_norm_unaligned_rows_take_scalar_path():
    base = torch.randn(4, 130, device="cuda", dtype=torch.float16)
    x = base[:, 1:101]  # misaligned start and odd width: no 16-byte vector loads
    w = torch.randn(100, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(cuda_ops.rms_norm(x, w), reference.rms_norm(x, w), **tolerance(torch.float16))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("hq", "hkv"), [(8, 8), (16, 4), (8, 1), (32, 8)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_paged_decode_attention(hq, hkv, head_dim, dtype):
    q, kv, _, _ = random_decode_problem(
        [1, 3, 16, 17, 300, 1025], hq, hkv, head_dim, 16, dtype=dtype, device="cuda"
    )
    got = cuda_ops.paged_decode_attention(q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens)
    want = reference.paged_decode_attention(q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens)
    torch.testing.assert_close(got, want, **tolerance(dtype))


def test_ops_work_under_torch_compile():
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    w = torch.randn(256, device="cuda", dtype=torch.float16)

    @torch.compile(fullgraph=True)
    def f(x, w):
        out = torch.empty_like(x)
        torch.ops.kernel_lab.rms_norm(out, x, w, 1e-6)
        return out * 2

    torch.testing.assert_close(f(x, w), reference.rms_norm(x, w) * 2, **tolerance(torch.float16))
