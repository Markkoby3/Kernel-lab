from __future__ import annotations

import pytest
import torch

from kernel_lab import reference
from kernel_lab.kv_cache import build_paged_kv, random_decode_problem
from kernel_lab.triton_kernels import paged_decode_attention
from tests.conftest import DTYPES, INTERPRETED, tolerance

# (num_q_heads, num_kv_heads): multi-head, grouped-query, multi-query.
HEADS = [(4, 4), (8, 2), (8, 1)]
# Lengths at 1, page boundaries (16, 32) and just past them (17, 33).
SEQ_LENS = [1, 15, 16, 17, 33, 70]


def run(q, kv, **kwargs):
    return paged_decode_attention(q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens, **kwargs)


def expected(q, kv):
    return reference.paged_decode_attention(q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("hq", "hkv"), HEADS)
def test_matches_reference(hq, hkv, dtype, device):
    q, kv, _, _ = random_decode_problem(SEQ_LENS, hq, hkv, 32, 16, dtype=dtype, device=device)
    torch.testing.assert_close(run(q, kv), expected(q, kv), **tolerance(dtype))


@pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 64])
def test_split_kv_is_exact_for_any_split_count(num_splits, device):
    """Splits only change how the sum is partitioned, never the result. Many splits
    leave some empty for short sequences, which the merge must weight at zero."""
    q, kv, _, _ = random_decode_problem(SEQ_LENS, 4, 2, 32, 16, dtype=torch.float32, device=device)
    torch.testing.assert_close(run(q, kv, num_splits=num_splits), expected(q, kv), atol=1e-5, rtol=1e-5)


def test_agrees_with_dense_sdpa(device):
    q, kv, keys, values = random_decode_problem([5, 40], 8, 2, 64, 16, dtype=torch.float32, device=device)
    torch.testing.assert_close(
        run(q, kv), reference.dense_decode_attention(q, keys, values), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_page_and_head_sizes(page_size, head_dim, device):
    q, kv, _, _ = random_decode_problem(
        [3, 50], 4, 2, head_dim, page_size, dtype=torch.float32, device=device
    )
    torch.testing.assert_close(run(q, kv), expected(q, kv), atol=1e-5, rtol=1e-5)


def test_reads_only_pages_in_the_block_table(device):
    """build_paged_kv fills unused pages with 1e4. Touching one would blow up the
    softmax, so passing this means the kernel only followed the block table."""
    q, kv, _, _ = random_decode_problem([20, 9], 4, 4, 32, 16, dtype=torch.float32, device=device)
    out = run(q, kv)
    assert out.abs().max() < 10
    torch.testing.assert_close(out, expected(q, kv), atol=1e-5, rtol=1e-5)


def test_max_seq_len_bounds_work_without_changing_results(device):
    q, kv, _, _ = random_decode_problem([5, 20], 4, 2, 32, 16, dtype=torch.float32, device=device)
    padded = torch.nn.functional.pad(kv.block_table, (0, 6))  # extra block-table columns
    kv.block_table = padded
    torch.testing.assert_close(run(q, kv, max_seq_len=20), expected(q, kv), atol=1e-5, rtol=1e-5)


def test_custom_scale(device):
    q, kv, _, _ = random_decode_problem([12], 2, 2, 32, 16, dtype=torch.float32, device=device)
    got = run(q, kv, scale=0.05)
    want = reference.paged_decode_attention(
        q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens, scale=0.05
    )
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_one_hot_attention_returns_that_value(device):
    """With one key far more aligned than the rest, attention picks its value."""
    head_dim = 32
    keys = [torch.zeros(20, 1, head_dim)]
    values = [torch.arange(20, dtype=torch.float32)[:, None, None].expand(20, 1, head_dim).clone()]
    keys[0][13, 0, 0] = 100.0
    kv = build_paged_kv([k.to(device) for k in keys], [v.to(device) for v in values], page_size=16)
    q = torch.zeros(1, 1, head_dim, device=device)
    q[0, 0, 0] = 1.0
    out = run(q, kv, scale=1.0)
    torch.testing.assert_close(out, torch.full_like(out, 13.0), atol=1e-4, rtol=0)


def test_validation(device):
    q, kv, _, _ = random_decode_problem([4], 3, 2, 32, 16, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="multiple"):
        run(q, kv)
    q, kv, _, _ = random_decode_problem([4], 2, 2, 24, 16, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="power of two"):
        run(q, kv)


@pytest.mark.skipif(INTERPRETED, reason="long sequences are slow in the CPU interpreter")
def test_long_context_on_gpu(device):
    q, kv, _, _ = random_decode_problem([4096, 31, 2000], 32, 8, 128, 16, dtype=torch.float16, device=device)
    torch.testing.assert_close(run(q, kv), expected(q, kv), **tolerance(torch.float16))
