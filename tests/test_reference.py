"""The references are the ground truth, so they are checked against independent math."""

from __future__ import annotations

import pytest
import torch

from kernel_lab import reference
from kernel_lab.kv_cache import build_paged_kv, pages_for, random_decode_problem


def test_rms_norm_unit_rms_output():
    x = torch.randn(8, 512, dtype=torch.float64)
    y = reference.rms_norm(x, torch.ones(512, dtype=torch.float64), eps=0.0)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(8, dtype=torch.float64))


def test_rms_norm_matches_torch_builtin():
    x = torch.randn(4, 64)
    w = torch.randn(64)
    torch.testing.assert_close(
        reference.rms_norm(x, w, eps=1e-5),
        torch.nn.functional.rms_norm(x, (64,), w, eps=1e-5),
    )


@pytest.mark.parametrize(("hq", "hkv"), [(4, 4), (8, 2), (8, 1)])
def test_paged_reference_matches_dense_sdpa(hq, hkv):
    q, kv, keys, values = random_decode_problem([1, 16, 17, 50], hq, hkv, 32, 16, dtype=torch.float32)
    torch.testing.assert_close(
        reference.paged_decode_attention(q, kv.k_cache, kv.v_cache, kv.block_table, kv.seq_lens),
        reference.dense_decode_attention(q, keys, values),
        atol=1e-5,
        rtol=1e-5,
    )


def test_build_paged_kv_round_trips_and_shuffles_pages():
    keys = [torch.randn(n, 2, 8) for n in (5, 33)]
    values = [torch.randn(n, 2, 8) for n in (5, 33)]
    kv = build_paged_kv(keys, values, page_size=16)
    assert kv.block_table.shape == (2, pages_for(33, 16))
    for b, k in enumerate(keys):
        n = k.shape[0]
        pages = kv.block_table[b, : pages_for(n, 16)].long()
        torch.testing.assert_close(kv.k_cache[pages].reshape(-1, 2, 8)[:n], k)
    assert kv.block_table[1].tolist() != sorted(kv.block_table[1].tolist())


def test_build_paged_kv_rejects_empty_sequences():
    with pytest.raises(ValueError):
        build_paged_kv([torch.randn(0, 1, 8)], [torch.randn(0, 1, 8)], page_size=16)
