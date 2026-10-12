"""Paged-KV decode attention in Triton, using split-KV ("flash-decoding").

During decode each sequence adds one query token and attends over its whole KV
cache. Per KV element there are only a few FLOPs, so the kernel is bound by how
fast it can stream K and V from DRAM. Three choices follow from that:

1. **GQA packing.** One program handles one KV head and *all* query heads that
   share it (``GROUP`` = num_q_heads / num_kv_heads). Each K/V tile is loaded
   once and reused by the whole group, instead of once per query head. The
   group is padded to 16 rows so both products run as ``tl.dot`` on tensor cores.

2. **Split-KV.** With small batches, one program per (sequence, KV head) leaves
   most SMs idle. The cache is split along the sequence into ``num_splits``
   chunks processed in parallel. Each writes a partial output and its
   log-sum-exp, and a small reduce kernel merges them exactly.

3. **Online softmax.** Each chunk keeps a running max ``m``, normaliser ``l`` and
   unnormalised output ``acc``, rescaling them as new pages arrive, so scores
   never need to be materialised. It uses base-2 exponentials (``exp2``), with
   log2(e) folded into the scale.

Pages are followed through the block table, so the cache can be scattered
anywhere in the page pool, as in vLLM and FlashInfer.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _decode_split_kernel(
    Q,
    K_CACHE,
    V_CACHE,
    BLOCK_TABLE,
    SEQ_LENS,
    O_PART,
    LSE_PART,
    stride_qb,
    stride_qh,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_bt,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    scale_log2,
    pages_per_split,
    GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(SEQ_LENS + b)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    page_start = split * pages_per_split
    page_end = tl.minimum(page_start + pages_per_split, num_pages)

    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, HEAD_DIM)
    t = tl.arange(0, PAGE_SIZE)
    g_mask = g < GROUP
    q_heads = kv_head * GROUP + g

    q = tl.load(
        Q + b * stride_qb + q_heads[:, None] * stride_qh + d[None, :],
        mask=g_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], dtype=tl.float32)

    for page_i in range(page_start, page_end):
        page = tl.load(BLOCK_TABLE + b * stride_bt + page_i).to(tl.int64)
        token = page_i * PAGE_SIZE + t
        t_mask = token < seq_len

        k = tl.load(
            K_CACHE + page * stride_kp + t[:, None] * stride_kt + kv_head * stride_kh + d[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale_log2  # [BLOCK_G, PAGE_SIZE]
        scores = tl.where(t_mask[None, :], scores, float("-inf"))

        # Every page in range holds at least one real token, so m_new is finite.
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V_CACHE + page * stride_vp + t[:, None] * stride_vt + kv_head * stride_vh + d[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # A split past the end of a short sequence did no work: mark it with lse = -inf
    # so the reduce gives it zero weight.
    has_work = l_i > 0.0
    safe_l = tl.where(has_work, l_i, 1.0)
    out = acc / safe_l[:, None]
    lse = tl.where(has_work, m_i + tl.log2(safe_l), float("-inf"))

    tl.store(
        O_PART + b * stride_ob + q_heads[:, None] * stride_oh + split * stride_os + d[None, :],
        out,
        mask=g_mask[:, None],
    )
    tl.store(LSE_PART + b * stride_lb + q_heads * stride_lh + split, lse, mask=g_mask)


@triton.jit
def _merge_splits_kernel(
    O_PART,
    LSE_PART,
    OUT,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    stride_outb,
    stride_outh,
    num_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """out = sum_s 2^(lse_s - M) * o_s / sum_s 2^(lse_s - M): the exact softmax over
    the whole sequence, rebuilt from per-chunk results."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, HEAD_DIM)
    s_mask = s < num_splits

    lse = tl.load(LSE_PART + b * stride_lb + h * stride_lh + s, mask=s_mask, other=float("-inf"))
    weights = tl.exp2(lse - tl.max(lse, axis=0))
    o = tl.load(
        O_PART + b * stride_ob + h * stride_oh + s[:, None] * stride_os + d[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    out = tl.sum(o * weights[:, None], axis=0) / tl.sum(weights, axis=0)
    tl.store(OUT + b * stride_outb + h * stride_outh + d, out.to(OUT.dtype.element_ty))


def choose_num_splits(batch: int, num_kv_heads: int, max_pages: int, device: torch.device) -> int:
    """Enough splits to give every SM about two programs, but never more than pages."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else 32
    wanted = math.ceil(2 * sms / max(batch * num_kv_heads, 1))
    return max(1, min(wanted, max_pages, 64))


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    num_splits: int | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Same contract as ``reference.paged_decode_attention``.

    ``max_seq_len`` bounds the work per sequence. Without it, the block table's
    width is used, which avoids reading ``seq_lens`` back to the host (a device
    sync) on every decode step.
    """
    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, head_dim_kv = k_cache.shape
    if head_dim_kv != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("q, k_cache and v_cache disagree on shape")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    for name, value in (("head_dim", head_dim), ("page_size", page_size)):
        if value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two >= 16 (tensor-core tile), got {value}")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("the head dimension must be contiguous")

    group = num_q_heads // num_kv_heads
    max_pages = block_table.shape[1]
    if max_seq_len is not None:
        max_pages = min(max_pages, triton.cdiv(max_seq_len, page_size))
    if num_splits is None:
        num_splits = choose_num_splits(batch, num_kv_heads, max_pages, q.device)
    pages_per_split = triton.cdiv(max_pages, max(1, num_splits))
    num_splits = triton.cdiv(max_pages, pages_per_split)  # drop splits no sequence can reach

    o_part = torch.empty(batch, num_q_heads, num_splits, head_dim, dtype=torch.float32, device=q.device)
    lse_part = torch.empty(batch, num_q_heads, num_splits, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    scale = head_dim**-0.5 if scale is None else scale

    _decode_split_kernel[(batch, num_kv_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        o_part,
        lse_part,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        o_part.stride(0),
        o_part.stride(1),
        o_part.stride(2),
        lse_part.stride(0),
        lse_part.stride(1),
        scale * LOG2E,
        pages_per_split,
        GROUP=group,
        BLOCK_G=max(16, triton.next_power_of_2(group)),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        num_warps=4,
    )
    _merge_splits_kernel[(batch, num_q_heads)](
        o_part,
        lse_part,
        out,
        o_part.stride(0),
        o_part.stride(1),
        o_part.stride(2),
        lse_part.stride(0),
        lse_part.stride(1),
        out.stride(0),
        out.stride(1),
        num_splits,
        HEAD_DIM=head_dim,
        BLOCK_S=max(triton.next_power_of_2(num_splits), 2),
        num_warps=4,
    )
    return out
