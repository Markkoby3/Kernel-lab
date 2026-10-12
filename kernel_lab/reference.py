"""PyTorch reference implementations: the ground truth every kernel is tested against.

They favour clarity over speed, compute in float32, and round to the input dtype
only where a real model would store a tensor (the residual stream and the
output), so the kernels and the references round at the same points.
"""

from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """y = x / sqrt(mean(x^2) + eps) * weight, over the last dimension."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-norm transformer step: residual += x, then normalize the new residual.

    Returns (normalized output, updated residual). The sum is rounded to the
    activation dtype before normalizing, because that rounded value is what the
    residual stream stores and what the next layer reads.
    """
    new_residual = (x.float() + residual.float()).to(x.dtype)
    return rms_norm(new_residual, weight, eps), new_residual


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Single-token decode attention over a paged KV cache.

    Shapes (NHD layout, as in FlashInfer):
        q            [batch, num_q_heads, head_dim]       one new token per sequence
        k/v_cache    [num_pages, page_size, num_kv_heads, head_dim]
        block_table  [batch, max_pages] int32           page ids of each sequence, in order
        seq_lens     [batch] int32                       tokens in each sequence's cache

    num_q_heads must be a multiple of num_kv_heads: grouped-query attention, where
    each KV head serves num_q_heads // num_kv_heads query heads.
    """
    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    group = num_q_heads // num_kv_heads
    scale = head_dim**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_table[b, : (length + page_size - 1) // page_size].long()
        k = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:length].float()
        v = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:length].float()
        # [L, Hkv, D] -> [Hq, L, D]: each KV head repeated for its query group.
        k = k.permute(1, 0, 2).repeat_interleave(group, dim=0)
        v = v.permute(1, 0, 2).repeat_interleave(group, dim=0)
        scores = torch.einsum("hd,hld->hl", q[b].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hl,hld->hd", probs, v).to(q.dtype)
    return out


def dense_decode_attention(
    q: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor], scale: float | None = None
) -> torch.Tensor:
    """Same computation on contiguous per-sequence K/V ([L, Hkv, D] each), via SDPA.

    An independent path used to check that the paging helpers and the paged
    reference agree with a standard attention implementation.
    """
    num_q_heads = q.shape[1]
    out = []
    for b, (k, v) in enumerate(zip(keys, values, strict=True)):
        group = num_q_heads // k.shape[1]
        kk = k.permute(1, 0, 2).repeat_interleave(group, dim=0).float()
        vv = v.permute(1, 0, 2).repeat_interleave(group, dim=0).float()
        o = torch.nn.functional.scaled_dot_product_attention(q[b].float().unsqueeze(1), kk, vv, scale=scale)
        out.append(o.squeeze(1).to(q.dtype))
    return torch.stack(out)
