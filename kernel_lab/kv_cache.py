"""Build paged KV caches for tests and benchmarks.

A serving engine stores each sequence's keys and values in fixed-size pages
scattered across one big pool, and a block table maps each sequence to its
pages. Allocating pages in a random order here makes sure kernels follow the
block table instead of accidentally relying on pages being contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedKV:
    k_cache: torch.Tensor  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor
    block_table: torch.Tensor  # [batch, max_pages] int32, padded with 0
    seq_lens: torch.Tensor  # [batch] int32
    page_size: int

    @property
    def max_pages(self) -> int:
        return self.block_table.shape[1]


def pages_for(seq_len: int, page_size: int) -> int:
    return (seq_len + page_size - 1) // page_size


def build_paged_kv(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    page_size: int,
    extra_pages: int = 3,
    seed: int = 0,
) -> PagedKV:
    """Scatter contiguous per-sequence K/V ([L, Hkv, D] each) into shuffled pages.

    ``extra_pages`` unused pages are filled with large values: if a kernel reads a
    page it should not, the result is visibly wrong rather than accidentally right.
    """
    if not keys or len(keys) != len(values):
        raise ValueError("need one K and one V tensor per sequence")
    _, num_kv_heads, head_dim = keys[0].shape
    lens = [k.shape[0] for k in keys]
    if min(lens) < 1:
        raise ValueError("every sequence needs at least one cached token")
    needed = [pages_for(n, page_size) for n in lens]
    total = sum(needed) + extra_pages
    device, dtype = keys[0].device, keys[0].dtype

    k_cache = torch.full((total, page_size, num_kv_heads, head_dim), 1e4, dtype=dtype, device=device)
    v_cache = torch.full_like(k_cache, 1e4)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator).tolist()

    block_table = torch.zeros((len(keys), max(needed)), dtype=torch.int32)
    cursor = 0
    for b, (k, v) in enumerate(zip(keys, values, strict=True)):
        for p in range(needed[b]):
            page = order[cursor]
            cursor += 1
            block_table[b, p] = page
            chunk = slice(p * page_size, min((p + 1) * page_size, lens[b]))
            n = chunk.stop - chunk.start
            k_cache[page, :n] = k[chunk]
            v_cache[page, :n] = v[chunk]
            k_cache[page, n:] = 0  # tail of a partly filled page: valid memory, never attended
            v_cache[page, n:] = 0

    return PagedKV(
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table.to(device),
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        page_size=page_size,
    )


def random_decode_problem(
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, PagedKV, list[torch.Tensor], list[torch.Tensor]]:
    """A random query batch and its paged cache, plus the contiguous K/V it came from."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    q = rand(len(seq_lens), num_q_heads, head_dim)
    keys = [rand(n, num_kv_heads, head_dim) for n in seq_lens]
    values = [rand(n, num_kv_heads, head_dim) for n in seq_lens]
    return q, build_paged_kv(keys, values, page_size, seed=seed), keys, values
