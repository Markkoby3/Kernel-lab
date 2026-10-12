"""RMSNorm and fused residual-add + RMSNorm in Triton.

RMSNorm does about 4 FLOPs per element and moves 2-4 bytes per element per
tensor, so it is purely memory-bound: speed is bytes moved / DRAM bandwidth.
The fused version earns its keep by touching memory once. Unfused, a model
reads x and residual, writes the sum, then reads the sum again to normalize.
Fused, each row is loaded once, summed and normalized in registers, and both
outputs are written, saving one full read of the hidden state per layer.

One program handles one row. A whole row (hidden size up to 16K) fits in
registers as a single block, so the sum of squares is one in-register reduction
with no second pass over global memory.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 16384


@triton.jit
def _rms_norm_kernel(
    X,
    RES,
    RES_OUT,
    W,
    Y,
    stride_x,
    stride_res,
    stride_y,
    n_cols,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(RES + row * stride_res + cols, mask=mask, other=0.0)
        # Round the sum to the activation dtype, as the residual stream stores it,
        # and normalize that stored value so both outputs agree.
        x = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
        tl.store(RES_OUT + row * stride_res + cols, x, mask=mask)

    xf = x.to(tl.float32)
    mean_sq = tl.sum(xf * xf, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = xf * rstd * w
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"hidden size {x.shape[-1]} does not match weight {weight.shape[0]}")
    hidden = x.shape[-1]
    if hidden > MAX_FUSED_SIZE:
        raise ValueError(f"hidden size {hidden} exceeds the single-block limit {MAX_FUSED_SIZE}")

    x2 = x.reshape(-1, hidden)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty((x2.shape[0], hidden), dtype=x.dtype, device=x.device)

    res2 = res_out = None
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError("residual must have the same shape as x")
        res2 = residual.reshape(-1, hidden).contiguous()
        res_out = torch.empty_like(res2)

    block = triton.next_power_of_2(hidden)
    # More warps for wide rows so each thread holds fewer elements in registers.
    num_warps = min(max(block // 256, 1), 16)
    _rms_norm_kernel[(x2.shape[0],)](
        x2,
        res2 if res2 is not None else x2,
        res_out if res_out is not None else y,
        weight,
        y,
        x2.stride(0),
        res2.stride(0) if res2 is not None else 0,
        y.stride(0),
        hidden,
        eps,
        HAS_RESIDUAL=residual is not None,
        BLOCK_N=block,
        num_warps=num_warps,
    )
    out = y.reshape(x.shape)
    return out, (res_out.reshape(x.shape) if res_out is not None else None)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _launch(x, weight, eps, None)[0]


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (normalized output, updated residual), like reference.fused_add_rms_norm."""
    out, new_residual = _launch(x, weight, eps, residual)
    assert new_residual is not None
    return out, new_residual
