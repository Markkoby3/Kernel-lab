"""Python wrappers for the CUDA kernels (torch.ops.kernel_lab.*).

The extension is optional: it is built only where a CUDA toolkit is present.
``available()`` says whether it loaded, so tests and benchmarks can skip it.
"""

from __future__ import annotations

import torch

try:
    from kernel_lab import _C  # noqa: F401  (registers torch.ops.kernel_lab)

    _LOADED = True
    _LOAD_ERROR = ""
except ImportError as exc:  # not built, or no CUDA toolkit at install time
    _LOADED = False
    _LOAD_ERROR = str(exc)


def available() -> bool:
    return _LOADED and torch.cuda.is_available()


def why_unavailable() -> str:
    if not _LOADED:
        return f"CUDA extension not built ({_LOAD_ERROR}); see README, 'Build the CUDA kernels'"
    return "no CUDA device"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    torch.ops.kernel_lab.rms_norm(out, x, weight, eps)
    return out


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional form: returns (out, new_residual) and leaves ``residual`` untouched.
    Call torch.ops.kernel_lab.fused_add_rms_norm directly to update in place."""
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    new_residual = residual.contiguous().clone()
    torch.ops.kernel_lab.fused_add_rms_norm(out, x, new_residual, weight, eps)
    return out, new_residual


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    out = torch.empty_like(q)
    torch.ops.kernel_lab.paged_decode_attention(
        out, q.contiguous(), k_cache, v_cache, block_table, seq_lens, scale
    )
    return out
