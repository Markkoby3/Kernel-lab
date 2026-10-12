from __future__ import annotations

import pytest
import torch

from kernel_lab import reference
from kernel_lab.triton_kernels import fused_add_rms_norm, rms_norm
from tests.conftest import DTYPES, tolerance

# Odd sizes catch masking bugs; powers of two and model sizes catch layout ones.
SHAPES = [(1, 16), (3, 100), (4, 256), (2, 3, 128), (7, 1000)]


def make(shape, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g).to(device=device, dtype=dtype)
    r = torch.randn(*shape, generator=g).to(device=device, dtype=dtype)
    w = (1 + 0.1 * torch.randn(shape[-1], generator=g)).to(device=device, dtype=dtype)
    return x, r, w


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_matches_reference(shape, dtype, device):
    x, _, w = make(shape, dtype, device)
    torch.testing.assert_close(rms_norm(x, w), reference.rms_norm(x, w), **tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_add_rms_norm_matches_reference(shape, dtype, device):
    x, r, w = make(shape, dtype, device)
    out, new_res = fused_add_rms_norm(x, r, w)
    ref_out, ref_res = reference.fused_add_rms_norm(x, r, w)
    torch.testing.assert_close(new_res, ref_res, atol=0, rtol=0)  # same rounding point
    torch.testing.assert_close(out, ref_out, **tolerance(dtype))
    assert out.shape == x.shape and new_res.shape == x.shape


def test_inputs_are_not_modified(device):
    x, r, w = make((3, 64), torch.float32, device)
    x0, r0 = x.clone(), r.clone()
    fused_add_rms_norm(x, r, w)
    assert torch.equal(x, x0) and torch.equal(r, r0)


def test_strided_rows_are_supported(device):
    base = torch.randn(4, 192, device=device)
    x = base[:, :128]  # row stride 192, not 128
    w = torch.ones(128, device=device)
    torch.testing.assert_close(rms_norm(x, w), reference.rms_norm(x, w), atol=1e-5, rtol=1e-5)


def test_eps_matters_for_near_zero_rows(device):
    x = torch.zeros(2, 32, device=device)
    w = torch.ones(32, device=device)
    assert torch.isfinite(rms_norm(x, w, eps=1e-6)).all()


def test_validation(device):
    x = torch.randn(2, 32, device=device)
    with pytest.raises(ValueError, match="does not match"):
        rms_norm(x, torch.ones(16, device=device))
    with pytest.raises(ValueError, match="same shape"):
        fused_add_rms_norm(x, torch.randn(2, 16, device=device), torch.ones(32, device=device))
