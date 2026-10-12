"""Run kernels on a GPU when there is one, otherwise through Triton's CPU interpreter.

TRITON_INTERPRET must be set before triton is imported, so it is set here, at the
very start of the test session.
"""

from __future__ import annotations

import os

import pytest

if "TRITON_INTERPRET" not in os.environ:
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        gpu = False
    if not gpu:
        os.environ["TRITON_INTERPRET"] = "1"

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture
def device() -> str:
    return DEVICE


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    return {
        torch.float32: {"atol": 1e-4, "rtol": 1e-4},
        torch.float16: {"atol": 2e-3, "rtol": 2e-3},
        torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    }[dtype]


# bfloat16 has no NumPy equivalent, so the CPU interpreter cannot run it.
DTYPES = [torch.float32, torch.float16] + ([] if INTERPRETED else [torch.bfloat16])
