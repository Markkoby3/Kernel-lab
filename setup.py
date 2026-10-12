"""Builds the optional CUDA extension (kernel_lab._C).

The Triton kernels, references and benchmarks are pure Python. The CUDA kernels
are compiled only when a CUDA toolkit is found (or KERNEL_LAB_BUILD_CUDA=1), so
`pip install .` works on a laptop and the CPU-only CI.

Build on a GPU machine:
    pip install -e . --no-build-isolation
Target specific GPUs (default: the GPU in this machine, else PyTorch's list):
    TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" pip install -e . --no-build-isolation
"""

from __future__ import annotations

import os

from setuptools import setup


def cuda_extension():
    if os.environ.get("KERNEL_LAB_NO_CUDA") == "1":
        return [], {}
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return [], {}
    if CUDA_HOME is None and os.environ.get("KERNEL_LAB_BUILD_CUDA") != "1":
        print("kernel-lab: no CUDA toolkit found; installing without the CUDA extension")
        return [], {}

    ext = CUDAExtension(
        name="kernel_lab._C",
        sources=["csrc/torch_bindings.cpp", "csrc/rmsnorm.cu", "csrc/paged_attention.cu"],
        include_dirs=[os.path.abspath("csrc")],
        extra_compile_args={
            # The C++ standard is left to PyTorch (C++17 or C++20 depending on version).
            "cxx": ["-O3"],
            # -lineinfo maps SASS back to source lines in Nsight Compute.
            "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        },
    )
    return [ext], {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


ext_modules, cmdclass = cuda_extension()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
