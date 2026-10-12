from __future__ import annotations

import pytest

from kernel_lab.bench import roofline
from kernel_lab.bench.__main__ import Row, main, to_markdown


def test_rms_norm_bytes_count_each_tensor_once():
    w = roofline.rms_norm_work(rows=2, hidden=8, elem_bytes=2, fused_residual=True)
    assert w.bytes == 2 * 8 * 2 * 4 + 8 * 2  # x, residual in; out, residual out; + weight
    assert roofline.rms_norm_work(2, 8, 2, False).bytes == 2 * 8 * 2 * 2 + 8 * 2


def test_decode_attention_is_memory_bound_on_every_listed_gpu():
    work = roofline.decode_attention_work([4096] * 8, 32, 8, 128, elem_bytes=2)
    # GQA group 4: about 4 FLOPs per byte of KV, far below every ridge point.
    assert 3.5 < work.intensity < 4.5
    for gpu in roofline.GPU_PEAKS:
        assert work.intensity < gpu.ridge_flops_per_byte


def test_decode_kv_bytes():
    work = roofline.decode_attention_work([10, 20], num_q_heads=4, num_kv_heads=2, head_dim=8, elem_bytes=2)
    assert work.bytes == 2 * 30 * 2 * 8 * 2 + 2 * 2 * 4 * 8 * 2
    assert work.flops == 4 * 30 * 4 * 8


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", "H100 80GB HBM3"),
        ("NVIDIA A100 80GB PCIe", "A100 80GB PCIe"),
        ("NVIDIA L40", "L40"),
        ("NVIDIA L40S", "L40S"),
        ("Tesla T4", None),
        ("NVIDIA H100 PCIe", "H100 PCIe"),
        ("NVIDIA A100-SXM4-80GB", "A100-SXM4-80GB"),
        ("NVIDIA A100-PCIE-40GB", "A100"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA A10G", "A10G"),
        ("NVIDIA GeForce RTX 4090", "RTX 4090"),
    ],
)
def test_gpu_lookup(device, expected):
    peak = roofline.lookup(device)
    assert (peak.name if peak else None) == expected


def test_markdown_marks_skips_and_wrong_results():
    rows = [
        Row("k", "s", "torch", 10.0, 100.0, 50.0, 1.0, 0.0),
        Row("k", "s", "bad", None, None, None, None, 0.5, "WRONG RESULT, not timed"),
    ]
    text = to_markdown(rows, "### test")
    assert "| k | s | torch | 10.0 | 100.0 | 50.0% | 1.0x | 0.0e+00 |" in text
    assert "bad (WRONG RESULT, not timed)" in text


def test_cli_without_gpu_exits_cleanly(capsys):
    import torch

    if torch.cuda.is_available():
        pytest.skip("GPU present")
    assert main(["rmsnorm"]) == 2
    assert "No CUDA device" in capsys.readouterr().err
