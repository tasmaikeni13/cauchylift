from __future__ import annotations

import math
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cauchylift.models.attention import (
    apply_rope,
    eager_causal_attention,
    precompute_rope_frequencies,
)


def test_rope_rotation_properties():
    """Verify RoPE rotation norm preservation and zero-position identity."""
    dim = 16
    max_len = 32
    angles = precompute_rope_frequencies(dim, max_len)

    # Position 0 should have 0 angle (cos=1, sin=0), so RoPE at pos 0 is identity
    x = torch.randn(2, 4, 1, dim)
    rotated = apply_rope(x, angles)
    assert (x - rotated).abs().max().item() < 1e-6, "RoPE at position 0 must be identity"

    # RoPE is an orthogonal rotation; it must preserve vector 2-norm
    x_full = torch.randn(2, 4, 16, dim)
    rotated_full = apply_rope(x_full, angles)
    orig_norm = torch.linalg.vector_norm(x_full, dim=-1)
    rot_norm = torch.linalg.vector_norm(rotated_full, dim=-1)
    assert (orig_norm - rot_norm).abs().max().item() < 1e-5, "RoPE must preserve vector norms"


@pytest.mark.rocm
def test_flash_attention_parity_with_eager_fp32():
    """Verify that flash attention on ROCm matches eager FP32 attention on forward and backward."""
    if not (torch.cuda.is_available() and getattr(torch.version, "hip", None)):
        pytest.skip("Requires PyTorch ROCm device")

    B, H, S, D = 2, 4, 64, 32
    torch.manual_seed(12345)

    q_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # Eager reference forward and backward
    out_eager = eager_causal_attention(q_fp32, k_fp32, v_fp32)
    loss_eager = (out_eager * 2.0).sum()
    loss_eager.backward()

    # Flash attention in BF16
    q_bf16 = q_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    k_bf16 = k_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    v_bf16 = v_fp32.detach().to(torch.bfloat16).requires_grad_(True)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = torch.nn.functional.scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16, is_causal=True
        )
    loss_flash = (out_flash * 2.0).sum()
    loss_flash.backward()

    # Output and gradient comparison
    out_diff = (out_eager - out_flash.float()).abs().max().item()
    gq_diff = (q_fp32.grad - q_bf16.grad.float()).abs().max().item()
    gk_diff = (k_fp32.grad - k_bf16.grad.float()).abs().max().item()
    gv_diff = (v_fp32.grad - v_bf16.grad.float()).abs().max().item()

    # BF16 representation tolerance
    assert out_diff < 0.05, f"Flash output diff {out_diff} exceeds 0.05"
    assert gq_diff < 0.05, f"Flash grad Q diff {gq_diff} exceeds 0.05"
    assert gk_diff < 0.05, f"Flash grad K diff {gk_diff} exceeds 0.05"
    assert gv_diff < 0.05, f"Flash grad V diff {gv_diff} exceeds 0.05"
