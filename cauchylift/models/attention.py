from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotary positional frequencies.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length.
        theta: Base frequency constant.

    Returns:
        Tensor of shape [max_seq_len, dim // 2] with complex frequencies (cos, sin).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for RoPE, got {dim}")
    half_dim = dim // 2
    freq_exponents = torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    freqs = 1.0 / (theta ** freq_exponents)  # [half_dim]
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    angles = torch.outer(positions, freqs)  # [max_seq_len, half_dim]
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to x.

    Args:
        x: Tensor of shape [batch_size, num_heads, seq_len, head_dim].
        angles: Angles of shape [seq_len, head_dim // 2].

    Returns:
        Tensor of shape [batch_size, num_heads, seq_len, head_dim] with RoPE applied.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2
    x_reshaped = x.reshape(batch_size, num_heads, seq_len, half_dim, 2)
    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]

    cos = torch.cos(angles[:seq_len]).to(dtype=x.dtype, device=x.device)  # [seq_len, half_dim]
    sin = torch.sin(angles[:seq_len]).to(dtype=x.dtype, device=x.device)  # [seq_len, half_dim]

    # Broadcast over batch and heads: [1, 1, seq_len, half_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack([out0, out1], dim=-1).reshape(batch_size, num_heads, seq_len, head_dim)
    return out


def eager_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Exact eager FP32 reference for causal scaled dot-product attention.

    Args:
        q: [batch_size, num_heads, seq_len_q, head_dim]
        k: [batch_size, num_heads, seq_len_kv, head_dim]
        v: [batch_size, num_heads, seq_len_kv, head_dim]
        scale: Scaling factor, default 1 / sqrt(head_dim)

    Returns:
        out: [batch_size, num_heads, seq_len_q, head_dim]
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len_q = q.shape[2]
    seq_len_kv = k.shape[2]

    # Compute attention scores in FP32 for maximum numerical fidelity
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)

    scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * scale  # [B, H, Sq, Skv]

    # Apply causal mask: mask positions where col > row
    causal_mask = torch.triu(
        torch.full((seq_len_q, seq_len_kv), float("-inf"), device=q.device, dtype=torch.float32),
        diagonal=1 + (seq_len_kv - seq_len_q),
    )
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_weights = F.softmax(scores, dim=-1)
    out_fp32 = torch.matmul(attn_weights, v_fp32)
    return out_fp32.to(dtype=q.dtype)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention supporting FlashAttention on ROCm."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        backend: str = "flash",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.backend = backend  # "flash", "eager", or "auto"
        self.dropout_p = attention_dropout

        # Trainable projection matrices (strictly bias-free)
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # Precompute RoPE angles as non-trainable persistent buffer
        rope_angles = precompute_rope_frequencies(self.head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("rope_angles", rope_angles, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_angles)
        k = apply_rope(k, self.rope_angles)

        # Handle GQA if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Attention dispatch: on ROCm MI300X, flash attention requires bfloat16 or float16
        use_flash = (
            (self.backend == "flash" or (self.backend == "auto" and x.is_cuda))
            and x.is_cuda
            and q.dtype in (torch.bfloat16, torch.float16)
        )

        if use_flash:
            # Enforce FLASH_ATTENTION backend kernel on ROCm / MI300X
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        is_causal=True,
                    )
            except Exception:
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        is_causal=True,
                    )
        else:
            # Eager FP32 reference
            attn_out = eager_causal_attention(q, k, v)

        # Reshape and project out
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(attn_out)
