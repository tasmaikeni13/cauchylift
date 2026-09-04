from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm


@dataclass
class VisionTransformerConfig:
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_dim: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * (self.patch_size ** 2)


class ViTAttention(nn.Module):
    """Bias-free multi-head self-attention for Vision Transformer."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class ViTBlock(nn.Module):
    """Pre-norm bias-free Vision Transformer block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = ViTAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp_fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.mlp_fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Small Vision Transformer obeying CauchyLift Phase 2 parameter semantics.

    - Bias-free architecture throughout
    - RMSNorm with 1D gain vectors
    - Non-overlapping linear patch projection
    - Learnable 2D position embedding matrix
    - Mean pooling over patch tokens
    - Bias-free classification head
    """

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_proj = nn.Linear(config.patch_dim, config.hidden_dim, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(config.num_patches, config.hidden_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,
                dropout=config.dropout,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.head = nn.Linear(config.hidden_dim, config.num_classes, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=std)

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass on image tensors.

        Args:
            images: [batch_size, channels, height, width]
            targets: Optional [batch_size] ground truth class indices.

        Returns:
            logits: [batch_size, num_classes]
            loss: Cross-entropy loss or None.
        """
        B, C, H, W = images.shape
        P = self.config.patch_size
        assert H == self.config.img_size and W == self.config.img_size, f"Expected {self.config.img_size}x{self.config.img_size}, got {H}x{W}"

        # Extract non-overlapping patches: [B, num_patches, patch_dim]
        # (B, C, H//P, P, W//P, P) -> (B, H//P, W//P, C, P, P) -> (B, num_patches, C*P*P)
        patches = (
            images.view(B, C, H // P, P, W // P, P)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(B, self.config.num_patches, self.config.patch_dim)
        )

        x = self.patch_proj(patches) + self.pos_embed.unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Global average pooling over patches
        x = x.mean(dim=1)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_fp32 = logits.to(torch.float32)
            loss = F.cross_entropy(logits_fp32, targets)

        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        head_params = self.head.weight.numel()
        return {
            "total_parameters": total,
            "head_parameters": head_params,
            "backbone_parameters": total - head_params,
        }

    def flops_per_example(self) -> float:
        counts = self.count_parameters()
        p_count = counts["total_parameters"]
        flops = 6.0 * p_count + 12.0 * self.config.num_layers * self.config.num_patches * self.config.hidden_dim
        return flops
