from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm


@dataclass
class ConvSSMConfig:
    vocab_size: int = 50257
    hidden_dim: int = 128
    intermediate_dim: int = 256
    num_layers: int = 4
    conv_kernel: int = 7
    state_dim: int = 16
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tied_embeddings: bool = True


class ConvSSMBlock(nn.Module):
    """Bias-free non-square Convolutional and State-Space block.

    Features non-square parameters across all projections:
    - in_proj: [intermediate_dim, hidden_dim]
    - conv_weight: [intermediate_dim, intermediate_dim // 2, kernel_size] -> flattened to non-square [intermediate_dim, (intermediate_dim // 2) * K]
    - B_proj: [intermediate_dim // 2, state_dim]
    - C_proj: [intermediate_dim // 2, state_dim]
    - out_proj: [hidden_dim, intermediate_dim]
    - mlp_in: [intermediate_dim, hidden_dim]
    - mlp_out: [hidden_dim, intermediate_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        conv_kernel: int = 7,
        state_dim: int = 16,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.conv_kernel = conv_kernel
        self.state_dim = state_dim
        self.d_inner = intermediate_dim // 2

        self.norm1 = RMSNorm(hidden_dim, eps=norm_eps)
        # Non-square input projection [intermediate_dim, hidden_dim]
        self.in_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)

        # Non-square 1D temporal convolution weight: [d_inner, d_inner, K]
        # In Phase 2, flattens to [d_inner, d_inner * K]
        self.conv_weight = nn.Parameter(
            torch.empty(self.d_inner, self.d_inner, conv_kernel)
        )

        # Non-square state-space projection matrices
        self.B_proj = nn.Parameter(torch.empty(self.d_inner, state_dim))
        self.C_proj = nn.Parameter(torch.empty(self.d_inner, state_dim))

        # Non-square output projection [hidden_dim, intermediate_dim]
        self.out_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        self.norm2 = RMSNorm(hidden_dim, eps=norm_eps)
        # Non-square MLP projections
        self.mlp_in = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.mlp_out = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        # Project to intermediate dimension: [B, L, intermediate_dim]
        projected = self.in_proj(x_norm)
        # Split into branch1 (conv + SSM) and branch2 (multiplicative gate)
        branch1, gate = projected.split(self.d_inner, dim=-1)

        # 1D Causal Convolution on branch1:
        # conv input: [B, d_inner, L]
        u = branch1.transpose(1, 2)
        # Left-pad for causality
        u_padded = F.pad(u, (self.conv_kernel - 1, 0))
        # Convolution with non-square weight [d_inner, d_inner, K]
        conv_out = F.conv1d(u_padded, self.conv_weight)  # [B, d_inner, L]
        conv_out = conv_out.transpose(1, 2)  # [B, L, d_inner]

        # Simple discrete state-space recurrence step along sequence:
        # s_t = s_{t-1} + u_t @ B, y_t = s_t @ C^T
        # Efficient parallel cumsum scan:
        u_B = conv_out @ self.B_proj  # [B, L, state_dim]
        state_seq = torch.cumsum(u_B, dim=1)  # [B, L, state_dim]
        ssm_out = state_seq @ self.C_proj.t()  # [B, L, d_inner]

        # Gated combination
        y = torch.cat([ssm_out, F.silu(gate)], dim=-1)
        y = self.out_proj(y)
        x = residual + y

        # MLP block
        residual = x
        x_norm = self.norm2(x)
        mlp_h = F.silu(self.mlp_in(x_norm))
        x = residual + self.mlp_out(mlp_h)

        return x


class ConvSSM(nn.Module):
    """Non-Square Convolutional / State-Space Model for held-out validation.

    Obeying Phase 2 parameter semantics:
    - Strictly bias-free
    - RMSNorm with 1D gain vectors
    - Non-square matrix dimensions across all components
    """

    def __init__(self, config: ConvSSMConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.blocks = nn.ModuleList([
            ConvSSMBlock(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                conv_kernel=config.conv_kernel,
                state_dim=config.state_dim,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        if config.tied_embeddings:
            self.output_head = None
        else:
            self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ConvSSMBlock):
            torch.nn.init.normal_(module.conv_weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.B_proj, mean=0.0, std=std)
            torch.nn.init.normal_(module.C_proj, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L = input_ids.shape
        x = self.tok_embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.config.tied_embeddings:
            logits = F.linear(x, self.tok_embeddings.weight)
        else:
            assert self.output_head is not None
            logits = self.output_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.config.vocab_size).to(torch.float32)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_embeddings.weight.numel()
        return {
            "total_parameters": total,
            "embedding_parameters": emb,
            "non_embedding_parameters": total - emb,
        }

    def flops_per_token(self) -> float:
        counts = self.count_parameters()
        return 6.0 * counts["non_embedding_parameters"]
