from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with trainable gain vector.

    Weight shape is [dim], which under CauchyLift Phase 2 matrixization
    reshapes to [dim, 1]. Strictly bias-free.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sensitive reduction in FP32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x_norm * self.weight


class MLP(nn.Module):
    """Bias-free Multi-Layer Perceptron supporting SwiGLU and GELU activations."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.activation = activation.lower()
        if self.activation == "swiglu":
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        elif self.activation == "gelu":
            self.fc = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Expected 'swiglu' or 'gelu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.proj(F.gelu(self.fc(x)))


class TransformerBlock(nn.Module):
    """Standard pre-norm decoder-only Transformer block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 2048,
        activation: str = "swiglu",
        norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            attention_dropout=attention_dropout,
            backend=attention_backend,
        )
        self.norm2 = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_dim: int | None = None
    max_seq_len: int = 1024
    activation: str = "swiglu"  # "swiglu" or "gelu"
    norm_eps: float = 1e-5
    tied_embeddings: bool = True
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    attention_backend: str = "flash"  # "flash", "eager", or "auto"
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.intermediate_dim is None:
            if self.activation.lower() == "swiglu":
                # Standard 8/3 * hidden_dim rounded to multiple of 64 or 256
                dim = int(8 * self.hidden_dim / 3)
                self.intermediate_dim = ((dim + 63) // 64) * 64
            else:
                self.intermediate_dim = 4 * self.hidden_dim


class Transformer(nn.Module):
    """Compact decoder-only Transformer obeying CauchyLift Phase 2 parameter semantics.

    - Bias-free architecture
    - RMSNorm with 1D gain vectors
    - RoPE rotary position embeddings (0 trainable positional parameters)
    - Configurable tied or untied embedding head
    - FlashAttention support on ROCm/MI300X with eager reference verification
    - Clean parameter matrixization for all optimizers
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,  # type: ignore
                max_seq_len=config.max_seq_len,
                activation=config.activation,
                norm_eps=config.norm_eps,
                attention_dropout=config.attention_dropout,
                attention_backend=config.attention_backend,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        if config.tied_embeddings:
            self.output_head = None
        else:
            self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        activation_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (logits, loss).

        Args:
            input_ids: [batch_size, seq_len]
            targets: Optional [batch_size, seq_len] for computing cross-entropy loss.
            activation_checkpointing: Whether to use gradient checkpointing.

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Scalar cross-entropy loss or None.
        """
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum sequence length {self.config.max_seq_len}"
            )

        x = self.tok_embeddings(input_ids)

        for block in self.blocks:
            if activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)

        if self.config.tied_embeddings:
            logits = F.linear(x, self.tok_embeddings.weight)
        else:
            assert self.output_head is not None
            logits = self.output_head(x)

        loss = None
        if targets is not None:
            # Shifted cross-entropy with FP32 reduction
            # Flatten predictions and targets
            logits_flat = logits.view(-1, self.config.vocab_size).to(torch.float32)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        """Count total, embedding, and non-embedding parameters."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_embeddings.weight.numel()
        if not self.config.tied_embeddings and self.output_head is not None:
            emb += self.output_head.weight.numel()
        non_emb = total - emb if self.config.tied_embeddings else total - emb
        return {
            "total_parameters": total,
            "embedding_parameters": emb,
            "non_embedding_parameters": non_emb,
            "tied_embeddings": self.config.tied_embeddings,
        }

    def flops_per_token(self) -> float:
        """Estimate FLOPs per token for forward and backward passes.

        Forward + backward FLOPs estimate:
        6 * non_embedding_parameters + 12 * num_layers * seq_len * hidden_dim
        """
        counts = self.count_parameters()
        p_non_emb = counts["non_embedding_parameters"]
        # Standard analytical approximation
        flops = 6.0 * p_non_emb + 12.0 * self.config.num_layers * self.config.max_seq_len * self.config.hidden_dim
        return flops
