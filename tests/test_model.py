from __future__ import annotations

import pytest
import torch

from cauchylift.common import matrixize
from cauchylift.models import Transformer, TransformerConfig


def test_parameter_matrixization_and_bias_free():
    """All parameters must obey Phase 2 matrixization: 2D or 1D (matrixized to [length, 1]).

    Architecture must be strictly bias-free.
    """
    config = TransformerConfig(
        vocab_size=500,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        tied_embeddings=True,
    )
    model = Transformer(config)

    for name, p in model.named_parameters():
        assert "bias" not in name, f"Parameter {name} has bias, violating bias-free contract"
        m = matrixize(p)
        assert m.ndim == 2, f"Parameter {name} matrixized to ndim {m.ndim}, expected 2"
        assert min(m.shape) >= 1, f"Parameter {name} has empty dimension: {m.shape}"


def test_causal_masking():
    """Verify that attention is strictly causal: perturbing token t+1 does not affect output at t."""
    config = TransformerConfig(
        vocab_size=200,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        tied_embeddings=True,
        attention_backend="eager",
    )
    model = Transformer(config)
    model.eval()

    x1 = torch.tensor([[10, 20, 30, 40, 50]])
    x2 = torch.tensor([[10, 20, 30, 99, 88]])  # tokens at index 3, 4 are modified

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    # Outputs at positions 0, 1, 2 must be identical
    diff_pos0_2 = (out1[:, :3, :] - out2[:, :3, :]).abs().max().item()
    assert diff_pos0_2 < 1e-6, f"Causal masking violated! Position 0..2 diff: {diff_pos0_2}"

    # Output at position 3 must differ because x2[3] is modified
    diff_pos3 = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
    assert diff_pos3 > 1e-4, f"Expected difference at position 3, got diff: {diff_pos3}"


def test_weight_tying():
    """Verify tied vs untied embedding configuration."""
    # Tied
    cfg_tied = TransformerConfig(vocab_size=300, hidden_dim=64, num_layers=1, num_heads=2, tied_embeddings=True)
    model_tied = Transformer(cfg_tied)
    assert model_tied.output_head is None
    counts_tied = model_tied.count_parameters()
    assert counts_tied["tied_embeddings"] is True
    assert counts_tied["embedding_parameters"] == 300 * 64

    # Untied
    cfg_untied = TransformerConfig(vocab_size=300, hidden_dim=64, num_layers=1, num_heads=2, tied_embeddings=False)
    model_untied = Transformer(cfg_untied)
    assert model_untied.output_head is not None
    assert model_untied.tok_embeddings.weight is not model_untied.output_head.weight
    counts_untied = model_untied.count_parameters()
    assert counts_untied["tied_embeddings"] is False
    assert counts_untied["embedding_parameters"] == 2 * 300 * 64


def test_flops_calculation():
    """Verify analytical FLOPs estimation helper."""
    config = TransformerConfig(vocab_size=500, hidden_dim=64, num_layers=2, num_heads=2, max_seq_len=32)
    model = Transformer(config)
    flops = model.flops_per_token()
    assert flops > 0, "FLOPs per token must be positive"
    counts = model.count_parameters()
    expected = 6.0 * counts["non_embedding_parameters"] + 12.0 * config.num_layers * config.max_seq_len * config.hidden_dim
    assert abs(flops - expected) < 1e-4
