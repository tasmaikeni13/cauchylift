from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cauchylift.baselines import create_optimizer
from cauchylift.models import Transformer, TransformerConfig


@pytest.mark.parametrize("opt_name,lr", [
    ("cauchylift", 5e-2),
    ("adamw", 1e-2),
    ("muon", 2e-2),
    ("soap", 1e-2),
    ("sinkgd", 5e-2),
    ("normalized_gd", 5e-2),
    ("sign_descent", 1e-2),
])
def test_tiny_model_overfit_each_optimizer(opt_name: str, lr: float):
    """Verify that a tiny Transformer can overfit a tiny fixed batch with each optimizer without failure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 256
    seq_len = 16
    batch_size = 2

    torch.manual_seed(42)
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=seq_len,
        tied_embeddings=True,
        attention_backend="auto",
    )
    model = Transformer(config).to(device)
    optimizer = create_optimizer(opt_name, model, lr=lr)

    # Fixed tiny batch
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    initial_loss = None
    final_loss = None

    use_autocast = device.startswith("cuda")
    for step in range(50):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(x, y)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss became non-finite at step {step} for {opt_name}"

        loss_val = float(loss.item())
        if initial_loss is None:
            initial_loss = loss_val

        loss.backward()
        optimizer.step()
        final_loss = loss_val

    assert initial_loss is not None and final_loss is not None
    # Verify that loss decreased significantly (by at least 50% or down to < 1.0)
    assert final_loss < initial_loss * 0.6, (
        f"Optimizer {opt_name} failed to overfit: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}"
    )
