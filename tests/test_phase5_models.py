from __future__ import annotations

import pytest
import torch

from cauchylift.baselines import create_optimizer
from cauchylift.models import (
    ConvSSM,
    ConvSSMConfig,
    VisionTransformer,
    VisionTransformerConfig,
)


def test_vit_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VisionTransformerConfig(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
    )
    model = VisionTransformer(config).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    logits, loss = model(x, y)
    assert logits.shape == (4, 10)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_conv_ssm_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ConvSSMConfig(
        vocab_size=256,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        conv_kernel=5,
        state_dim=8,
        max_seq_len=32,
    )
    model = ConvSSM(config).to(device)
    x = torch.randint(0, 256, (2, 16), device=device)
    y = torch.randint(0, 256, (2, 16), device=device)

    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("opt_name", [
    "cauchylift",
    "adamw",
    "muon",
    "soap",
    "sinkgd",
    "normalized_gd",
    "sign_descent",
])
def test_vit_optimizer_step(opt_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VisionTransformerConfig(
        img_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=4,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_dim=64,
    )
    model = VisionTransformer(config).to(device)
    optimizer = create_optimizer(opt_name, model, lr=1e-3)
    x = torch.randn(2, 3, 16, 16, device=device)
    y = torch.randint(0, 4, (2,), device=device)

    optimizer.zero_grad()
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("opt_name", [
    "cauchylift",
    "adamw",
    "muon",
    "soap",
    "sinkgd",
    "normalized_gd",
    "sign_descent",
])
def test_conv_ssm_optimizer_step(opt_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ConvSSMConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        conv_kernel=5,
        state_dim=8,
        max_seq_len=16,
    )
    model = ConvSSM(config).to(device)
    optimizer = create_optimizer(opt_name, model, lr=1e-3)
    x = torch.randint(0, 64, (2, 8), device=device)
    y = torch.randint(0, 64, (2, 8), device=device)

    optimizer.zero_grad()
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    optimizer.step()
