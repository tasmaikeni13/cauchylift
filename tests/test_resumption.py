from __future__ import annotations

import os
import shutil
import tempfile
import pytest
import torch

from cauchylift.baselines import create_optimizer
from cauchylift.data import PackedTokenDataset
from cauchylift.models import Transformer, TransformerConfig
from cauchylift.train import Trainer, TrainingConfig


def test_uninterrupted_vs_resumed_run():
    """Uninterrupted and resumed runs must agree within declared tolerance and preserve exact token order."""
    temp_dir = tempfile.mkdtemp(prefix="cauchylift_test_resume_")
    ckpt_dir = os.path.join(temp_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 42

        def make_components():
            torch.manual_seed(seed)
            cfg = TransformerConfig(
                vocab_size=50257,
                hidden_dim=64,
                num_layers=2,
                num_heads=2,
                max_seq_len=32,
                tied_embeddings=True,
                attention_backend="auto",
            )
            model = Transformer(cfg).to(device)
            opt = create_optimizer("cauchylift", model, lr=1e-3)
            data = PackedTokenDataset(split="train", max_seq_len=32, batch_size=2, seed=seed)
            return model, opt, data

        # 1. Uninterrupted Run: 10 steps, saving checkpoint at step 5
        model_a, opt_a, data_a = make_components()
        cfg_a = TrainingConfig(
            max_steps=10,
            batch_size=2,
            seq_len=32,
            device=device,
            precision="bf16" if device.startswith("cuda") else "fp32",
            checkpoint_interval=5,
            checkpoint_dir=ckpt_dir,
            log_file=os.path.join(temp_dir, "metrics_uninterrupted.jsonl"),
            seed=seed,
        )
        trainer_a = Trainer(model_a, opt_a, data_a, config=cfg_a)
        records_a = trainer_a.train()

        losses_uninterrupted = [r.loss for r in records_a]
        tokens_uninterrupted = [r.tokens for r in records_a]

        # 2. Resumed Run: start fresh, resume from step 5 checkpoint, run remaining 5 steps
        model_b, opt_b, data_b = make_components()
        cfg_b = TrainingConfig(
            max_steps=10,
            batch_size=2,
            seq_len=32,
            device=device,
            precision="bf16" if device.startswith("cuda") else "fp32",
            checkpoint_interval=100,  # no extra save
            checkpoint_dir=ckpt_dir,
            log_file=os.path.join(temp_dir, "metrics_resumed.jsonl"),
            seed=seed,
        )
        trainer_b = Trainer(model_b, opt_b, data_b, config=cfg_b)
        ckpt_step_5 = os.path.join(ckpt_dir, "checkpoint_step_5.pt")
        trainer_b.resume_from_checkpoint(ckpt_step_5)

        assert trainer_b.step == 5
        assert trainer_b.tokens_seen == tokens_uninterrupted[4]

        records_b = trainer_b.train()
        losses_resumed = [r.loss for r in records_b]
        tokens_resumed = [r.tokens for r in records_b]

        # Verify token counts match exactly
        assert tokens_resumed == tokens_uninterrupted[5:], (
            f"Token counts diverged: resumed={tokens_resumed}, uninterrupted={tokens_uninterrupted[5:]}"
        )

        # Verify loss values match within declared numerical tolerance
        tolerance = 1e-2 if device.startswith("cuda") else 1e-5
        for i, (loss_unint, loss_res) in enumerate(zip(losses_uninterrupted[5:], losses_resumed), start=6):
            diff = abs(loss_unint - loss_res)
            assert diff <= tolerance, (
                f"Loss at step {i} diverged: uninterrupted={loss_unint}, resumed={loss_res}, diff={diff}"
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
