#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cauchylift.baselines import create_optimizer
from cauchylift.data import (
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_LICENSE,
    FINEWEB_EDU_REPO,
    FINEWEB_EDU_REVISION,
    PARTITION_SHARDS,
    TOKENIZER_LICENSE,
    TOKENIZER_NAME,
    PackedTokenDataset,
    verify_partition_disjointness,
)
from cauchylift.models import Transformer, TransformerConfig
from cauchylift.models.attention import eager_causal_attention
from cauchylift.train import Trainer, TrainingConfig


def run_preflight() -> dict[str, Any]:
    print("Running Phase 4 preflight checks...")
    is_cuda = torch.cuda.is_available()
    hip_version = getattr(torch.version, "hip", None)
    device_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    bf16_ok = torch.cuda.is_bf16_supported() if is_cuda else False

    part_info = verify_partition_disjointness()

    return {
        "is_cuda": is_cuda,
        "hip_version": hip_version,
        "device_name": device_name,
        "bf16_supported": bf16_ok,
        "partitions_disjoint": part_info["is_disjoint"],
        "dataset_repo": FINEWEB_EDU_REPO,
        "dataset_revision": FINEWEB_EDU_REVISION,
        "dataset_config": FINEWEB_EDU_CONFIG,
        "dataset_license": FINEWEB_EDU_LICENSE,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_license": TOKENIZER_LICENSE,
    }


def run_flash_attention_parity() -> dict[str, float]:
    print("Testing Flash Attention parity on MI300X...")
    B, H, S, D = 2, 4, 64, 32
    torch.manual_seed(42)

    q_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True)

    out_eager = eager_causal_attention(q_fp32, k_fp32, v_fp32)
    loss_eager = out_eager.sum()
    loss_eager.backward()

    q_bf16 = q_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    k_bf16 = k_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    v_bf16 = v_fp32.detach().to(torch.bfloat16).requires_grad_(True)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = torch.nn.functional.scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16, is_causal=True
        )
    loss_flash = out_flash.sum()
    loss_flash.backward()

    out_diff = float((out_eager - out_flash.float()).abs().max().item())
    gq_diff = float((q_fp32.grad - q_bf16.grad.float()).abs().max().item())
    gk_diff = float((k_fp32.grad - k_bf16.grad.float()).abs().max().item())
    gv_diff = float((v_fp32.grad - v_bf16.grad.float()).abs().max().item())

    return {
        "out_diff": out_diff,
        "grad_q_diff": gq_diff,
        "grad_k_diff": gk_diff,
        "grad_v_diff": gv_diff,
    }


def run_overfit_screen() -> dict[str, Any]:
    print("Running overfit screen for all viable optimizers...")
    optimizers = [
        ("cauchylift", 5e-2),
        ("adamw", 1e-2),
        ("muon", 2e-2),
        ("soap", 1e-2),
        ("sinkgd", 5e-2),
        ("normalized_gd", 5e-2),
        ("sign_descent", 1e-2),
    ]

    results = {}
    vocab_size = 256
    seq_len = 16
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for opt_name, lr in optimizers:
        torch.manual_seed(42)
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=seq_len,
            tied_embeddings=True,
            attention_backend="auto",
        )
        model = Transformer(cfg).to(device)
        optimizer = create_optimizer(opt_name, model, lr=lr)

        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        init_loss = None
        final_loss = None

        t0 = time.perf_counter()
        use_autocast = device.startswith("cuda")

        for step in range(50):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                _, loss = model(x, y)
            assert loss is not None
            loss_val = float(loss.item())
            if init_loss is None:
                init_loss = loss_val
            loss.backward()
            optimizer.step()
            final_loss = loss_val

        elapsed = time.perf_counter() - t0
        passed = final_loss is not None and init_loss is not None and final_loss < init_loss * 0.6
        results[opt_name] = {
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "loss_reduction_pct": float(100.0 * (init_loss - final_loss) / init_loss) if init_loss else 0.0,
            "elapsed_seconds": elapsed,
            "passed": passed,
        }
        print(f"  [{'PASS' if passed else 'FAIL'}] {opt_name}: init={init_loss:.4f} -> final={final_loss:.4f} ({results[opt_name]['loss_reduction_pct']:.1f}% reduction in {elapsed:.2f}s)")

    return results


def run_resumption_check() -> dict[str, Any]:
    print("Testing checkpoint resumption determinism...")
    temp_dir = tempfile.mkdtemp(prefix="cauchylift_smoke_resume_")
    ckpt_dir = os.path.join(temp_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    try:
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

        # Uninterrupted run
        m_a, o_a, d_a = make_components()
        c_a = TrainingConfig(
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
        t_a = Trainer(m_a, o_a, d_a, config=c_a)
        records_a = t_a.train()

        losses_unint = [r.loss for r in records_a]
        tokens_unint = [r.tokens for r in records_a]

        # Resumed run
        m_b, o_b, d_b = make_components()
        c_b = TrainingConfig(
            max_steps=10,
            batch_size=2,
            seq_len=32,
            device=device,
            precision="bf16" if device.startswith("cuda") else "fp32",
            checkpoint_interval=100,
            checkpoint_dir=ckpt_dir,
            log_file=os.path.join(temp_dir, "metrics_resumed.jsonl"),
            seed=seed,
        )
        t_b = Trainer(m_b, o_b, d_b, config=c_b)
        ckpt_5 = os.path.join(ckpt_dir, "checkpoint_step_5.pt")
        t_b.resume_from_checkpoint(ckpt_5)
        records_b = t_b.train()

        losses_res = [r.loss for r in records_b]
        tokens_res = [r.tokens for r in records_b]

        tokens_match = tokens_res == tokens_unint[5:]
        max_loss_diff = max(abs(u - r) for u, r in zip(losses_unint[5:], losses_res))
        tolerance = 1e-2 if device.startswith("cuda") else 1e-5
        passed = tokens_match and max_loss_diff <= tolerance

        print(f"  Resumption check: tokens_match={tokens_match}, max_loss_diff={max_loss_diff:.6e}, passed={passed}")
        return {
            "tokens_match": tokens_match,
            "max_loss_diff": max_loss_diff,
            "declared_tolerance": tolerance,
            "passed": passed,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    os.makedirs("analysis/results", exist_ok=True)
    t_start = time.perf_counter()

    preflight = run_preflight()
    parity = run_flash_attention_parity()
    overfit = run_overfit_screen()
    resumption = run_resumption_check()

    all_passed = (
        preflight["partitions_disjoint"]
        and parity["out_diff"] < 0.05
        and all(v["passed"] for v in overfit.values())
        and resumption["passed"]
    )

    summary = {
        "gate": "PASS" if all_passed else "FAIL",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_seconds": time.perf_counter() - t_start,
        "preflight": preflight,
        "flash_attention_parity": parity,
        "overfit_screen": overfit,
        "checkpoint_resumption": resumption,
    }

    out_path = "analysis/results/phase4_smoke_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 4 smoke suite complete. Result: {summary['gate']}. Written to {out_path}")


if __name__ == "__main__":
    main()
