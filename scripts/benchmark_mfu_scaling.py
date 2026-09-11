#!/usr/bin/env python3
"""Benchmark throughput and MFU scaling on Google Cloud TPU v6e."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

PEAK_TFLOPS_PER_CHIP = 460.0  # BF16 dense peak for TPU v6e (Trillium)


def compute_model_flops_per_token(cfg, num_params: int) -> float:
    return 6.0 * float(num_params) + 12.0 * cfg.num_layers * cfg.hidden_dim * cfg.max_seq_len


def run_1x(batch_size: int = 4, seq_len: int = 2048, num_steps: int = 15):
    from cauchylift import CauchyLift
    from cauchylift.models.transformer import Transformer, TransformerConfig

    dev = torch_xla.device()
    cfg = TransformerConfig(
        vocab_size=50257,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=2048,
        max_seq_len=seq_len,
        activation="swiglu",
        norm_eps=1e-5,
        tied_embeddings=True,
        attention_backend="flash",
    )
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
    opt = CauchyLift(model.parameters(), lr=1e-3, backend="auto")
    num_params = sum(p.numel() for p in model.parameters())

    tokens_per_step = batch_size * seq_len
    flops_per_token = compute_model_flops_per_token(cfg, num_params)

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        torch_xla.sync()

    times = []
    for step in range(num_steps):
        opt.zero_grad()
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        t0 = time.perf_counter()
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        torch_xla.sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    tok_per_sec = tokens_per_step / avg_time
    tflops = (tok_per_sec * flops_per_token) / 1e12
    mfu = (tflops / PEAK_TFLOPS_PER_CHIP) * 100.0

    res = {
        "devices": 1,
        "batch_size_per_chip": batch_size,
        "global_batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "step_time_ms": avg_time * 1000.0,
        "tokens_per_sec": tok_per_sec,
        "achieved_tflops": tflops,
        "mfu_percent": mfu,
    }
    print(f"1x TPU: {res['step_time_ms']:.1f} ms/step | {res['tokens_per_sec']:,.0f} tok/s | {res['achieved_tflops']:.1f} TFLOPs | MFU: {res['mfu_percent']:.1f}%")
    with open("/tmp/bench_1x.json", "w") as f:
        json.dump(res, f)


def _bench_8x_rank(index: int):
    from cauchylift import CauchyLift
    from cauchylift.models.transformer import Transformer, TransformerConfig

    dev = torch_xla.device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()

    batch_size = 4
    seq_len = 2048
    num_steps = 15

    cfg = TransformerConfig(
        vocab_size=50257,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=2048,
        max_seq_len=seq_len,
        activation="swiglu",
        norm_eps=1e-5,
        tied_embeddings=True,
        attention_backend="flash",
    )
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
    opt = CauchyLift(model.parameters(), lr=1e-3, backend="auto")
    num_params = sum(p.numel() for p in model.parameters())

    tokens_per_step = batch_size * seq_len * world_size
    flops_per_token = compute_model_flops_per_token(cfg, num_params)

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        xm.reduce_gradients(opt)
        opt.step()
        torch_xla.sync()

    times = []
    for step in range(num_steps):
        opt.zero_grad()
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
        t0 = time.perf_counter()
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        xm.reduce_gradients(opt)
        opt.step()
        torch_xla.sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    tok_per_sec = tokens_per_step / avg_time
    total_peak = PEAK_TFLOPS_PER_CHIP * world_size
    tflops = (tok_per_sec * flops_per_token) / 1e12
    mfu = (tflops / total_peak) * 100.0

    if rank == 0:
        res = {
            "devices": world_size,
            "batch_size_per_chip": batch_size,
            "global_batch_size": batch_size * world_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "step_time_ms": avg_time * 1000.0,
            "tokens_per_sec": tok_per_sec,
            "achieved_tflops": tflops,
            "mfu_percent": mfu,
        }
        print(f"8x TPU: {res['step_time_ms']:.1f} ms/step | {res['tokens_per_sec']:,.0f} tok/s | {res['achieved_tflops']:.1f} TFLOPs | MFU: {res['mfu_percent']:.1f}%")
        with open("/tmp/bench_8x.json", "w") as f:
            json.dump(res, f)


def run_8x():
    xmp.spawn(_bench_8x_rank, args=())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["1x", "8x", "all"], default="all")
    args = parser.parse_args()

    if args.mode == "1x":
        run_1x(batch_size=4, seq_len=2048)
    elif args.mode == "8x":
        run_8x()
    elif args.mode == "all":
        print("=" * 70)
        print("Running 1x TPU Benchmark in isolated process...")
        subprocess.run([sys.executable, __file__, "--mode", "1x"], check=True)

        print("\n" + "=" * 70)
        print("Running 8x TPU Distributed Benchmark in isolated process...")
        subprocess.run([sys.executable, __file__, "--mode", "8x"], check=True)

        with open("/tmp/bench_1x.json") as f:
            res_1x = json.load(f)
        with open("/tmp/bench_8x.json") as f:
            res_8x = json.load(f)

        speedup = res_8x["tokens_per_sec"] / res_1x["tokens_per_sec"]
        scaling_eff = (speedup / 8.0) * 100.0
        print("\n" + "=" * 70)
        print(f"1x TPU: {res_1x['step_time_ms']:.1f} ms/step | {res_1x['tokens_per_sec']:,.0f} tok/s | MFU: {res_1x['mfu_percent']:.1f}%")
        print(f"8x TPU: {res_8x['step_time_ms']:.1f} ms/step | {res_8x['tokens_per_sec']:,.0f} tok/s | MFU: {res_8x['mfu_percent']:.1f}%")
        print(f"Scaling Speedup: {speedup:.2f}x / 8.0x ({scaling_eff:.1f}% parallel scaling efficiency)")
        print("=" * 70)

        out_file = "artifacts/phase6/mfu_scaling.json"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump({
                "single_chip": res_1x,
                "distributed_8x": res_8x,
                "speedup": speedup,
                "scaling_efficiency_percent": scaling_eff,
            }, f, indent=2)
        print(f"Saved scaling results to {out_file}")


if __name__ == "__main__":
    main()
