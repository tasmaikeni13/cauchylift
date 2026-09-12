#!/usr/bin/env python3
"""Benchmark throughput and MFU scaling on Google Cloud TPU (v4 and v6e)."""

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

# Hardware peak BF16 TFLOPS specifications
PEAK_TFLOPS_TPU_V4 = 275.0   # Dense BF16 peak per TPU v4 chip (137.5 TFLOPS per TensorCore)
PEAK_TFLOPS_TPU_V6E = 460.0  # Dense BF16 peak per TPU v6e chip


def get_peak_tflops_per_chip() -> float:
    """Detect TPU generation and return dense BF16 peak TFLOPS."""
    try:
        from torch_xla._internal import tpu
        tpu_type = tpu.get_tpu_type()
        if "v4" in tpu_type:
            return PEAK_TFLOPS_TPU_V4
        elif "v6" in tpu_type:
            return PEAK_TFLOPS_TPU_V6E
    except Exception:
        pass
    # Fallback to TPU v4 for this server
    return PEAK_TFLOPS_TPU_V4


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
    peak_tflops = get_peak_tflops_per_chip()

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
    mfu = (tflops / peak_tflops) * 100.0

    res = {
        "devices": 1,
        "batch_size_per_chip": batch_size,
        "global_batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "step_time_ms": avg_time * 1000.0,
        "tokens_per_sec": tok_per_sec,
        "achieved_tflops": tflops,
        "peak_tflops_per_chip": peak_tflops,
        "mfu_percent": mfu,
    }
    print(f"1x TPU: {res['step_time_ms']:.1f} ms/step | {res['tokens_per_sec']:,.0f} tok/s | {res['achieved_tflops']:.1f} TFLOPs | MFU: {res['mfu_percent']:.1f}%")
    with open("/tmp/bench_1x.json", "w") as f:
        json.dump(res, f)


def _bench_dist_rank(index: int):
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
    peak_tflops = get_peak_tflops_per_chip()

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
    total_peak = peak_tflops * world_size
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
            "peak_tflops_per_chip": peak_tflops,
            "mfu_percent": mfu,
        }
        print(f"{world_size}x TPU: {res['step_time_ms']:.1f} ms/step | {res['tokens_per_sec']:,.0f} tok/s | {res['achieved_tflops']:.1f} TFLOPs | MFU: {res['mfu_percent']:.1f}%")
        with open("/tmp/bench_dist.json", "w") as f:
            json.dump(res, f)


def run_dist():
    xmp.spawn(_bench_dist_rank, args=())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["1x", "dist", "all"], default="all")
    args = parser.parse_args()

    if args.mode == "1x":
        run_1x(batch_size=4, seq_len=2048)
    elif args.mode == "dist":
        run_dist()
    elif args.mode == "all":
        print("=" * 70)
        print("Running 1x TPU Benchmark in isolated process...")
        subprocess.run([sys.executable, __file__, "--mode", "1x"], check=True)

        print("\n" + "=" * 70)
        print("Running Multi-TPU Distributed Benchmark in isolated process...")
        subprocess.run([sys.executable, __file__, "--mode", "dist"], check=True)

        with open("/tmp/bench_1x.json") as f:
            res_1x = json.load(f)
        with open("/tmp/bench_dist.json") as f:
            res_dist = json.load(f)

        world_size = res_dist["devices"]
        speedup = res_dist["tokens_per_sec"] / res_1x["tokens_per_sec"]
        scaling_eff = (speedup / float(world_size)) * 100.0
        print("\n" + "=" * 70)
        print(f"1x TPU:          {res_1x['step_time_ms']:.1f} ms/step | {res_1x['tokens_per_sec']:,.0f} tok/s | MFU: {res_1x['mfu_percent']:.1f}%")
        print(f"{world_size}x TPU Distributed: {res_dist['step_time_ms']:.1f} ms/step | {res_dist['tokens_per_sec']:,.0f} tok/s | MFU: {res_dist['mfu_percent']:.1f}%")
        print(f"Scaling Speedup: {speedup:.2f}x / {world_size:.1f}x ({scaling_eff:.1f}% parallel scaling efficiency)")
        print("=" * 70)

        out_file = "artifacts/phase6/mfu_scaling.json"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump({
                "single_chip": res_1x,
                f"distributed_{world_size}x": res_dist,
                "speedup": speedup,
                "scaling_efficiency_percent": scaling_eff,
            }, f, indent=2)
        print(f"Saved scaling results to {out_file}")


if __name__ == "__main__":
    main()
