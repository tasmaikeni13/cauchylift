import os
import sys
import time
import traceback
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from cauchylift.models.transformer import Transformer, TransformerConfig
from cauchylift.baselines.muon import Muon

def _run_rank(index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    dev = xm.xla_device()

    try:
        cfg = TransformerConfig(
            vocab_size=50257,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            intermediate_dim=2048,
            max_seq_len=2048,
            activation="swiglu",
            norm_eps=1e-5,
            tied_embeddings=True,
            attention_backend="flash",
        )
        torch.manual_seed(42)
        model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
        opt = Muon(model.parameters(), lr=0.02, backend="auto")

        if rank == 0:
            print(f"Starting 10 steps of 125M with Muon across {world_size} TPU chips...", flush=True)

        for step in range(1, 11):
            t0 = time.perf_counter()
            opt.zero_grad()
            x = torch.randint(0, cfg.vocab_size, (4, 2048), device=dev)
            y = torch.randint(0, cfg.vocab_size, (4, 2048), device=dev)

            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss.backward()
            xm.reduce_gradients(opt)
            opt.step()
            torch_xla.sync()
            dur = time.perf_counter() - t0

            global_loss = xm.all_reduce("sum", loss.to(torch.float32)) / float(world_size)
            loss_val = float(global_loss.item())

            if rank == 0:
                print(f"Step {step:2d}/10 | Latency: {dur*1000:.1f}ms | Global Loss: {loss_val:.4f}", flush=True)

        xm.rendezvous("finish_125m_test")
        if rank == 0:
            print("Test completed successfully!", flush=True)
    except Exception as e:
        print(f"[ERROR on rank {rank}]: {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    xmp.spawn(_run_rank, args=())
    os._exit(0)
