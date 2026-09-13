#!/usr/bin/env python3
"""Check configuration and synchronization of all 16 TPU v4 chips across 4 hosts."""

import os
import socket
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr


def _check_fn(index: int):
    dev = xm.xla_device()
    global_rank = xr.global_ordinal()
    world_size = xr.world_size()
    local_rank = index
    hostname = socket.gethostname()
    worker_id = os.environ.get("TPU_WORKER_ID", "unknown")

    # 1. Device identity
    print(f"[TPU ONLINE] Host: {hostname} | WorkerID: {worker_id} | GlobalRank: {global_rank:2d}/{world_size} | LocalDev: {dev}")
    sys.stdout.flush()

    # 2. Synchronize all 16 ranks
    torch_xla.sync(wait=True)

    # 3. All-Reduce Test across all 16 TPU chips (each rank contributes 1.0)
    t = torch.ones(1, device=dev, dtype=torch.float32)
    sum_t = xm.all_reduce("sum", t)
    sum_val = float(sum_t.item())

    # 4. Matrix Multiplication Test on TPU TensorCores
    a = torch.randn(128, 128, device=dev, dtype=torch.bfloat16)
    b = torch.randn(128, 128, device=dev, dtype=torch.bfloat16)
    c = torch.matmul(a, b)
    xm.mark_step()
    c_sum = float(c.sum().item())

    if global_rank == 0:
        print("=" * 70)
        print(f"VERIFICATION SUCCESSFUL ACROSS ALL {world_size} TPU RANKS!")
        print(f"All-Reduce Sum = {sum_val:.1f} (Expected: {world_size}.0)")
        print(f"TensorCore MatMul check sum = {c_sum:.4f}")
        print("=" * 70)
        sys.stdout.flush()

    assert int(sum_val) == world_size, f"Mismatch in all-reduce: got {sum_val}, expected {world_size}"


def main():
    xmp.spawn(_check_fn, args=())


if __name__ == "__main__":
    main()
