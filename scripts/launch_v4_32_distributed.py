#!/usr/bin/env python3
"""Launch distributed training across all 16 chips (4 hosts) of a Google Cloud TPU v4-32 slice.

Coordinates simultaneous job launching across:
- Worker 0: 10.130.0.10 (4 chips)
- Worker 1: 10.130.0.13 (4 chips)
- Worker 2: 10.130.0.12 (4 chips)
- Worker 3: 10.130.0.11 (4 chips)
Total: 16 TPU v4 chips (32 TensorCores) across a 2x2x4 3D Torus ICI interconnect.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

WORKER_IPS = [
    "10.130.0.10",
    "10.130.0.13",
    "10.130.0.12",
    "10.130.0.11",
]

SSH_KEY = os.path.expanduser("~/.ssh/google_compute_engine")


def check_ssh_connectivity() -> bool:
    print("[CHECK] Verifying passwordless SSH across all 4 worker hosts...")
    for i, ip in enumerate(WORKER_IPS):
        cmd = [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            ip,
            "hostname",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [FAIL] Worker {i} ({ip}) unreachable: {res.stderr.strip()}")
            return False
        print(f"  [OK] Worker {i} ({ip}): {res.stdout.strip()}")
    return True


def sync_codebase_to_workers() -> None:
    print("\n[SYNC] Syncing repository files to workers 1, 2, 3...")
    repo_dir = "/home/tas_ken_rt25/cauchylift"
    for ip in WORKER_IPS[1:]:
        cmd = [
            "rsync",
            "-az",
            "-e", f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no",
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", "runs",
            f"{repo_dir}/",
            f"{ip}:{repo_dir}/",
        ]
        subprocess.run(cmd, check=True)
        print(f"  [SYNCED] {ip}:{repo_dir}")


def launch_distributed_command(command_args: list[str]) -> int:
    workers_str = ",".join(WORKER_IPS)
    processes = []

    print("\n" + "=" * 80)
    print("LAUNCHING TPU V4-32 DISTRIBUTED RUN ACROSS ALL 16 CHIPS")
    print(f"Hosts: {len(WORKER_IPS)} nodes (4 chips per node = 16 chips total)")
    print(f"Command: {' '.join(command_args)}")
    print("=" * 80 + "\n")

    for rank, ip in enumerate(WORKER_IPS):
        env_vars = (
            f"TPU_ACCELERATOR_TYPE=v4-32 "
            f"TPU_PROCESS_BOUNDS=1,1,4 "
            f"TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 "
            f"TPU_WORKER_HOSTNAMES={workers_str} "
            f"TPU_WORKER_ID={rank} "
            f"PYTHONPATH=/home/tas_ken_rt25/cauchylift "
        )
        remote_cmd = (
            f"cd /home/tas_ken_rt25/cauchylift && "
            f"{env_vars} /home/tas_ken_rt25/venv/bin/python {' '.join(command_args)}"
        )
        ssh_cmd = [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            ip,
            remote_cmd,
        ]
        p = subprocess.Popen(
            ssh_cmd,
            stdout=sys.stdout if rank == 0 else subprocess.DEVNULL,
            stderr=sys.stderr if rank == 0 else subprocess.DEVNULL,
        )
        processes.append((rank, ip, p))

    # Wait for all workers to finish
    exit_codes = [p.wait() for _, _, p in processes]
    all_success = all(code == 0 for code in exit_codes)
    if all_success:
        print("\n" + "=" * 80)
        print("TPU v4-32 distributed run across all 16 chips completed successfully!")
        print("=" * 80)
        return 0
    else:
        print(f"\n[ERROR] Worker exit codes: {exit_codes}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training across all 16 chips of TPU v4-32 slice")
    parser.add_argument("--sync", action="store_true", help="Sync repo files to all workers before launching")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Python script and arguments to run")
    args = parser.parse_args()

    cmd = args.cmd if args.cmd else ["scripts/train_distributed.py"]
    if cmd[0] == "--":
        cmd = cmd[1:]

    if not check_ssh_connectivity():
        sys.exit(1)

    if args.sync:
        sync_codebase_to_workers()

    sys.exit(launch_distributed_command(cmd))


if __name__ == "__main__":
    main()
