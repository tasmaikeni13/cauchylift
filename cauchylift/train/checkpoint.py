from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from cauchylift.data import StreamCursor


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_environment_fingerprint() -> dict[str, Any]:
    """Capture environment fingerprint."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_name": device_name,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Any,
    cursor: StreamCursor,
    step: int,
    tokens_seen: int,
    config: dict[str, Any],
) -> None:
    """Save an atomic checkpoint to disk using a temporary file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"

    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "cursor": cursor.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "source_commit": get_git_commit(),
        "environment_fingerprint": get_environment_fingerprint(),
    }

    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load an atomic checkpoint and restore model, optimizer, scheduler, and RNG states."""
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    if scheduler is not None and checkpoint_data.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    if scaler is not None and checkpoint_data.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

    # Restore RNG states
    rng_states = checkpoint_data.get("rng_states", {})
    if "torch" in rng_states and rng_states["torch"] is not None:
        torch_rng = rng_states["torch"]
        if isinstance(torch_rng, torch.Tensor):
            torch_rng = torch_rng.cpu()
        torch.set_rng_state(torch_rng)
    if torch.cuda.is_available() and "cuda" in rng_states and rng_states["cuda"] is not None:
        cuda_rng = rng_states["cuda"]
        if isinstance(cuda_rng, torch.Tensor):
            cuda_rng = cuda_rng.cpu()
        torch.cuda.set_rng_state(cuda_rng)
    if "python" in rng_states and rng_states["python"] is not None:
        random.setstate(rng_states["python"])
    if "numpy" in rng_states and rng_states["numpy"] is not None:
        np.random.set_state(rng_states["numpy"])

    cursor = StreamCursor.from_dict(checkpoint_data["cursor"])
    step = int(checkpoint_data["step"])
    tokens_seen = int(checkpoint_data["tokens_seen"])

    return {
        "cursor": cursor,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": checkpoint_data.get("config", {}),
        "source_commit": checkpoint_data.get("source_commit", "unknown"),
        "environment_fingerprint": checkpoint_data.get("environment_fingerprint", {}),
    }
