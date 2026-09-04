from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cauchylift.data import PackedTokenDataset
from cauchylift.models import Transformer
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import (
    MI300X_BF16_PEAK_TFLOPS,
    MetricsLogger,
    StepRecord,
    compute_gradient_and_update_metrics,
)


@dataclass
class TrainingConfig:
    max_steps: int = 100
    batch_size: int = 4
    seq_len: int = 256
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bf16"  # "bf16" or "fp32"
    activation_checkpointing: bool = False
    eval_interval: int = 20
    eval_steps: int = 5
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_file: str = "metrics.jsonl"
    seed: int = 42


def get_cosine_lr(step: int, config: TrainingConfig) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / max(1, config.warmup_steps)
    if step > config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


class Trainer:
    """Unified training engine for CauchyLift and baseline optimizers.

    Enforces:
    - Identical model architecture
    - Identical batch semantics and token counter
    - BF16 training with FP32 sensitive reductions
    - Gradient accumulation and optional activation checkpointing
    - Exact optimizer-only timing via CUDA events
    - Atomic resumable checkpointing
    - Comprehensive structured metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataset: Any,
        val_dataset: Any | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = MetricsLogger(self.config.log_file)
        self.step = 0
        self.tokens_seen = 0
        self.start_wall_time = time.perf_counter()

    def evaluate(self) -> float:
        """Compute validation loss on val_dataset."""
        if self.val_dataset is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(self.config.eval_steps):
                x, y, _ = self.val_dataset.next_batch(device=self.config.device)
                use_bf16 = self.config.precision == "bf16" and self.config.device.startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    _, loss = self.model(x, y)
                total_loss += float(loss.item()) if loss is not None else 0.0

        self.model.train()
        return total_loss / max(1, self.config.eval_steps)

    def train_step(self) -> StepRecord:
        """Execute one full training step with gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        step_start_time = time.perf_counter()
        accum_loss = 0.0
        step_tokens = 0

        use_bf16 = self.config.precision == "bf16" and self.config.device.startswith("cuda")

        # 1. Forward and backward with gradient accumulation
        for micro_step in range(self.config.gradient_accumulation_steps):
            x, y, tokens_in_batch = self.train_dataset.next_batch(device=self.config.device)
            step_tokens += tokens_in_batch

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = self.model(
                    x, y,
                    activation_checkpointing=self.config.activation_checkpointing,
                )

            assert loss is not None
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            accum_loss += float(loss.item()) / self.config.gradient_accumulation_steps

        # 2. Update learning rate
        current_lr = get_cosine_lr(self.step, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr

        # 3. Snapshot parameters before update for metrics analysis
        params_before: dict[int, torch.Tensor] = {}
        for p in self.model.parameters():
            if p.grad is not None:
                params_before[id(p)] = p.detach().clone()

        # 4. Measure optimizer-only execution time accurately
        is_cuda = self.config.device.startswith("cuda")
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.optimizer.step()
            end_event.record()
            torch.cuda.synchronize()
            opt_time_sec = start_event.elapsed_time(end_event) / 1000.0
        else:
            opt_start = time.perf_counter()
            self.optimizer.step()
            opt_time_sec = time.perf_counter() - opt_start

        step_duration = time.perf_counter() - step_start_time
        self.tokens_seen += step_tokens
        self.step += 1

        # 5. Validation evaluation if scheduled
        val_loss = None
        if self.val_dataset is not None and self.step % self.config.eval_interval == 0:
            val_loss = self.evaluate()

        # 6. Compute metrics
        metric_data = compute_gradient_and_update_metrics(self.model, params_before, current_lr)
        is_spike = self.logger.check_loss_spike(accum_loss)

        throughput = step_tokens / max(1e-6, step_duration)
        peak_mem = torch.cuda.max_memory_allocated() if is_cuda else 0

        # FLOPs and MFU estimate
        if hasattr(self.model, "flops_per_token"):
            flops_per_token = self.model.flops_per_token()
        elif hasattr(self.model, "flops_per_example"):
            flops_per_token = self.model.flops_per_example()
        else:
            flops_per_token = 1e6
        achieved_tflops = (throughput * flops_per_token) / 1e12
        mfu_estimate = achieved_tflops / MI300X_BF16_PEAK_TFLOPS

        record = StepRecord(
            step=self.step,
            loss=accum_loss,
            val_loss=val_loss,
            lr=current_lr,
            tokens=self.tokens_seen,
            wall_time=time.perf_counter() - self.start_wall_time,
            step_time=step_duration,
            opt_time=opt_time_sec,
            throughput_tok_per_sec=throughput,
            peak_memory_bytes=peak_mem,
            grad_update_cosine=metric_data["grad_update_cosine"],
            row_concentration=metric_data["row_concentration"],
            col_concentration=metric_data["col_concentration"],
            effective_support=metric_data["effective_support"],
            update_stable_rank=metric_data["update_stable_rank"],
            min_denominator=metric_data["min_denominator"],
            max_denominator=metric_data["max_denominator"],
            boundary_frequency=metric_data["boundary_frequency"],
            loss_spike=is_spike,
            mfu_estimate=mfu_estimate,
        )

        self.logger.log_step(record)

        # 7. Checkpointing if scheduled
        if self.step % self.config.checkpoint_interval == 0:
            ckpt_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt")
            save_checkpoint(
                path=ckpt_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=None,
                scaler=None,
                cursor=self.train_dataset.get_cursor(),
                step=self.step,
                tokens_seen=self.tokens_seen,
                config=self.config.__dict__,
            )

        return record

    def train(self) -> list[StepRecord]:
        """Run training loop up to config.max_steps."""
        records = []
        while self.step < self.config.max_steps:
            rec = self.train_step()
            records.append(rec)

        # Final checkpoint
        final_ckpt = os.path.join(self.config.checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(
            path=final_ckpt,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            scaler=None,
            cursor=self.train_dataset.get_cursor(),
            step=self.step,
            tokens_seen=self.tokens_seen,
            config=self.config.__dict__,
        )
        return records

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training state from a saved atomic checkpoint."""
        info = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.config.device,
        )
        self.step = info["step"]
        self.tokens_seen = info["tokens_seen"]
        self.train_dataset.cursor = info["cursor"]
        self.train_dataset.doc_iter = self.train_dataset.stream.iter_documents(
            shard_idx=self.train_dataset.cursor.shard_idx,
            doc_idx=self.train_dataset.cursor.doc_idx,
        )
