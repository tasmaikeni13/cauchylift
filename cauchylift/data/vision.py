from __future__ import annotations

import os
from typing import Any
import torch

from .dataset import StreamCursor


class VisionDataset:
    """Deterministic image batch provider for CIFAR-10.

    Uses pre-extracted tensors in data_cache/cifar10_tensors.pt.
    Normalizes images to standard channel mean/std in FP32.
    """

    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def __init__(
        self,
        split: str = "train",
        batch_size: int = 64,
        data_path: str = "data_cache/cifar10_tensors.pt",
        seed: int = 42,
    ) -> None:
        self.split = split
        self.batch_size = batch_size
        self.seed = seed

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CIFAR-10 tensor cache not found at {data_path}")

        cached = torch.load(data_path, weights_only=True)
        if split == "train":
            self.images = cached["train_imgs"]
            self.labels = cached["train_labels"]
        elif split in ("valid", "test"):
            self.images = cached["test_imgs"]
            self.labels = cached["test_labels"]
        else:
            raise ValueError(f"Unknown split {split}. Must be 'train' or 'valid'/'test'.")

        self.num_samples = len(self.labels)
        self.cursor_pos = 0
        self.samples_seen = 0

        # Deterministic shuffle indices
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(self.num_samples, generator=g)

    def get_cursor(self) -> StreamCursor:
        return StreamCursor(
            split=self.split,
            shard_idx=0,
            doc_idx=0,
            token_offset=self.cursor_pos,
            tokens_seen=self.samples_seen,
            buffer=[],
        )

    def next_batch(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
        needed = self.batch_size
        if self.cursor_pos + needed > self.num_samples:
            # Reshuffle deterministically upon wrap
            self.cursor_pos = 0

        idx = self.perm[self.cursor_pos : self.cursor_pos + needed]
        self.cursor_pos += needed

        batch_imgs = self.images[idx].to(torch.float32) / 255.0
        # Normalize
        batch_imgs = (batch_imgs - self.MEAN) / self.STD
        batch_imgs = batch_imgs.to(device)

        batch_labels = self.labels[idx].to(device)
        self.samples_seen += needed
        return batch_imgs, batch_labels, needed
