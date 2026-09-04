from __future__ import annotations

import os
from typing import Any
import torch

from .dataset import StreamCursor


class WikiTextDataset:
    """Deterministic token streaming for WikiText-103.

    Uses pre-tokenized tensors in data_cache/wikitext103_tokens.pt.
    Tracks exact non-padding tokens.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        batch_size: int = 8,
        data_path: str = "data_cache/wikitext103_tokens.pt",
        seed: int = 42,
    ) -> None:
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"WikiText-103 token cache not found at {data_path}")

        cached = torch.load(data_path, weights_only=True)
        if split not in cached:
            raise ValueError(f"Unknown split: {split}. Available: {list(cached.keys())}")

        self.tokens = cached[split].to(torch.long)
        self.total_tokens = len(self.tokens)
        self.cursor_pos = 0
        self.tokens_seen = 0

    def get_cursor(self) -> StreamCursor:
        return StreamCursor(
            split=self.split,
            shard_idx=0,
            doc_idx=0,
            token_offset=self.cursor_pos,
            tokens_seen=self.tokens_seen,
            buffer=[],
        )

    def next_batch(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
        needed = self.batch_size * (self.seq_len + 1)
        if self.cursor_pos + needed > self.total_tokens:
            self.cursor_pos = 0  # Wrap deterministically

        chunk = self.tokens[self.cursor_pos : self.cursor_pos + needed]
        self.cursor_pos += needed

        tensor_data = chunk.reshape(self.batch_size, self.seq_len + 1)
        input_ids = tensor_data[:, :-1].to(device)
        target_ids = tensor_data[:, 1:].to(device)

        step_tokens = self.batch_size * self.seq_len
        self.tokens_seen += step_tokens
        return input_ids, target_ids, step_tokens
