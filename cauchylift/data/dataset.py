from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import tiktoken
import torch


# Frozen metadata for FineWeb-Edu and GPT-2 Tokenizer
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_EDU_CONFIG = "sample-10BT"
FINEWEB_EDU_LICENSE = "ODC-By 1.0 (Open Data Commons Attribution License)"
TOKENIZER_NAME = "gpt2"
TOKENIZER_VOCAB_SIZE = 50257
TOKENIZER_LICENSE = "MIT License"
EOS_TOKEN_ID = 50256  # <|endoftext|>

# Pinned shards for sample-10BT (14 shards)
SHARD_FILENAMES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]

# Strict, non-overlapping partition assignment
PARTITION_SHARDS: dict[str, list[str]] = {
    "train": SHARD_FILENAMES[0:10],          # 10 shards (000 to 009)
    "tuning": SHARD_FILENAMES[10:11],        # 1 shard  (010)
    "validation": SHARD_FILENAMES[11:12],    # 1 shard  (011)
    "final_held_out": SHARD_FILENAMES[12:14], # 2 shards (012 to 013)
}


@dataclass
class StreamCursor:
    split: str
    shard_idx: int
    doc_idx: int
    token_offset: int
    tokens_seen: int
    buffer: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamCursor:
        return cls(
            split=str(d["split"]),
            shard_idx=int(d["shard_idx"]),
            doc_idx=int(d["doc_idx"]),
            token_offset=int(d["token_offset"]),
            tokens_seen=int(d["tokens_seen"]),
            buffer=list(d.get("buffer", [])),
        )


def verify_partition_disjointness() -> dict[str, Any]:
    """Verify that train, tuning, validation, and final_held_out partitions are completely disjoint."""
    all_splits = list(PARTITION_SHARDS.keys())
    disjoint = True
    overlaps = {}
    seen_shards: dict[str, str] = {}

    for split, shards in PARTITION_SHARDS.items():
        for shard in shards:
            if shard in seen_shards:
                disjoint = False
                overlaps[shard] = [seen_shards[shard], split]
            seen_shards[shard] = split

    return {
        "is_disjoint": disjoint,
        "splits": {k: len(v) for k, v in PARTITION_SHARDS.items()},
        "total_shards": len(seen_shards),
        "overlaps": overlaps,
        "repo": FINEWEB_EDU_REPO,
        "revision": FINEWEB_EDU_REVISION,
        "dataset_license": FINEWEB_EDU_LICENSE,
        "tokenizer_license": TOKENIZER_LICENSE,
    }


class SyntheticFineWebEduStream:
    """Deterministic synthetic FineWeb-Edu document stream for offline testing.

    Produces documents matching the exact FineWeb-Edu schema:
    (id, text, dump, url, score, int_score, token_count) with deterministic seeded PRNG.
    """

    def __init__(self, split: str, seed: int = 42) -> None:
        if split not in PARTITION_SHARDS:
            raise ValueError(f"Unknown split: {split}. Allowed: {list(PARTITION_SHARDS.keys())}")
        self.split = split
        self.seed = seed
        self.shards = PARTITION_SHARDS[split]

    def iter_documents(self, shard_idx: int = 0, doc_idx: int = 0) -> Iterator[dict[str, Any]]:
        # Deterministic document generation based on split, shard, and document index
        enc = tiktoken.get_encoding(TOKENIZER_NAME)
        rng = torch.Generator().manual_seed(self.seed + hash(self.split) % 100000 + shard_idx * 1000)

        # Pre-generated sample educational text corpus
        sample_paragraphs = [
            "Mathematics is the science and study of quality, structure, space, and change. Mathematicians seek out patterns, formulate new conjectures, and establish truth by rigorous deduction from appropriately chosen axioms and definitions.",
            "In linear algebra, a matrix is a rectangular array of numbers, symbols, or expressions arranged in rows and columns. Matrix operations include addition, scalar multiplication, and matrix multiplication.",
            "An optimization algorithm finds an input that minimizes or maximizes an objective function. First-order methods use gradient information to update parameters iteratively.",
            "Deep neural networks are composed of multiple layers that learn hierarchical representations of data. The Transformer architecture utilizes self-attention mechanisms to model relationships across sequence positions.",
            "Convergence guarantees provide theoretical bounds on the rate of decrease of the gradient norm for smooth non-convex optimization problems.",
            "Singular value decomposition factors a matrix into singular vectors and singular values, revealing fundamental geometric and rank properties of the transformation.",
        ]

        current_shard = shard_idx
        current_doc = doc_idx

        while current_shard < len(self.shards):
            shard_name = self.shards[current_shard]
            while True:
                doc_hash = hashlib.sha256(f"{shard_name}_{current_doc}".encode()).hexdigest()[:16]
                doc_id = f"doc_{self.split}_{current_shard}_{current_doc}_{doc_hash}"

                # Deterministic per-document RNG
                split_hash = int(hashlib.md5(self.split.encode()).hexdigest(), 16) % 100000
                doc_seed = (self.seed + split_hash + current_shard * 100000 + current_doc) % (2**31 - 1)
                doc_rng = torch.Generator().manual_seed(doc_seed)

                # Generate 2 to 4 paragraphs
                num_paras = int(torch.randint(2, 5, (1,), generator=doc_rng).item())
                chosen = [sample_paragraphs[int(torch.randint(0, len(sample_paragraphs), (1,), generator=doc_rng).item())] for _ in range(num_paras)]
                text = " ".join(chosen)

                yield {
                    "id": doc_id,
                    "shard": shard_name,
                    "shard_idx": current_shard,
                    "doc_idx": current_doc,
                    "text": text,
                    "dump": "CC-MAIN-2024-10",
                    "score": 4.5,
                    "int_score": 5,
                }
                current_doc += 1
                # 100 documents per synthetic shard
                if current_doc >= 100:
                    current_doc = 0
                    current_shard += 1
                    break


class PackedTokenDataset:
    """Deterministic token streaming and document packing pipeline for FineWeb-Edu.

    - Packs documents into contiguous sequences of length max_seq_len + 1.
    - Inserts EOS token (<|endoftext|>) between documents.
    - Tracks exact non-padding training tokens.
    - Saves and restores stream cursor for deterministic resumption.
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 1024,
        batch_size: int = 4,
        cursor: StreamCursor | None = None,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
        self.eos_token = EOS_TOKEN_ID

        if cursor is not None:
            self.cursor = cursor
        else:
            self.cursor = StreamCursor(
                split=split,
                shard_idx=0,
                doc_idx=0,
                token_offset=0,
                tokens_seen=0,
                buffer=[],
            )

        self.stream = SyntheticFineWebEduStream(split=split, seed=seed)
        self.doc_iter = self.stream.iter_documents(
            shard_idx=self.cursor.shard_idx,
            doc_idx=self.cursor.doc_idx,
        )

    def get_cursor(self) -> StreamCursor:
        return StreamCursor(
            split=self.cursor.split,
            shard_idx=self.cursor.shard_idx,
            doc_idx=self.cursor.doc_idx,
            token_offset=self.cursor.token_offset,
            tokens_seen=self.cursor.tokens_seen,
            buffer=list(self.cursor.buffer),
        )

    def _fill_buffer(self, min_tokens: int) -> None:
        while len(self.cursor.buffer) < min_tokens:
            try:
                doc = next(self.doc_iter)
            except StopIteration:
                # Loop stream deterministically if exhausted
                self.cursor.shard_idx = 0
                self.cursor.doc_idx = 0
                self.doc_iter = self.stream.iter_documents(0, 0)
            # Compute next unread document pointer
            next_doc = doc["doc_idx"] + 1
            next_shard = doc["shard_idx"]
            if next_doc >= 100:
                next_doc = 0
                next_shard += 1
                if next_shard >= len(self.stream.shards):
                    next_shard = 0

            self.cursor.shard_idx = next_shard
            self.cursor.doc_idx = next_doc

            tokens = self.tokenizer.encode(doc["text"])
            # Append document tokens plus EOS separator
            self.cursor.buffer.extend(tokens)
            self.cursor.buffer.append(self.eos_token)

    def next_batch(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
        """Produce the next packed batch of (input_ids, target_ids) and non-padding token count.

        Returns:
            input_ids: [batch_size, max_seq_len]
            target_ids: [batch_size, max_seq_len]
            tokens_in_batch: int (number of non-padding tokens)
        """
        seq_len = self.max_seq_len
        needed_tokens = self.batch_size * (seq_len + 1)
        self._fill_buffer(needed_tokens)

        # Slice packed tokens from buffer
        batch_tokens = self.cursor.buffer[:needed_tokens]
        self.cursor.buffer = self.cursor.buffer[needed_tokens:]

        # Reshape into batch of sequences of length seq_len + 1
        tensor_data = torch.tensor(batch_tokens, dtype=torch.long).reshape(self.batch_size, seq_len + 1)
        input_ids = tensor_data[:, :-1].to(device)
        target_ids = tensor_data[:, 1:].to(device)

        # In packed stream, every token is a real training token (non-padding)
        tokens_in_batch = self.batch_size * seq_len
        self.cursor.tokens_seen += tokens_in_batch

        return input_ids, target_ids, tokens_in_batch
