#!/usr/bin/env python3
"""Tokenize real FineWeb-Edu parquet shards into binary token files for high-throughput TPU training."""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken


def tokenize_parquet_to_bin(
    parquet_path: pathlib.Path | str,
    output_bin_path: pathlib.Path | str,
    max_tokens: int | None = None,
    batch_size: int = 10000,
    num_threads: int = 64,
) -> int:
    parquet_path = pathlib.Path(parquet_path)
    output_bin_path = pathlib.Path(output_bin_path)
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[TOKENIZE] Starting tokenization: {parquet_path.name} -> {output_bin_path.name}")
    t0 = time.time()

    table = pq.read_table(parquet_path, columns=["text"])
    num_rows = table.num_rows
    print(f"  Loaded {num_rows:,} documents from {parquet_path.name} in {time.time() - t0:.2f}s")

    enc = tiktoken.get_encoding("gpt2")
    eos_id = 50256  # <|endoftext|>

    total_tokens = 0
    doc_idx = 0

    # Write incrementally to binary file to preserve RAM and provide streaming writes
    temp_bin = output_bin_path.with_suffix(".tmp")
    with open(temp_bin, "wb") as f_out:
        while doc_idx < num_rows:
            if max_tokens is not None and total_tokens >= max_tokens:
                break

            batch_end = min(doc_idx + batch_size, num_rows)
            texts = [table["text"][i].as_py() for i in range(doc_idx, batch_end)]
            batch_enc = enc.encode_batch(texts, num_threads=num_threads)

            batch_tokens = []
            for doc in batch_enc:
                batch_tokens.extend(doc)
                batch_tokens.append(eos_id)

            if max_tokens is not None and total_tokens + len(batch_tokens) > max_tokens:
                needed = max_tokens - total_tokens
                batch_tokens = batch_tokens[:needed]

            arr = np.array(batch_tokens, dtype=np.uint16)
            arr.tofile(f_out)

            total_tokens += len(batch_tokens)
            doc_idx = batch_end

            elapsed = time.time() - t0
            tok_s = total_tokens / max(0.01, elapsed)
            print(f"  Docs: {doc_idx:,}/{num_rows:,} | Tokens: {total_tokens:,} | Speed: {tok_s:,.0f} tok/s | Elapsed: {elapsed:.1f}s")

            if max_tokens is not None and total_tokens >= max_tokens:
                break

    temp_bin.rename(output_bin_path)
    total_time = time.time() - t0
    file_size_mb = output_bin_path.stat().st_size / (1024 * 1024)
    print(f"[COMPLETE] Saved {total_tokens:,} tokens to {output_bin_path} ({file_size_mb:.1f} MB) in {total_time:.1f}s ({total_tokens / total_time:,.0f} tok/s)\n")
    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu Parquet to Binary")
    parser.add_argument("--data_dir", type=str, default="/home/tas_ken_rt25/cauchylift/data/fineweb_edu")
    parser.add_argument("--max_tuning_tokens", type=int, default=100_000_000, help="Max tokens for tuning partition")
    parser.add_argument("--max_val_tokens", type=int, default=20_000_000, help="Max tokens for val partition")
    parser.add_argument("--max_train_tokens", type=int, default=200_000_000, help="Max tokens for train partition")
    parser.add_argument("--num_threads", type=int, default=128)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)

    # 1. Tuning partition (shard 010)
    p_tuning = data_dir / "010_00000.parquet"
    out_tuning = data_dir / "tuning_tokens.bin"
    if p_tuning.exists():
        tokenize_parquet_to_bin(p_tuning, out_tuning, max_tokens=args.max_tuning_tokens, num_threads=args.num_threads)

    # 2. Validation partition (shard 011)
    p_val = data_dir / "011_00000.parquet"
    out_val = data_dir / "val_tokens.bin"
    if p_val.exists():
        tokenize_parquet_to_bin(p_val, out_val, max_tokens=args.max_val_tokens, num_threads=args.num_threads)

    # 3. Train partition (shard 000)
    p_train = data_dir / "000_00000.parquet"
    out_train = data_dir / "train_tokens.bin"
    if p_train.exists():
        tokenize_parquet_to_bin(p_train, out_train, max_tokens=args.max_train_tokens, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
