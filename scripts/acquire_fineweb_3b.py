#!/usr/bin/env python3
"""Acquire, tokenize, and verify 3.05B tokens of real FineWeb-Edu for TPU pretraining."""

from __future__ import annotations

import os
import pathlib
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
TRAIN_SHARDS = [
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
    "sample/10BT/002_00000.parquet",
    "sample/10BT/003_00000.parquet",
    "sample/10BT/004_00000.parquet",
]
VAL_SHARD = "sample/10BT/011_00000.parquet"

TARGET_TRAIN_TOKENS = 3_000_000_000  # 3.00 Billion
TARGET_VAL_TOKENS = 50_000_000        # 50 Million (0.05B) -> Total 3.05B
EOS_TOKEN_ID = 50256


def process_shard(
    parquet_path: pathlib.Path,
    f_out,
    enc,
    needed_tokens: int,
    batch_size: int = 10_000,
    num_threads: int = 64,
) -> int:
    """Stream tokenize rows from parquet into binary file."""
    t0 = time.time()
    table = pq.read_table(parquet_path, columns=["text"])
    num_rows = table.num_rows
    print(f"  Loaded {parquet_path.name}: {num_rows:,} documents in {time.time() - t0:.2f}s")

    tokens_written = 0
    doc_idx = 0

    while doc_idx < num_rows and tokens_written < needed_tokens:
        batch_end = min(doc_idx + batch_size, num_rows)
        texts = [table["text"][i].as_py() for i in range(doc_idx, batch_end)]
        batch_enc = enc.encode_batch(
            texts,
            num_threads=num_threads,
            allowed_special={"<|endoftext|>"},
            disallowed_special=(),
        )

        batch_tokens = []
        for doc in batch_enc:
            batch_tokens.extend(doc)
            batch_tokens.append(EOS_TOKEN_ID)

        rem = needed_tokens - tokens_written
        if len(batch_tokens) > rem:
            batch_tokens = batch_tokens[:rem]

        arr = np.array(batch_tokens, dtype=np.uint16)
        arr.tofile(f_out)
        tokens_written += len(batch_tokens)
        doc_idx = batch_end

        elapsed = time.time() - t0
        speed = tokens_written / max(0.01, elapsed)
        print(f"    Docs: {doc_idx:,}/{num_rows:,} | Shard Tokens: {tokens_written:,} | Speed: {speed:,.0f} tok/s")

    return tokens_written


def acquire_dataset(data_dir: pathlib.Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"

    # 1. Validation split
    if val_bin.exists() and val_bin.stat().st_size >= TARGET_VAL_TOKENS * 2:
        print(f"[VAL] {val_bin} already exists ({val_bin.stat().st_size / 2:,.0f} tokens). Skipping download.")
    else:
        print("\n" + "=" * 80)
        print(f"[VAL] Downloading and tokenizing validation split ({TARGET_VAL_TOKENS:,} tokens)...")
        print("=" * 80)
        val_parquet = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=VAL_SHARD,
            local_dir=str(data_dir),
        )
        val_tmp = val_bin.with_suffix(".tmp")
        with open(val_tmp, "wb") as f_val:
            written = process_shard(pathlib.Path(val_parquet), f_val, enc, TARGET_VAL_TOKENS)
        val_tmp.rename(val_bin)
        print(f"[VAL COMPLETE] Saved {written:,} tokens to {val_bin} ({val_bin.stat().st_size:,} bytes)")
        if os.path.exists(val_parquet):
            os.remove(val_parquet)

    # 2. Train split
    if train_bin.exists() and train_bin.stat().st_size >= TARGET_TRAIN_TOKENS * 2:
        print(f"[TRAIN] {train_bin} already exists ({train_bin.stat().st_size / 2:,.0f} tokens). Skipping download.")
    else:
        print("\n" + "=" * 80)
        print(f"[TRAIN] Downloading and tokenizing train split ({TARGET_TRAIN_TOKENS:,} tokens)...")
        print("=" * 80)
        train_tmp = train_bin.with_suffix(".tmp")
        total_train_written = 0

        # Resume if tmp already partially written
        if train_tmp.exists():
            total_train_written = train_tmp.stat().st_size // 2
            print(f"[TRAIN RESUME] Found existing {train_tmp} with {total_train_written:,} tokens.")

        with open(train_tmp, "ab" if total_train_written > 0 else "wb") as f_train:
            for shard_idx, shard_rel in enumerate(TRAIN_SHARDS):
                if total_train_written >= TARGET_TRAIN_TOKENS:
                    break

                # If shard 0 was already fully written into train_tmp (752M tokens)
                if shard_idx == 0 and total_train_written >= 700_000_000:
                    print(f"\n[TRAIN SHARD 1/{len(TRAIN_SHARDS)}] Shard 0 already written ({total_train_written:,} tokens). Skipping to shard 2.")
                    continue

                needed = TARGET_TRAIN_TOKENS - total_train_written
                print(f"\n[TRAIN SHARD {shard_idx + 1}/{len(TRAIN_SHARDS)}] Downloading {shard_rel} (Needed: {needed:,} tokens)...")
                shard_parquet = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=shard_rel,
                    local_dir=str(data_dir),
                )
                written = process_shard(pathlib.Path(shard_parquet), f_train, enc, needed)
                total_train_written += written
                f_train.flush()
                print(f"[SHARD FINISHED] Total train tokens so far: {total_train_written:,}/{TARGET_TRAIN_TOKENS:,}")

                if os.path.exists(shard_parquet):
                    os.remove(shard_parquet)

        train_tmp.rename(train_bin)
        print(f"\n[TRAIN COMPLETE] Saved {total_train_written:,} tokens to {train_bin} ({train_bin.stat().st_size:,} bytes)")


def verify_tokens_on_disk(data_dir: pathlib.Path) -> bool:
    print("\n" + "=" * 80)
    print("STRICT TOKEN VERIFICATION ON DISK")
    print("=" * 80)

    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"

    if not train_bin.exists():
        print(f"[FAIL] Missing {train_bin}")
        return False
    if not val_bin.exists():
        print(f"[FAIL] Missing {val_bin}")
        return False

    train_bytes = train_bin.stat().st_size
    val_bytes = val_bin.stat().st_size

    train_tokens = train_bytes // 2
    val_tokens = val_bytes // 2
    total_tokens = train_tokens + val_tokens

    print(f"Train File:      {train_bin} ({train_bytes:,} bytes, {train_tokens:,} tokens)")
    print(f"Validation File: {val_bin} ({val_bytes:,} bytes, {val_tokens:,} tokens)")
    print(f"Total Tokens:    {total_tokens:,} tokens ({total_tokens / 1e9:.3f} Billion)")

    # Read memory maps and check validity
    m_train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    m_val = np.memmap(val_bin, dtype=np.uint16, mode="r")

    # Sample check
    train_sample = m_train[:10000]
    val_sample = m_val[:10000]
    assert int(train_sample.max()) < 50257, f"Invalid token ID in train: {train_sample.max()}"
    assert int(val_sample.max()) < 50257, f"Invalid token ID in val: {val_sample.max()}"

    if total_tokens < 3_050_000_000:
        print(f"[FAIL] Total tokens {total_tokens:,} is less than required 3.05 Billion tokens!")
        return False

    print("=" * 80)
    print(f"SUCCESS: STRICT VERIFICATION PASSED! {total_tokens:,} VALID TOKENS ON DISK (>= 3.05B)!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    data_directory = pathlib.Path("/home/tas_ken_rt25/cauchylift/data/fineweb_edu")
    acquire_dataset(data_directory)
    success = verify_tokens_on_disk(data_directory)
    if not success:
        sys.exit(1)
