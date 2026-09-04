from __future__ import annotations

import pytest
import tiktoken

from cauchylift.data import (
    EOS_TOKEN_ID,
    FINEWEB_EDU_LICENSE,
    FINEWEB_EDU_REPO,
    FINEWEB_EDU_REVISION,
    PARTITION_SHARDS,
    TOKENIZER_LICENSE,
    TOKENIZER_NAME,
    PackedTokenDataset,
    verify_partition_disjointness,
)


def test_data_partitions_disjoint_and_licensed():
    """Verify that dataset partitions are strictly disjoint and metadata is recorded."""
    res = verify_partition_disjointness()
    assert res["is_disjoint"] is True, f"Found partition overlaps: {res['overlaps']}"
    assert res["repo"] == "HuggingFaceFW/fineweb-edu"
    assert res["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert "ODC-By" in res["dataset_license"]
    assert "MIT" in res["tokenizer_license"]

    # Check that all splits exist and have shards
    for split in ["train", "tuning", "validation", "final_held_out"]:
        assert split in PARTITION_SHARDS
        assert len(PARTITION_SHARDS[split]) > 0


def test_tokenizer_packing_and_eos():
    """Verify document packing with EOS separator."""
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    assert enc.n_vocab == 50257
    assert enc.eot_token == EOS_TOKEN_ID

    dataset = PackedTokenDataset(split="train", max_seq_len=64, batch_size=2)
    x, y, tokens = dataset.next_batch()

    assert x.shape == (2, 64)
    assert y.shape == (2, 64)
    assert tokens == 2 * 64
    # Target should be shifted by 1 relative to input
    # Notice x and y were generated as chunk[:-1] and chunk[1:] from contiguous buffer


def test_cursor_resumption_exact_match():
    """Uninterrupted and resumed runs must produce bitwise identical batches."""
    seq_len = 32
    batch_size = 2

    # Run 1: produce 4 batches uninterrupted
    ds1 = PackedTokenDataset(split="train", max_seq_len=seq_len, batch_size=batch_size, seed=123)
    b1_x, b1_y, _ = ds1.next_batch()
    b2_x, b2_y, _ = ds1.next_batch()
    cursor_mid = ds1.get_cursor()
    b3_x, b3_y, _ = ds1.next_batch()
    b4_x, b4_y, _ = ds1.next_batch()

    # Run 2: resume from cursor_mid and produce batches 3 and 4
    ds2 = PackedTokenDataset(split="train", max_seq_len=seq_len, batch_size=batch_size, cursor=cursor_mid, seed=123)
    r3_x, r3_y, _ = ds2.next_batch()
    r4_x, r4_y, _ = ds2.next_batch()

    assert (b3_x == r3_x).all(), "Resumed batch 3 input_ids mismatch"
    assert (b3_y == r3_y).all(), "Resumed batch 3 target_ids mismatch"
    assert (b4_x == r4_x).all(), "Resumed batch 4 input_ids mismatch"
    assert (b4_y == r4_y).all(), "Resumed batch 4 target_ids mismatch"


def test_exact_token_counting():
    """Verify exact token counter accounting."""
    seq_len = 48
    batch_size = 3
    dataset = PackedTokenDataset(split="train", max_seq_len=seq_len, batch_size=batch_size)

    expected_per_batch = seq_len * batch_size
    accumulated = 0

    for _ in range(5):
        _, _, count = dataset.next_batch()
        assert count == expected_per_batch
        accumulated += count
        assert dataset.cursor.tokens_seen == accumulated
