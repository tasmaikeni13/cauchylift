import os
import time
import numpy as np
import pyarrow.parquet as pq
import tiktoken

def main():
    parquet_path = "/root/cauchylift/data/fineweb_edu/000_00000.parquet"
    out_dir = "/root/cauchylift/data/fineweb_edu"
    train_bin = os.path.join(out_dir, "train_tokens_350m.bin")
    val_bin = os.path.join(out_dir, "val_tokens.bin")

    target_train_tokens = 355_000_000  # slightly more than 350M to guarantee coverage
    target_val_tokens = 10_000_000

    print("Reading parquet table...")
    t0 = time.time()
    table = pq.read_table(parquet_path, columns=["text"])
    print(f"Loaded {table.num_rows} documents in {time.time() - t0:.2f}s")

    enc = tiktoken.get_encoding("gpt2")
    eos_id = 50256

    print("Tokenizing train split...")
    t0 = time.time()
    train_tokens = []
    doc_idx = 0
    total_tokens = 0

    while doc_idx < table.num_rows and total_tokens < target_train_tokens:
        batch_end = min(doc_idx + 10000, table.num_rows)
        texts = [table["text"][i].as_py() for i in range(doc_idx, batch_end)]
        batch_enc = enc.encode_batch(texts, num_threads=16)
        for doc in batch_enc:
            train_tokens.extend(doc)
            train_tokens.append(eos_id)
        total_tokens = len(train_tokens)
        doc_idx = batch_end
        print(f"Doc {doc_idx}/{table.num_rows}: {total_tokens:,} train tokens ({total_tokens / (time.time() - t0):,.0f} tok/s)")

    train_arr = np.array(train_tokens[:target_train_tokens], dtype=np.uint16)
    train_arr.tofile(train_bin)
    print(f"Saved {len(train_arr):,} train tokens to {train_bin} ({os.path.getsize(train_bin) / 1e6:.1f} MB)")

    print("Tokenizing validation split...")
    val_tokens = []
    val_start_doc = doc_idx
    while doc_idx < table.num_rows and len(val_tokens) < target_val_tokens:
        batch_end = min(doc_idx + 5000, table.num_rows)
        texts = [table["text"][i].as_py() for i in range(doc_idx, batch_end)]
        batch_enc = enc.encode_batch(texts, num_threads=16)
        for doc in batch_enc:
            val_tokens.extend(doc)
            val_tokens.append(eos_id)
        doc_idx = batch_end

    val_arr = np.array(val_tokens[:target_val_tokens], dtype=np.uint16)
    val_arr.tofile(val_bin)
    print(f"Saved {len(val_arr):,} val tokens to {val_bin} ({os.path.getsize(val_bin) / 1e6:.1f} MB)")
    print(f"Finished preparation! Train docs: 0 to {val_start_doc}, Val docs: {val_start_doc} to {doc_idx}. Non-overlapping partitions!")

if __name__ == "__main__":
    main()
