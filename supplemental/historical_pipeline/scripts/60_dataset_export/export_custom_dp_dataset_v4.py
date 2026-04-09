#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from dp_tokenizer_lib import load_dp_vocab, encode_dp


MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
DTYPE = np.uint16
SHARD_SIZE_DEFAULT = 100_000_000


def load_docs_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) and "text" in obj else obj
            if isinstance(text, str) and text:
                yield text


def write_shard(path: Path, token_ids: list[int]) -> None:
    arr = np.asarray(token_ids, dtype=DTYPE)
    header = np.zeros((HEADER_INTS,), dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(arr.size)

    with path.open("wb") as f:
        header.tofile(f)
        arr.tofile(f)


def flush_shard(output_dir: Path, split: str, shard_idx: int, buf: list[int]) -> None:
    path = output_dir / f"fineweb_{split}_{shard_idx:06d}.bin"
    write_shard(path, buf)
    size_bytes = path.stat().st_size
    print(f"[flush] wrote {path.name} tokens={len(buf):,} size_bytes={size_bytes:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-jsonl", type=Path, required=True)
    ap.add_argument("--vocab-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-val-docs", type=int, default=50_000)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE_DEFAULT)
    ap.add_argument("--max-train-shards", type=int, default=80)
    ap.add_argument("--append-eos", action="store_true")
    ap.add_argument("--eos-id", type=int, default=None)
    args = ap.parse_args()

    vocab = load_dp_vocab(args.vocab_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"docs_jsonl:       {args.docs_jsonl}")
    print(f"vocab_jsonl:      {args.vocab_jsonl}")
    print(f"output_dir:       {args.output_dir}")
    print(f"vocab_size:       {vocab.vocab_size}")
    print(f"num_val_docs:     {args.num_val_docs}")
    print(f"shard_size:       {args.shard_size}")
    print(f"eos_id:           {args.eos_id}")
    print(f"append_eos:       {args.append_eos}")
    print(f"max_train_shards: {args.max_train_shards}")

    if args.append_eos and args.eos_id is None:
        raise ValueError("--append-eos requires --eos-id")

    t0 = time.time()

    total_docs = 0
    val_docs = 0
    train_docs = 0
    total_tokens = 0

    val_buf: list[int] = []
    train_buf: list[int] = []
    train_shard_idx = 0
    val_written = False

    last_report = 0

    for text in load_docs_jsonl(args.docs_jsonl):
        total_docs += 1

        enc = encode_dp(text, vocab)
        ids = enc.ids
        if args.append_eos:
            ids = ids + [args.eos_id]

        total_tokens += len(ids)

        if total_docs <= args.num_val_docs:
            val_docs += 1
            val_buf.extend(ids)
        else:
            train_docs += 1
            if train_shard_idx >= args.max_train_shards:
                break

            remaining = ids
            while remaining:
                take = min(args.shard_size - len(train_buf), len(remaining))
                train_buf.extend(remaining[:take])
                remaining = remaining[take:]

                if len(train_buf) == args.shard_size:
                    flush_shard(args.output_dir, "train", train_shard_idx, train_buf)
                    train_shard_idx += 1
                    train_buf = []

                    if train_shard_idx >= args.max_train_shards:
                        break

            if train_shard_idx >= args.max_train_shards:
                break

        # flush val once it is complete and not yet written
        if not val_written and total_docs == args.num_val_docs:
            flush_shard(args.output_dir, "val", 0, val_buf)
            val_written = True
            val_buf = []

        if total_docs - last_report >= 1000:
            elapsed = max(time.time() - t0, 1e-9)
            tok_per_sec = total_tokens / elapsed
            docs_per_sec = total_docs / elapsed
            files = train_shard_idx + (1 if val_written else 0)
            print(
                f"docs={total_docs:,} "
                f"train_docs={train_docs:,} "
                f"val_docs={val_docs:,} "
                f"tokens={total_tokens:,} "
                f"files={files} "
                f"docs/sec={docs_per_sec:.1f} "
                f"tok/sec={tok_per_sec:,.1f}"
            )
            last_report = total_docs

    # final val flush if docs file ended before val flush
    if not val_written and val_buf:
        flush_shard(args.output_dir, "val", 0, val_buf)
        val_written = True

    # final partial train shard flush
    if train_buf and train_shard_idx < args.max_train_shards:
        flush_shard(args.output_dir, "train", train_shard_idx, train_buf)

    elapsed = time.time() - t0
    print()
    print("DONE")
    print(f"elapsed_sec:      {elapsed:,.2f}")
    print(f"docs_total:       {total_docs}")
    print(f"docs_val:         {val_docs}")
    print(f"docs_train:       {train_docs}")
    print(f"tokens_total:     {total_tokens}")
    print(f"files_train:      {min(train_shard_idx + (1 if train_buf else 0), args.max_train_shards)}")
    print(f"files_val:        {1 if val_written else 0}")
    print(f"stopped_early:    {train_shard_idx >= args.max_train_shards}")


if __name__ == "__main__":
    main()