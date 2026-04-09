from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_SHARD_SIZE_TOKENS = 100_000_000
DEFAULT_BATCH_DOCS = 512


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_docs(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["text"]


def write_datafile(path: Path, toks: np.ndarray) -> None:
    if toks.dtype != np.uint16:
        raise ValueError(f"expected uint16 tokens, got {toks.dtype}")
    if len(toks) >= 2**31:
        raise ValueError("token count too large for shard header")

    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())


class FastCustomDPTokenizer:
    """
    Same semantics as the original DP tokenizer:
    - minimize number of tokens
    - tie-break toward longer piece
    - fallback to byte pieces for uncovered chars

    Main speedups:
    - first-char pruning
    - cached byte fallback
    """

    def __init__(self, vocab_rows: list[dict]):
        self.rows = vocab_rows
        self.id_to_piece_map = {int(row["id"]): row["piece"] for row in vocab_rows}
        self.piece_to_id_map = {row["piece"]: int(row["id"]) for row in vocab_rows}

        if not self.piece_to_id_map:
            raise ValueError("empty vocab")

        self.vocab_size = max(self.id_to_piece_map) + 1
        if self.vocab_size > 65535:
            raise ValueError(f"vocab too large for uint16 shards: {self.vocab_size}")

        self.byte_piece_to_id: dict[int, int] = {}
        for b in range(256):
            piece = f"<0x{b:02X}>"
            pid = self.piece_to_id_map.get(piece)
            if pid is not None:
                self.byte_piece_to_id[b] = pid

        candidates_by_first_char: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        max_piece_len = 0

        for piece, pid in self.piece_to_id_map.items():
            if not piece:
                continue
            L = len(piece)
            max_piece_len = max(max_piece_len, L)
            candidates_by_first_char[piece[0]].append((piece, pid, L))

        for ch in candidates_by_first_char:
            candidates_by_first_char[ch].sort(key=lambda x: x[2], reverse=True)

        self.candidates_by_first_char = dict(candidates_by_first_char)
        self.max_piece_len = max_piece_len
        self._char_fallback_cache: dict[str, list[int]] = {}

    def _byte_fallback_ids_for_char(self, ch: str) -> list[int]:
        cached = self._char_fallback_cache.get(ch)
        if cached is not None:
            return cached

        out: list[int] = []
        for b in ch.encode("utf-8"):
            pid = self.byte_piece_to_id.get(b)
            if pid is None:
                raise ValueError(
                    f"missing byte fallback piece for char={repr(ch)} byte=<0x{b:02X}>"
                )
            out.append(pid)

        self._char_fallback_cache[ch] = out
        return out

    def encode_dp(self, text: str) -> list[int]:
        n = len(text)
        dp_cost = [math.inf] * (n + 1)
        dp_next: list[tuple[str, int | list[int], int] | None] = [None] * (n + 1)
        dp_cost[n] = 0

        for i in range(n - 1, -1, -1):
            best_cost = math.inf
            best_choice: tuple[str, int | list[int], int] | None = None

            first_char = text[i]
            candidates = self.candidates_by_first_char.get(first_char, ())
            if candidates:
                remaining = n - i
                for piece, pid, L in candidates:
                    if L > remaining:
                        continue
                    if text.startswith(piece, i):
                        cand_cost = 1 + dp_cost[i + L]
                        if cand_cost < best_cost:
                            best_cost = cand_cost
                            best_choice = ("piece", pid, L)
                        elif cand_cost == best_cost and best_choice is not None:
                            if L > best_choice[2]:
                                best_choice = ("piece", pid, L)

            fallback_ids = self._byte_fallback_ids_for_char(first_char)
            fallback_cost = len(fallback_ids) + dp_cost[i + 1]
            if fallback_cost < best_cost:
                best_cost = fallback_cost
                best_choice = ("bytes", fallback_ids, 1)
            elif fallback_cost == best_cost and best_choice is None:
                best_choice = ("bytes", fallback_ids, 1)

            dp_cost[i] = best_cost
            dp_next[i] = best_choice

        ids: list[int] = []
        i = 0
        while i < n:
            choice = dp_next[i]
            if choice is None:
                raise RuntimeError(f"DP reconstruction failed at position {i}")
            kind, payload, advance = choice
            if kind == "piece":
                ids.append(int(payload))
            else:
                ids.extend(payload)  # type: ignore[arg-type]
            i += advance

        return ids


_TOKENIZER: FastCustomDPTokenizer | None = None
_BOS_ID: int | None = None
_EOS_ID: int | None = None
_APPEND_EOS: bool = False


def _worker_init(vocab_rows: list[dict], bos_id: int | None, eos_id: int | None, append_eos: bool) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _APPEND_EOS
    _TOKENIZER = FastCustomDPTokenizer(vocab_rows)
    _BOS_ID = bos_id
    _EOS_ID = eos_id
    _APPEND_EOS = append_eos


def _tokenize_batch(batch: list[tuple[int, str]]) -> list[tuple[int, np.ndarray]]:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _APPEND_EOS
    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer not initialized")

    out: list[tuple[int, np.ndarray]] = []

    for doc_idx, text in batch:
        ids: list[int] = []
        if _BOS_ID is not None:
            ids.append(_BOS_ID)
        ids.extend(_TOKENIZER.encode_dp(text))
        if _APPEND_EOS:
            if _EOS_ID is None:
                raise ValueError("--append-eos was set but no --eos-id provided")
            ids.append(_EOS_ID)

        out.append((doc_idx, np.asarray(ids, dtype=np.uint16)))

    return out


def batched_docs(docs_jsonl: Path, batch_docs: int) -> Iterable[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for doc_idx, text in enumerate(iter_docs(docs_jsonl)):
        batch.append((doc_idx, text))
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def export_shards_parallel(
    *,
    docs_jsonl: Path,
    vocab_rows: list[dict],
    output_dir: Path,
    num_val_docs: int,
    shard_size_tokens: int,
    bos_id: int | None,
    eos_id: int | None,
    append_eos: bool,
    workers: int,
    batch_docs: int,
    max_shards: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    stats = {
        "docs_total": 0,
        "docs_val": 0,
        "docs_train": 0,
        "files_total": 0,
        "files_val": 0,
        "files_train": 0,
        "tokens_total": 0,
        "tokens_val": 0,
        "tokens_train": 0,
        "stopped_early": False,
        "max_shards": max_shards,
    }

    split = "val"
    shard_idx = {"val": 0, "train": 0}
    buf = np.empty((shard_size_tokens,), dtype=np.uint16)
    fill = 0
    stop_requested = False

    def flush(current_split: str) -> None:
        nonlocal fill, stop_requested

        if fill == 0:
            return

        if max_shards is not None and stats["files_total"] >= max_shards:
            stop_requested = True
            return

        out_path = output_dir / f"fineweb_{current_split}_{shard_idx[current_split]:06d}.bin"
        write_datafile(out_path, buf[:fill])

        print(
            f"[flush] wrote {out_path.name} tokens={fill:,} size_bytes={out_path.stat().st_size:,}",
            flush=True,
        )

        stats["files_total"] += 1
        stats[f"files_{current_split}"] += 1
        shard_idx[current_split] += 1
        fill = 0

        if max_shards is not None and stats["files_total"] >= max_shards:
            stop_requested = True

    start_t = time.time()
    next_log_docs = 1_000

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(vocab_rows, bos_id, eos_id, append_eos),
    ) as ex:
        for tokenized_batch in ex.map(_tokenize_batch, batched_docs(docs_jsonl, batch_docs), chunksize=1):
            if stop_requested:
                break

            for doc_idx, ids_np in tokenized_batch:
                if stop_requested:
                    break

                split_for_doc = "val" if doc_idx < num_val_docs else "train"
                if split_for_doc != split:
                    flush(split)
                    if stop_requested:
                        break
                    split = split_for_doc

                stats["docs_total"] += 1
                stats[f"docs_{split}"] += 1
                stats["tokens_total"] += len(ids_np)
                stats[f"tokens_{split}"] += len(ids_np)

                pos = 0
                while pos < len(ids_np):
                    remaining_capacity = shard_size_tokens - fill
                    take = min(remaining_capacity, len(ids_np) - pos)
                    buf[fill:fill + take] = ids_np[pos:pos + take]
                    fill += take
                    pos += take

                    if fill == shard_size_tokens:
                        flush(split)
                        if stop_requested:
                            break

                if stop_requested:
                    break

                if stats["docs_total"] >= next_log_docs:
                    elapsed = time.time() - start_t
                    docs_per_sec = stats["docs_total"] / max(elapsed, 1e-9)
                    toks_per_sec = stats["tokens_total"] / max(elapsed, 1e-9)
                    print(
                        f"docs={stats['docs_total']:,} "
                        f"train_docs={stats['docs_train']:,} "
                        f"val_docs={stats['docs_val']:,} "
                        f"tokens={stats['tokens_total']:,} "
                        f"files={stats['files_total']:,} "
                        f"docs/sec={docs_per_sec:,.1f} "
                        f"tok/sec={toks_per_sec:,.1f}",
                        flush=True,
                    )
                    if next_log_docs < 10_000:
                        next_log_docs += 1_000
                    else:
                        next_log_docs += 10_000

    if stop_requested:
        stats["stopped_early"] = True
        print(f"Reached max_shards={max_shards}, stopping early.", flush=True)
    else:
        flush(split)

    return stats


def build_manifest(
    *,
    output_dir: Path,
    vocab_path: Path,
    stats: dict,
    num_val_docs: int,
    shard_size_tokens: int,
    bos_id: int | None,
    eos_id: int | None,
    append_eos: bool,
    vocab_size: int,
    workers: int,
    batch_docs: int,
    max_shards: int | None,
) -> None:
    manifest = {
        "name": output_dir.name,
        "tokenizer_kind": "custom_dp_jsonl",
        "tokenizer_path": str(vocab_path),
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "append_eos": append_eos,
        "num_val_docs": num_val_docs,
        "shard_size_tokens": shard_size_tokens,
        "workers": workers,
        "batch_docs": batch_docs,
        "max_shards": max_shards,
        "stats": stats,
        "train_glob": str(output_dir / "fineweb_train_*.bin"),
        "val_glob": str(output_dir / "fineweb_val_*.bin"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Export FineWeb docs into custom DP-tokenized shards (multiprocess)")
    p.add_argument(
        "--docs-jsonl",
        default="data/docs_selected.jsonl",
        help="Path to docs_selected.jsonl",
    )
    p.add_argument(
        "--vocab-jsonl",
        required=True,
        help="Path to custom vocab JSONL",
    )
    p.add_argument(
        "--output-dir",
        default="data/datasets/fineweb10B_customdp1024",
        help="Output dataset directory",
    )
    p.add_argument(
        "--num-val-docs",
        type=int,
        default=DEFAULT_NUM_VAL_DOCS,
        help="Number of validation docs from the front of docs_selected.jsonl",
    )
    p.add_argument(
        "--shard-size-tokens",
        type=int,
        default=DEFAULT_SHARD_SIZE_TOKENS,
        help="Max tokens per shard",
    )
    p.add_argument(
        "--bos-id",
        type=int,
        default=None,
        help="Optional BOS token id to prepend to every doc",
    )
    p.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="Optional EOS token id",
    )
    p.add_argument(
        "--append-eos",
        action="store_true",
        help="Append EOS to every doc",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 8) - 1)),
        help="Number of worker processes",
    )
    p.add_argument(
        "--batch-docs",
        type=int,
        default=DEFAULT_BATCH_DOCS,
        help="Documents per worker batch",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional cap on total number of shards to write",
    )
    return p.parse_args()


def main():
    args = parse_args()

    docs_jsonl = Path(args.docs_jsonl).resolve()
    vocab_jsonl = Path(args.vocab_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not docs_jsonl.is_file():
        raise FileNotFoundError(f"docs file not found: {docs_jsonl}")
    if not vocab_jsonl.is_file():
        raise FileNotFoundError(f"vocab file not found: {vocab_jsonl}")

    vocab_rows = load_jsonl(vocab_jsonl)
    tokenizer_probe = FastCustomDPTokenizer(vocab_rows)

    print(f"docs_jsonl:   {docs_jsonl}")
    print(f"vocab_jsonl:  {vocab_jsonl}")
    print(f"output_dir:   {output_dir}")
    print(f"vocab_size:   {tokenizer_probe.vocab_size}")
    print(f"num_val_docs: {args.num_val_docs}")
    print(f"shard_size:   {args.shard_size_tokens}")
    print(f"bos_id:       {args.bos_id}")
    print(f"eos_id:       {args.eos_id}")
    print(f"append_eos:   {args.append_eos}")
    print(f"workers:      {args.workers}")
    print(f"batch_docs:   {args.batch_docs}")
    print(f"max_shards:   {args.max_shards}")
    print()

    t0 = time.time()

    stats = export_shards_parallel(
        docs_jsonl=docs_jsonl,
        vocab_rows=vocab_rows,
        output_dir=output_dir,
        num_val_docs=args.num_val_docs,
        shard_size_tokens=args.shard_size_tokens,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        append_eos=args.append_eos,
        workers=args.workers,
        batch_docs=args.batch_docs,
        max_shards=args.max_shards,
    )

    build_manifest(
        output_dir=output_dir,
        vocab_path=vocab_jsonl,
        stats=stats,
        num_val_docs=args.num_val_docs,
        shard_size_tokens=args.shard_size_tokens,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        append_eos=args.append_eos,
        vocab_size=tokenizer_probe.vocab_size,
        workers=args.workers,
        batch_docs=args.batch_docs,
        max_shards=args.max_shards,
    )

    elapsed = time.time() - t0

    print("\nDONE")
    print(f"elapsed_sec: {elapsed:,.2f}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()