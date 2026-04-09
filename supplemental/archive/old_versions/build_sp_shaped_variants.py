#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_vocab_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_special_or_byte(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def make_variant(
    base_vocab: list[dict],
    replacement_rows: list[dict],
    num_swaps: int,
    out_path: Path,
) -> None:
    id_to_row = {int(r["id"]): dict(r) for r in base_vocab}
    piece_to_id = {r["piece"]: int(r["id"]) for r in base_vocab}

    used_ids = set(id_to_row.keys())
    seen_pieces = set(piece_to_id.keys())

    applied = 0
    for rr in replacement_rows:
        if applied >= num_swaps:
            break

        remove_id = int(rr["remove_custom_token_id"])
        remove_piece = rr["remove_custom_piece"]
        add_piece = rr["add_sp_piece"]

        if remove_id not in id_to_row:
            continue
        if id_to_row[remove_id]["piece"] != remove_piece:
            continue
        if add_piece in seen_pieces:
            continue
        if is_special_or_byte(remove_piece):
            continue
        if is_special_or_byte(add_piece):
            continue

        id_to_row[remove_id]["piece"] = add_piece
        seen_pieces.discard(remove_piece)
        seen_pieces.add(add_piece)
        applied += 1

    final_rows = [id_to_row[i] for i in sorted(id_to_row)]
    write_vocab_jsonl(final_rows, out_path)
    print(f"Wrote {out_path} with {applied} swaps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-vocab", type=Path, required=True)
    ap.add_argument("--replacement-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--swap-counts", type=int, nargs="+", default=[25, 50, 100])
    args = ap.parse_args()

    base_vocab = load_jsonl(args.base_vocab)
    replacement_rows = load_csv_rows(args.replacement_map)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.base_vocab.stem

    for n in args.swap_counts:
        out_path = args.out_dir / f"{stem}_sp_shaped_swap{n}.jsonl"
        make_variant(base_vocab, replacement_rows, n, out_path)


if __name__ == "__main__":
    main()