#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


SPECIAL_IDS_DEFAULT = {0, 1, 2, 3}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def is_protected_token(row: dict, special_ids: set[int]) -> bool:
    tid = int(row["id"])
    piece = row["piece"]
    return tid in special_ids or is_byte_piece(piece)


def safe_float(x: str | None, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: str | None, default: int = 0) -> int:
    if x is None or x == "":
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def normalize_phrase_piece(piece: str) -> str:
    # Keep literal text form. If caller wants leading-space phrases,
    # they should already be present in the CSV as such.
    return piece


def choose_removal_candidates(
    sp_vocab: list[dict],
    token_report_rows: list[dict] | None,
    num_remove: int,
    special_ids: set[int],
) -> list[dict]:
    """
    Remove weakest learned SP tokens only.
    Priority:
      1. Use token_report.csv stats if provided
      2. Otherwise remove from highest token IDs downward among learned pieces
    """
    protected_ids = {
        int(r["id"]) for r in sp_vocab if is_protected_token(r, special_ids)
    }

    if token_report_rows:
        # Filter to SP-only rows from token_report if available, but allow overlap too.
        # We care about weak learned SP pieces.
        report_by_piece = {
            r["piece"]: r for r in token_report_rows if r.get("source") == "sp"
        }

        scored: list[tuple[float, dict]] = []
        for row in sp_vocab:
            tid = int(row["id"])
            piece = row["piece"]
            if tid in protected_ids:
                continue
            rep = report_by_piece.get(piece)
            if rep is None:
                # unknown report row => conservative, make harder to remove
                weakness = -1e9
            else:
                freq = safe_int(rep.get("freq"))
                doc_freq = safe_int(rep.get("doc_freq"))
                context_div = safe_int(rep.get("context_diversity"))
                compression_credit = safe_float(rep.get("compression_credit"))
                piece_bytes = safe_float(rep.get("piece_bytes"), 1.0)

                # Higher weakness => more removable
                weakness = (
                    (1.0 / max(freq, 1)) * 5000.0
                    + (1.0 / max(doc_freq, 1)) * 5000.0
                    + (1.0 / max(context_div, 1)) * 100.0
                    + max(0.0, -compression_credit) * 5.0
                    + max(0.0, piece_bytes - 4.0) * 0.5
                )
            scored.append((weakness, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:num_remove]]

    # Fallback: remove highest-id learned tokens first
    learned = [r for r in sp_vocab if int(r["id"]) not in protected_ids]
    learned.sort(key=lambda r: int(r["id"]), reverse=True)
    return learned[:num_remove]


def choose_phrase_candidates(
    phrase_rows: list[dict],
    existing_pieces: set[str],
    num_add: int,
    min_score: float,
    score_col: str,
    piece_col: str,
) -> list[str]:
    chosen: list[str] = []
    seen = set(existing_pieces)

    for row in phrase_rows:
        piece = normalize_phrase_piece(row[piece_col])
        score = safe_float(row.get(score_col))
        if score < min_score:
            continue
        if not piece or piece in seen:
            continue
        if is_byte_piece(piece):
            continue
        chosen.append(piece)
        seen.add(piece)
        if len(chosen) >= num_add:
            break

    return chosen


def make_variant(
    sp_vocab: list[dict],
    remove_rows: list[dict],
    add_pieces: list[str],
) -> list[dict]:
    if len(remove_rows) != len(add_pieces):
        raise ValueError(f"remove/add length mismatch: {len(remove_rows)} vs {len(add_pieces)}")

    out = [dict(r) for r in sp_vocab]
    id_to_idx = {int(r["id"]): i for i, r in enumerate(out)}

    for remove_row, new_piece in zip(remove_rows, add_pieces, strict=True):
        tid = int(remove_row["id"])
        idx = id_to_idx[tid]
        out[idx]["piece"] = new_piece

    out.sort(key=lambda r: int(r["id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp-vocab-jsonl", type=Path, required=True)
    ap.add_argument("--phrase-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument(
        "--swap-counts",
        type=int,
        nargs="+",
        default=[10, 25, 50],
        help="How many SP learned tokens to replace with phrase candidates",
    )
    ap.add_argument(
        "--piece-col",
        type=str,
        default="piece",
        help="Column in phrase CSV containing token text",
    )
    ap.add_argument(
        "--score-col",
        type=str,
        default="score",
        help="Column in phrase CSV used to rank phrase candidates",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=float("-inf"),
        help="Minimum phrase score to include",
    )
    ap.add_argument(
        "--token-report-csv",
        type=Path,
        default=None,
        help="Optional token_report.csv from compare_vocab_to_baseline.py to choose weak SP removals more intelligently",
    )
    ap.add_argument(
        "--name-prefix",
        type=str,
        default="vocab_sp_plus_phrases",
    )

    args = ap.parse_args()

    sp_vocab = load_jsonl(args.sp_vocab_jsonl)
    phrase_rows = load_csv_rows(args.phrase_csv)
    token_report_rows = load_csv_rows(args.token_report_csv) if args.token_report_csv else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    special_ids = set(SPECIAL_IDS_DEFAULT)
    existing_pieces = {r["piece"] for r in sp_vocab}

    for n in args.swap_counts:
        remove_rows = choose_removal_candidates(
            sp_vocab=sp_vocab,
            token_report_rows=token_report_rows,
            num_remove=n,
            special_ids=special_ids,
        )

        add_pieces = choose_phrase_candidates(
            phrase_rows=phrase_rows,
            existing_pieces=existing_pieces,
            num_add=n,
            min_score=args.min_score,
            score_col=args.score_col,
            piece_col=args.piece_col,
        )

        if len(add_pieces) < n:
            raise ValueError(
                f"Only found {len(add_pieces)} addable phrase candidates, need {n}. "
                f"Relax filters or provide more phrase rows."
            )

        variant = make_variant(sp_vocab, remove_rows, add_pieces)

        out_path = args.out_dir / f"{args.name_prefix}_swap{n}.jsonl"
        write_jsonl(variant, out_path)

        print(f"Wrote {out_path}")
        print("  removed:")
        for r in remove_rows[: min(5, len(remove_rows))]:
            print(f"    id={r['id']} piece={r['piece']!r}")
        print("  added:")
        for p in add_pieces[: min(5, len(add_pieces))]:
            print(f"    piece={p!r}")


if __name__ == "__main__":
    main()