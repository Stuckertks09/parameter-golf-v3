#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SPECIAL_IDS_DEFAULT = {0, 1, 2, 3}

PUNCT_PROTECT = {
    ".", ",", ":", ";", "!", "?", "'", '"', "`",
    "’", "“", "”", "(", ")", "[", "]", "{", "}",
    "-", "–", "—", "/", "\\", "&", "%", "$", "#",
    "@", "*", "+", "=", "_", "|", "<", ">"
}

CONTRACTION_PROTECT = {
    "'s", "’s", "'t", "’t", "'re", "’re", "'ve", "’ve",
    "'ll", "’ll", "'m", "’m", "'d", "’d",
    " s", " t", " re", " ve", " ll", " m", " d",
    "’", "'"
}


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


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def is_single_alpha(piece: str) -> bool:
    s = piece.strip()
    return len(s) == 1 and s.isalpha()


def is_single_digit(piece: str) -> bool:
    s = piece.strip()
    return len(s) == 1 and s.isdigit()


def is_whitespace_like(piece: str) -> bool:
    return piece != "" and piece.strip() == ""


def is_short_structural(piece: str) -> bool:
    s = piece.strip()
    if s in PUNCT_PROTECT:
        return True
    if piece in CONTRACTION_PROTECT or s in CONTRACTION_PROTECT:
        return True
    if is_single_alpha(piece):
        return True
    if is_single_digit(piece):
        return True
    if is_whitespace_like(piece):
        return True
    return False


def is_protected_piece(piece: str) -> bool:
    if is_byte_piece(piece):
        return True
    if is_short_structural(piece):
        return True
    return False


def compute_keep_score(row: dict) -> float:
    freq = safe_int(row.get("freq"))
    doc_freq = safe_int(row.get("doc_freq"))
    context_div = safe_int(row.get("context_diversity"))
    compression_credit = safe_float(row.get("compression_credit"))

    # Higher keep score => harder to remove
    return (
        (freq ** 0.5) * 2.0
        + (doc_freq ** 0.5) * 3.0
        + (context_div ** 0.5) * 1.5
        + max(0.0, compression_credit) * 4.0
    )


def compute_add_score(row: dict) -> float:
    freq = safe_int(row.get("freq"))
    doc_freq = safe_int(row.get("doc_freq"))
    context_div = safe_int(row.get("context_diversity"))
    compression_credit = safe_float(row.get("compression_credit"))
    piece = row["piece"]

    bonus = 0.0
    if piece.startswith(" "):
        bonus += 1.0
    if len(piece.strip()) >= 2:
        bonus += 0.5

    return (
        (freq ** 0.5) * 2.0
        + (doc_freq ** 0.5) * 3.0
        + (context_div ** 0.5) * 1.0
        + max(0.0, compression_credit) * 4.0
        + bonus
    )


def choose_remove_candidates(
    base_vocab_rows,
    token_report_rows,
    num_remove,
    special_ids,
):
    report_by_piece = {
        r["piece"]: r for r in token_report_rows if r.get("source") == "sp"
    }

    candidates = []

    for row in base_vocab_rows:
        tid = int(row["id"])
        piece = row["piece"]

        if tid in special_ids:
            continue
        if is_protected_piece(piece):
            continue

        rep = report_by_piece.get(piece)
        if rep is None:
            continue

        freq = safe_int(rep.get("freq"))
        doc_freq = safe_int(rep.get("doc_freq"))
        context_div = safe_int(rep.get("context_diversity"))
        compression_credit = safe_float(rep.get("compression_credit"))

        score = (
            freq * 1.0
            + doc_freq * 2.0
            + context_div * 0.5
            + max(0.0, compression_credit) * 5.0
        )

        candidates.append((score, row))

    candidates.sort(key=lambda x: x[0])
    return [row for _, row in candidates[:num_remove]]


def choose_add_candidates(
    token_report_rows: list[dict],
    existing_pieces: set[str],
    num_add: int,
    min_freq: int,
    min_doc_freq: int,
) -> list[str]:
    """
    Add strong non-base pieces (typically custom-only candidates) that:
    - are not already present in the SP base
    - are not byte pieces
    - are not protected structural tokens
    - are reasonably frequent and broad
    """
    candidates = []

    for row in token_report_rows:
        piece = row["piece"]

        if piece in existing_pieces:
            continue
        if is_byte_piece(piece):
            continue
        if is_protected_piece(piece):
            continue

        freq = safe_int(row.get("freq"))
        doc_freq = safe_int(row.get("doc_freq"))
        in_sp = safe_int(row.get("in_sp"))
        in_custom = safe_int(row.get("in_custom"))

        # for SP-shaped variants, additions must be outside the SP base
        if in_sp == 1:
            continue
        if in_custom != 1:
            continue
        if freq < min_freq:
            continue
        if doc_freq < min_doc_freq:
            continue

        add_score = compute_add_score(row)
        candidates.append((add_score, piece, row))

    candidates.sort(key=lambda x: x[0], reverse=True)

    chosen = []
    seen = set(existing_pieces)
    for _, piece, _ in candidates:
        if piece in seen:
            continue
        chosen.append(piece)
        seen.add(piece)
        if len(chosen) >= num_add:
            break

    print("add candidate count:", len(candidates))
    print("top add candidates:", [p for _, p, _ in candidates[:10]])

    return chosen


def make_variant(base_vocab_rows: list[dict], remove_rows: list[dict], add_pieces: list[str]) -> list[dict]:
    if len(remove_rows) != len(add_pieces):
        raise ValueError(f"remove/add mismatch: {len(remove_rows)} vs {len(add_pieces)}")

    out = [dict(r) for r in base_vocab_rows]
    id_to_idx = {int(r["id"]): i for i, r in enumerate(out)}

    for remove_row, add_piece in zip(remove_rows, add_pieces, strict=True):
        tid = int(remove_row["id"])
        idx = id_to_idx[tid]
        out[idx]["piece"] = add_piece

    out.sort(key=lambda r: int(r["id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-vocab", type=Path, required=True)
    ap.add_argument("--token-report-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--swap-counts", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--min-add-freq", type=int, default=500)
    ap.add_argument("--min-add-doc-freq", type=int, default=200)
    ap.add_argument("--name-prefix", type=str, default="vocab_sp_shaped_safe")

    args = ap.parse_args()

    base_vocab_rows = load_jsonl(args.base_vocab)
    token_report_rows = load_csv_rows(args.token_report_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    special_ids = set(SPECIAL_IDS_DEFAULT)
    existing_pieces = {r["piece"] for r in base_vocab_rows}

    for n in args.swap_counts:
        remove_rows = choose_remove_candidates(
            base_vocab_rows=base_vocab_rows,
            token_report_rows=token_report_rows,
            num_remove=n,
            special_ids=special_ids,
        )

        add_pieces = choose_add_candidates(
            token_report_rows=token_report_rows,
            existing_pieces=existing_pieces,
            num_add=n,
            min_freq=args.min_add_freq,
            min_doc_freq=args.min_add_doc_freq,
        )

        if len(remove_rows) < n:
            raise ValueError(
                f"Only found {len(remove_rows)} removable tokens, need {n}."
            )

        if len(add_pieces) < n:
            raise ValueError(
                f"Only found {len(add_pieces)} add candidates, need {n}."
            )

        variant = make_variant(base_vocab_rows, remove_rows, add_pieces)
        out_path = args.out_dir / f"{args.name_prefix}_swap{n}.jsonl"
        write_jsonl(variant, out_path)

        print(f"Wrote {out_path}")
        print("  remove preview:")
        for r in remove_rows[:5]:
            print(f"    id={r['id']} piece={r['piece']!r}")
        print("  add preview:")
        for p in add_pieces[:5]:
            print(f"    piece={p!r}")


if __name__ == "__main__":
    main()