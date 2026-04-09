#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import sentencepiece as spm

from dp_tokenizer_lib import load_dp_vocab, encode_dp


def load_docs_jsonl(path: Path, limit: int) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(docs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) and "text" in obj else obj
            if isinstance(text, str) and text:
                docs.append(text)
    return docs


def analyze_variant(
    docs: list[str],
    sp: spm.SentencePieceProcessor,
    vocab,
) -> dict:
    total_sp = 0
    total_custom = 0
    total_custom_fallback = 0
    docs_custom_beats_sp = 0
    docs_sp_beats_custom = 0
    docs_tied = 0

    biggest_sp_win = {"delta": -10**9, "doc_id": None, "chars": None, "sp_tokens": None, "custom_tokens": None}
    biggest_custom_win = {"delta": -10**9, "doc_id": None, "chars": None, "sp_tokens": None, "custom_tokens": None}

    for doc_id, text in enumerate(docs):
        sp_ids = list(sp.encode(text, out_type=int))
        enc = encode_dp(text, vocab)
        custom_ids = enc.ids

        n_sp = len(sp_ids)
        n_custom = len(custom_ids)
        delta = n_sp - n_custom  # positive => custom uses fewer tokens

        total_sp += n_sp
        total_custom += n_custom
        total_custom_fallback += enc.fallback_count

        if n_custom < n_sp:
            docs_custom_beats_sp += 1
        elif n_custom > n_sp:
            docs_sp_beats_custom += 1
        else:
            docs_tied += 1

        if n_custom - n_sp > biggest_sp_win["delta"]:
            biggest_sp_win = {
                "delta": n_custom - n_sp,
                "doc_id": doc_id,
                "chars": len(text),
                "sp_tokens": n_sp,
                "custom_tokens": n_custom,
            }

        if delta > biggest_custom_win["delta"]:
            biggest_custom_win = {
                "delta": delta,
                "doc_id": doc_id,
                "chars": len(text),
                "sp_tokens": n_sp,
                "custom_tokens": n_custom,
            }

    return {
        "docs": len(docs),
        "sp_tokens": total_sp,
        "custom_tokens": total_custom,
        "token_delta_sp_minus_custom": total_sp - total_custom,
        "ratio_custom_over_sp": (total_custom / total_sp) if total_sp else None,
        "custom_fallback_tokens": total_custom_fallback,
        "custom_fallback_share": (total_custom_fallback / total_custom) if total_custom else None,
        "docs_custom_beats_sp": docs_custom_beats_sp,
        "docs_sp_beats_custom": docs_sp_beats_custom,
        "docs_tied": docs_tied,
        "biggest_sp_win_delta": biggest_sp_win["delta"],
        "biggest_sp_win_doc_id": biggest_sp_win["doc_id"],
        "biggest_custom_win_delta": biggest_custom_win["delta"],
        "biggest_custom_win_doc_id": biggest_custom_win["doc_id"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--sp-model", type=Path, required=True)
    ap.add_argument("--variant-glob", type=str, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--num-docs", type=int, default=10000)
    args = ap.parse_args()

    docs = load_docs_jsonl(args.docs, args.num_docs)
    sp = spm.SentencePieceProcessor(model_file=str(args.sp_model))
    variant_paths = [Path(p) for p in sorted(glob.glob(args.variant_glob))]

    if not variant_paths:
        raise FileNotFoundError(f"No variant files matched: {args.variant_glob}")

    rows: list[dict] = []
    for vp in variant_paths:
        vocab = load_dp_vocab(vp)
        stats = analyze_variant(docs, sp, vocab)
        row = {"variant": vp.name, **stats}
        rows.append(row)
        print(vp.name, stats)

    rows.sort(key=lambda r: float(r["ratio_custom_over_sp"]) if r["ratio_custom_over_sp"] is not None else 999.0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()