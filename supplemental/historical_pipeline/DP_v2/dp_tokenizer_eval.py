import argparse
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from dp_tokenizer_lib import (
    decode_ids,
    encode_dp,
    ids_to_pieces,
    load_dp_vocab,
)

MAX_LINE_REPORT = 40


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def top_counter(counter, n=40):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="data/tokenizers/fineweb_1024_bpe.model")
    ap.add_argument("--vocab-jsonl", default="vocab/vocab_best.jsonl")
    ap.add_argument("--sample-text", default="sample_text_large.txt")
    ap.add_argument("--max-line-report", type=int, default=MAX_LINE_REPORT)
    args = ap.parse_args()

    lines = load_lines(Path(args.sample_text))
    vocab = load_dp_vocab(args.vocab_jsonl)

    sp = spm.SentencePieceProcessor()
    ok = sp.load(args.base_model)
    if not ok:
        raise FileNotFoundError(f"Could not load baseline model: {args.base_model}")

    totals = {
        "base": 0,
        "dp_min_tokens": 0,
        "dp_boundary": 0,
        "fallback_min_tokens": 0,
        "fallback_boundary": 0,
        "fallback_runs_min_tokens": 0,
        "fallback_runs_boundary": 0,
        "boundary_fallback_min_tokens": 0,
        "boundary_fallback_boundary": 0,
    }

    summary = {
        "boundary_better_token_count": 0,
        "boundary_same_token_count_better_fallback": 0,
        "boundary_same_token_count_better_runs": 0,
        "boundary_same_token_count_better_boundary": 0,
        "boundary_worse_token_count": 0,
        "min_better_than_base": 0,
        "boundary_better_than_base": 0,
    }

    improved_lines = []
    worsened_lines = []
    piece_gain_counter = Counter()
    piece_loss_counter = Counter()

    for idx, text in enumerate(lines):
        base_ids = sp.encode(text, out_type=int)
        res_min = encode_dp(text, vocab, mode="min_tokens")
        res_boundary = encode_dp(text, vocab, mode="boundary")

        totals["base"] += len(base_ids)
        totals["dp_min_tokens"] += res_min.token_count
        totals["dp_boundary"] += res_boundary.token_count
        totals["fallback_min_tokens"] += res_min.fallback_count
        totals["fallback_boundary"] += res_boundary.fallback_count
        totals["fallback_runs_min_tokens"] += res_min.fallback_runs
        totals["fallback_runs_boundary"] += res_boundary.fallback_runs
        totals["boundary_fallback_min_tokens"] += res_min.boundary_fallback_count
        totals["boundary_fallback_boundary"] += res_boundary.boundary_fallback_count

        if res_min.token_count < len(base_ids):
            summary["min_better_than_base"] += 1
        if res_boundary.token_count < len(base_ids):
            summary["boundary_better_than_base"] += 1

        record = {
            "line_idx": idx,
            "text": text,
            "base_len": len(base_ids),
            "min_len": res_min.token_count,
            "boundary_len": res_boundary.token_count,
            "min_fallback": res_min.fallback_count,
            "boundary_fallback": res_boundary.fallback_count,
            "min_runs": res_min.fallback_runs,
            "boundary_runs": res_boundary.fallback_runs,
            "min_boundary_fallback": res_min.boundary_fallback_count,
            "boundary_boundary_fallback": res_boundary.boundary_fallback_count,
            "min_score": res_min.score,
            "boundary_score": res_boundary.score,
            "min_pieces": ids_to_pieces(res_min.ids, vocab),
            "boundary_pieces": ids_to_pieces(res_boundary.ids, vocab),
        }

        if res_boundary.token_count < res_min.token_count:
            summary["boundary_better_token_count"] += 1
            improved_lines.append(record)
        elif res_boundary.token_count > res_min.token_count:
            summary["boundary_worse_token_count"] += 1
            worsened_lines.append(record)
        else:
            improved = False
            if res_boundary.fallback_count < res_min.fallback_count:
                summary["boundary_same_token_count_better_fallback"] += 1
                improved = True
            if res_boundary.fallback_runs < res_min.fallback_runs:
                summary["boundary_same_token_count_better_runs"] += 1
                improved = True
            if res_boundary.boundary_fallback_count < res_min.boundary_fallback_count:
                summary["boundary_same_token_count_better_boundary"] += 1
                improved = True
            if improved:
                improved_lines.append(record)

        min_counter = Counter(record["min_pieces"])
        b_counter = Counter(record["boundary_pieces"])
        for piece, count in b_counter.items():
            if min_counter[piece] < count:
                piece_gain_counter[piece] += count - min_counter[piece]
        for piece, count in min_counter.items():
            if b_counter[piece] < count:
                piece_loss_counter[piece] += count - b_counter[piece]

        if decode_ids(res_min.ids, vocab) != text:
            raise RuntimeError(f"min_tokens decode mismatch on line {idx}")
        if decode_ids(res_boundary.ids, vocab) != text:
            raise RuntimeError(f"boundary decode mismatch on line {idx}")

    improved_lines.sort(
        key=lambda r: (
            r["boundary_len"] - r["min_len"],
            r["boundary_fallback"] - r["min_fallback"],
            r["boundary_runs"] - r["min_runs"],
            r["boundary_boundary_fallback"] - r["min_boundary_fallback"],
        )
    )
    worsened_lines.sort(
        key=lambda r: (
            r["boundary_len"] - r["min_len"],
            r["boundary_fallback"] - r["min_fallback"],
        ),
        reverse=True,
    )

    print("CONFIG")
    print(f"BASE_MODEL:   {args.base_model}")
    print(f"VOCAB_PATH:   {args.vocab_jsonl}")
    print(f"SAMPLE_TEXT:  {args.sample_text}")
    print()

    print("TOTALS")
    for k, v in totals.items():
        print(f"{k}: {v}")
    print()

    print("SUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print()

    base = totals["base"]
    print("VS BASELINE")
    print(f"min_tokens abs_saved: {base - totals['dp_min_tokens']}")
    print(f"min_tokens pct_saved: {(base - totals['dp_min_tokens']) / base:.12f}")
    print(f"boundary   abs_saved: {base - totals['dp_boundary']}")
    print(f"boundary   pct_saved: {(base - totals['dp_boundary']) / base:.12f}")
    print()

    print("BOUNDARY VS MIN_TOKENS")
    print(f"token delta:          {totals['dp_min_tokens'] - totals['dp_boundary']}")
    print(f"fallback delta:       {totals['fallback_min_tokens'] - totals['fallback_boundary']}")
    print(f"fallback run delta:   {totals['fallback_runs_min_tokens'] - totals['fallback_runs_boundary']}")
    print(f"boundary fb delta:    {totals['boundary_fallback_min_tokens'] - totals['boundary_fallback_boundary']}")
    print()

    print(f"TOP {args.max_line_report} IMPROVED LINES")
    for r in improved_lines[:args.max_line_report]:
        print("-" * 100)
        print(
            f"line={r['line_idx']} "
            f"base={r['base_len']} "
            f"min={r['min_len']} "
            f"boundary={r['boundary_len']} "
            f"min_fb={r['min_fallback']} "
            f"boundary_fb={r['boundary_fallback']} "
            f"min_runs={r['min_runs']} "
            f"boundary_runs={r['boundary_runs']} "
            f"min_bfb={r['min_boundary_fallback']} "
            f"boundary_bfb={r['boundary_boundary_fallback']}"
        )
        print(repr(r["text"]))
        print("MIN TOKENS:", r["min_pieces"])
        print("BOUNDARY:  ", r["boundary_pieces"])

    if worsened_lines:
        print()
        print(f"TOP {args.max_line_report} WORSENED LINES")
        for r in worsened_lines[:args.max_line_report]:
            print("-" * 100)
            print(
                f"line={r['line_idx']} "
                f"base={r['base_len']} "
                f"min={r['min_len']} "
                f"boundary={r['boundary_len']} "
                f"min_fb={r['min_fallback']} "
                f"boundary_fb={r['boundary_fallback']}"
            )
            print(repr(r["text"]))
            print("MIN TOKENS:", r["min_pieces"])
            print("BOUNDARY:  ", r["boundary_pieces"])

    print()
    print("TOP PIECES GAINED BY BOUNDARY SCORER")
    for piece, count in top_counter(piece_gain_counter, 50):
        print(f"{count:6d}  {repr(piece)}")

    print()
    print("TOP PIECES REDUCED BY BOUNDARY SCORER")
    for piece, count in top_counter(piece_loss_counter, 50):
        print(f"{count:6d}  {repr(piece)}")


if __name__ == "__main__":
    main()
