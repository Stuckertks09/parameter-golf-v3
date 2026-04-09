import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm


def load_candidates(path: str, top_n: int | None = None):
    candidates = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            ids = tuple(row["ids"])
            text = row.get("text", "")
            freq = row.get("freq", 0)
            candidates.append({
                "ids": ids,
                "text": text,
                "freq": freq,
            })
            if top_n is not None and len(candidates) >= top_n:
                break
    return candidates


def apply_greedy_merges(token_ids: list[int], merge_set: set[tuple[int, int, int]]):
    """
    Simulate replacing any matching trigram with a single merged token placeholder.
    We do not need actual vocab IDs yet, only length reduction and examples.
    """
    merged = []
    merge_hits = 0
    i = 0
    n = len(token_ids)

    while i < n:
        if i <= n - 3:
            tri = (token_ids[i], token_ids[i + 1], token_ids[i + 2])
            if tri in merge_set:
                merged.append(("MERGE", tri))
                merge_hits += 1
                i += 3
                continue

        merged.append(token_ids[i])
        i += 1

    return merged, merge_hits


def merged_length(simulated_tokens) -> int:
    return len(simulated_tokens)


def preview_merged_sequence(sp, simulated_tokens, max_items=30):
    """
    Render a human-readable preview of a simulated merged token stream.
    """
    out = []
    for item in simulated_tokens[:max_items]:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "MERGE":
            tri = item[1]
            try:
                text = sp.decode(list(tri))
            except Exception:
                text = "<decode_error>"
            out.append(f"[{text}]")
        else:
            try:
                text = sp.decode([item])
            except Exception:
                text = "<decode_error>"
            out.append(text)
    return "".join(out)


def load_text_samples(path: str, max_lines: int | None = None, min_chars: int = 40):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < min_chars:
                continue
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True,
                        help="Plain text file with one sample per line")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Optional limit on number of merge candidates to load")
    parser.add_argument("--max-lines", type=int, default=1000,
                        help="How many lines to analyze from the text file")
    parser.add_argument("--preview-count", type=int, default=5,
                        help="How many examples to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str,
                        default="ngram/retokenization_simulation_report.json")
    args = parser.parse_args()

    random.seed(args.seed)

    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise ValueError(f"Failed to load tokenizer model: {args.tokenizer}")

    candidates = load_candidates(args.candidates, args.top_n)
    merge_set = {c["ids"] for c in candidates}

    print(f"Loaded tokenizer: {args.tokenizer}")
    print(f"Loaded {len(candidates)} merge candidates")
    print(f"Reading text from: {args.text_file}")

    samples = load_text_samples(args.text_file, max_lines=args.max_lines)
    if not samples:
        raise ValueError("No usable text lines found in text-file")

    total_before = 0
    total_after = 0
    total_merge_hits = 0

    example_rows = []

    for idx, text in enumerate(samples):
        token_ids = sp.encode(text, out_type=int)
        before = len(token_ids)

        simulated_tokens, merge_hits = apply_greedy_merges(token_ids, merge_set)
        after = merged_length(simulated_tokens)

        total_before += before
        total_after += after
        total_merge_hits += merge_hits

        if len(example_rows) < args.preview_count:
            example_rows.append({
                "text": text,
                "before_len": before,
                "after_len": after,
                "merge_hits": merge_hits,
                "before_preview": sp.decode(token_ids[: min(len(token_ids), 40)]),
                "after_preview": preview_merged_sequence(sp, simulated_tokens, max_items=40),
            })

    saved = total_before - total_after
    pct_saved = (saved / total_before * 100.0) if total_before else 0.0
    avg_before = total_before / len(samples)
    avg_after = total_after / len(samples)

    report = {
        "tokenizer": args.tokenizer,
        "candidates": args.candidates,
        "num_candidates_loaded": len(candidates),
        "text_file": args.text_file,
        "num_samples": len(samples),
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "total_tokens_saved": saved,
        "percent_saved": pct_saved,
        "average_tokens_before": avg_before,
        "average_tokens_after": avg_after,
        "total_merge_hits": total_merge_hits,
        "examples": example_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- RESULTS ---")
    print(f"Samples analyzed:      {len(samples):,}")
    print(f"Total tokens before:   {total_before:,}")
    print(f"Total tokens after:    {total_after:,}")
    print(f"Total tokens saved:    {saved:,}")
    print(f"Percent saved:         {pct_saved:.4f}%")
    print(f"Average before/line:   {avg_before:.2f}")
    print(f"Average after/line:    {avg_after:.2f}")
    print(f"Total merge hits:      {total_merge_hits:,}")
    print(f"Saved report to:       {out_path}")

    print("\n--- EXAMPLES ---")
    for i, ex in enumerate(example_rows, start=1):
        print(f"\nExample {i}")
        print(f"before_len={ex['before_len']} after_len={ex['after_len']} merge_hits={ex['merge_hits']}")
        print(f"text:          {ex['text'][:250]}")
        print(f"after_preview: {ex['after_preview'][:250]}")


if __name__ == "__main__":
    main()