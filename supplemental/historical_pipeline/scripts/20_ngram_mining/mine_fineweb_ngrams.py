#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import sentencepiece as spm
except ImportError as exc:
    raise SystemExit("Install sentencepiece: pip install sentencepiece") from exc


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    arr = np.fromfile(file, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)
    if arr.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return arr


def capped_add(d: dict[tuple[int, ...], set[int]], key: tuple[int, ...], value: int, cap: int) -> None:
    s = d[key]
    if len(s) < cap:
        s.add(int(value))


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/analysis/fineweb10B_sp1024/ngram_candidates.jsonl"))
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--context-cap", type=int, default=256)
    parser.add_argument("--decode-top-k", type=int, default=2000)
    args = parser.parse_args()

    if args.max_n < 2:
        raise ValueError("--max-n must be >= 2")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))

    shard_paths = sorted(args.dataset_dir.glob("fineweb_train_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No train shards found under {args.dataset_dir}")

    unigram = Counter()
    ngram_counts: dict[int, Counter[tuple[int, ...]]] = {n: Counter() for n in range(2, args.max_n + 1)}
    left_contexts: dict[int, dict[tuple[int, ...], set[int]]] = {
        n: defaultdict(set) for n in range(2, args.max_n + 1)
    }
    right_contexts: dict[int, dict[tuple[int, ...], set[int]]] = {
        n: defaultdict(set) for n in range(2, args.max_n + 1)
    }
    shard_occurrence: dict[int, dict[tuple[int, ...], int]] = {
        n: defaultdict(int) for n in range(2, args.max_n + 1)
    }

    total_tokens = 0

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        tokens = load_data_shard(shard_path)
        total_tokens += int(tokens.size)
        unigram.update(tokens.tolist())

        seen_in_shard: dict[int, set[tuple[int, ...]]] = {n: set() for n in range(2, args.max_n + 1)}

        for n in range(2, args.max_n + 1):
            if tokens.size < n:
                continue
            for i in range(0, tokens.size - n + 1):
                gram = tuple(int(x) for x in tokens[i:i + n])
                ngram_counts[n][gram] += 1
                seen_in_shard[n].add(gram)

                if i > 0:
                    capped_add(left_contexts[n], gram, int(tokens[i - 1]), args.context_cap)
                if i + n < tokens.size:
                    capped_add(right_contexts[n], gram, int(tokens[i + n]), args.context_cap)

        for n in range(2, args.max_n + 1):
            for gram in seen_in_shard[n]:
                shard_occurrence[n][gram] += 1

        if shard_idx % 10 == 0:
            print(f"Processed {shard_idx}/{len(shard_paths)} shards...")

    # gather candidates
    candidates = []
    for n in range(2, args.max_n + 1):
        for gram, count in ngram_counts[n].items():
            if count < args.min_count:
                continue

            # PMI-like association
            p_ngram = count / max(total_tokens - n + 1, 1)
            p_parts = 1.0
            for tok in gram:
                p_parts *= unigram[tok] / max(total_tokens, 1)
            score_pmi = safe_log(p_ngram / max(p_parts, 1e-18))

            left_div = len(left_contexts[n].get(gram, set()))
            right_div = len(right_contexts[n].get(gram, set()))
            shard_count = shard_occurrence[n][gram]

            tokens_saved_per_occurrence = n - 1
            estimated_tokens_saved_total = count * tokens_saved_per_occurrence

            # merge score: count, saved tokens, association, and contextual reusability
            merge_value = (
                estimated_tokens_saved_total *
                max(score_pmi, 0.0) *
                (1.0 + math.log1p(left_div + right_div))
            )

            candidates.append({
                "n": n,
                "token_ids": list(gram),
                "count": count,
                "shard_count": shard_count,
                "left_context_count": left_div,
                "right_context_count": right_div,
                "tokens_saved_per_occurrence": tokens_saved_per_occurrence,
                "estimated_tokens_saved_total": estimated_tokens_saved_total,
                "score_pmi": score_pmi,
                "score_merge_value": merge_value,
            })

    candidates.sort(key=lambda x: x["score_merge_value"], reverse=True)

    # decode only top portion
    for row in candidates[:args.decode_top_k]:
        try:
            row["decoded_text"] = sp.decode_ids(row["token_ids"])
        except Exception:
            row["decoded_text"] = ""

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in candidates[:args.top_k]:
            if "decoded_text" not in row:
                row["decoded_text"] = ""
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote top {min(len(candidates), args.top_k)} candidates to {args.output_jsonl}")
    print("Top 20 candidates:")
    for row in candidates[:20]:
        decoded = row.get("decoded_text", "")
        print(
            f"n={row['n']} count={row['count']} "
            f"merge={row['score_merge_value']:.2f} "
            f"pmi={row['score_pmi']:.3f} "
            f"text={decoded[:80]!r}"
        )


if __name__ == "__main__":
    main()