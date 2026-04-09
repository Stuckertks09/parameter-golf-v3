import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


MAGIC = 20240520
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
HEADER_BYTES = HEADER_INTS * HEADER_DTYPE.itemsize


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if len(header) < HEADER_INTS:
        raise ValueError(f"Incomplete header in shard: {file}")
    if int(header[0]) != MAGIC:
        raise ValueError(f"Bad shard: {file}")

    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype=TOKEN_DTYPE, count=num_tokens, offset=HEADER_BYTES)

    if len(tokens) != num_tokens:
        raise ValueError(
            f"Shard {file} expected {num_tokens} tokens but read {len(tokens)}"
        )

    return tokens.astype(np.uint16, copy=False)


def load_candidates(path: str, top_n: int):
    candidates = []
    seen = set()

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("n") != 3:
                continue

            ids = tuple(row["ids"])
            if ids in seen:
                continue

            seen.add(ids)
            candidates.append({
                "ids": ids,
                "freq": row["freq"],
            })

            if len(candidates) >= top_n:
                break

    return candidates


def build_trigram_counter(tokens: np.ndarray) -> Counter:
    counts = Counter()
    n = len(tokens)
    if n < 3:
        return counts

    # One pass over the shard
    for i in range(n - 2):
        tri = (int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2]))
        counts[tri] += 1

    return counts


def greedy_apply_patterns(tokens: np.ndarray, patterns_set: set[tuple[int, int, int]]) -> tuple[int, int]:
    """
    Apply trigram merges greedily left-to-right.
    Returns:
      saved_tokens: total tokens saved
      merged_occurrences: number of merges applied
    """
    n = len(tokens)
    i = 0
    saved_tokens = 0
    merged_occurrences = 0

    while i < n:
        if i <= n - 3:
            tri = (int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2]))
            if tri in patterns_set:
                saved_tokens += 2  # 3 -> 1
                merged_occurrences += 1
                i += 3
                continue
        i += 1

    return saved_tokens, merged_occurrences


def decode_safe(sp, ids):
    try:
        return sp.decode(list(ids))
    except Exception:
        return "<decode_error>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=str, default="ngram/merge_savings_report.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shard_paths = sorted(dataset_dir.glob("fineweb_train_*.bin"))[: args.num_shards]

    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {dataset_dir}")

    candidates = load_candidates(args.candidates, args.top_n)
    candidate_ids = [c["ids"] for c in candidates]
    candidate_set = set(candidate_ids)

    print(f"Using {len(shard_paths)} shard(s)")
    print(f"Loaded {len(candidates)} trigram candidate(s)")

    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise ValueError(f"Failed to load tokenizer model: {args.tokenizer}")

    total_tokens = 0
    total_naive_saved = 0
    total_greedy_saved = 0
    total_greedy_merges = 0

    per_candidate_counts = Counter()

    for shard_path in tqdm(shard_paths, desc="Scanning shards", unit="shard"):
        tokens = load_data_shard(shard_path)
        total_tokens += len(tokens)

        trigram_counts = build_trigram_counter(tokens)

        # Naive upper bound: look up candidate trigram counts from one shard-wide counter
        for ids in candidate_ids:
            count = trigram_counts.get(ids, 0)
            if count:
                per_candidate_counts[ids] += count
                total_naive_saved += count * 2

        # Greedy overlap-aware estimate
        shard_greedy_saved, shard_greedy_merges = greedy_apply_patterns(tokens, candidate_set)
        total_greedy_saved += shard_greedy_saved
        total_greedy_merges += shard_greedy_merges

    naive_pct = (total_naive_saved / total_tokens * 100.0) if total_tokens else 0.0
    greedy_pct = (total_greedy_saved / total_tokens * 100.0) if total_tokens else 0.0

    ranked = []
    for c in candidates:
        ids = c["ids"]
        count = per_candidate_counts.get(ids, 0)
        ranked.append({
            "ids": list(ids),
            "text": decode_safe(sp, ids),
            "candidate_freq_rank_source": c["freq"],
            "occurrences_in_sample": count,
            "naive_tokens_saved": count * 2,
        })

    ranked.sort(key=lambda x: x["naive_tokens_saved"], reverse=True)

    report = {
        "dataset_dir": str(dataset_dir),
        "num_shards": len(shard_paths),
        "top_n_candidates": len(candidates),
        "total_tokens_scanned": total_tokens,
        "naive_tokens_saved": total_naive_saved,
        "naive_percent_saved": naive_pct,
        "greedy_tokens_saved": total_greedy_saved,
        "greedy_percent_saved": greedy_pct,
        "greedy_merge_occurrences": total_greedy_merges,
        "top_candidates_by_sample_savings": ranked[:100],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- RESULTS ---")
    print(f"Total tokens scanned: {total_tokens:,}")
    print(f"Naive tokens saved:  {total_naive_saved:,} ({naive_pct:.4f}%)")
    print(f"Greedy tokens saved: {total_greedy_saved:,} ({greedy_pct:.4f}%)")
    print(f"Greedy merge uses:   {total_greedy_merges:,}")
    print(f"Saved report to:     {output_path}")

    print("\nTop sample candidates:")
    for row in ranked[:15]:
        print(
            f"{row['text']!r:20} "
            f"occ={row['occurrences_in_sample']:<8} "
            f"saved={row['naive_tokens_saved']}"
        )


if __name__ == "__main__":
    main()