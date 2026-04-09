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
    return tokens.astype(np.uint16, copy=False)


def load_candidates(path: str, top_n: int | None):
    candidates = []
    seen = set()
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            n = int(row["n"])
            ids = tuple(row["ids"])
            if (n, ids) in seen:
                continue
            seen.add((n, ids))
            candidates.append({
                "n": n,
                "ids": ids,
                "freq": row.get("freq", 0),
                "text": row.get("text", ""),
            })
            if top_n is not None and len(candidates) >= top_n:
                break
    return candidates


def build_ngram_counters(tokens: np.ndarray):
    bigrams = Counter()
    trigrams = Counter()
    n = len(tokens)
    for i in range(n - 1):
        bigrams[(int(tokens[i]), int(tokens[i + 1]))] += 1
    for i in range(n - 2):
        trigrams[(int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2]))] += 1
    return bigrams, trigrams


def greedy_apply(tokens: np.ndarray, bigram_set: set, trigram_set: set):
    i = 0
    saved_tokens = 0
    merge_occurrences = 0
    n = len(tokens)

    while i < n:
        if i <= n - 3:
            tri = (int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2]))
            if tri in trigram_set:
                saved_tokens += 2
                merge_occurrences += 1
                i += 3
                continue
        if i <= n - 2:
            bg = (int(tokens[i]), int(tokens[i + 1]))
            if bg in bigram_set:
                saved_tokens += 1
                merge_occurrences += 1
                i += 2
                continue
        i += 1

    return saved_tokens, merge_occurrences


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
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=str, default="ngram/ngram_merge_savings_report.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shard_paths = sorted(dataset_dir.glob("fineweb_train_*.bin"))[: args.num_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {dataset_dir}")

    candidates = load_candidates(args.candidates, args.top_n)
    bigram_ids = [c["ids"] for c in candidates if c["n"] == 2]
    trigram_ids = [c["ids"] for c in candidates if c["n"] == 3]
    bigram_set = set(bigram_ids)
    trigram_set = set(trigram_ids)

    print(f"Using {len(shard_paths)} shard(s)")
    print(f"Loaded {len(bigram_ids)} bigram candidate(s)")
    print(f"Loaded {len(trigram_ids)} trigram candidate(s)")

    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise ValueError(f"Failed to load tokenizer model: {args.tokenizer}")

    total_tokens = 0
    total_naive_saved = 0
    total_greedy_saved = 0
    total_merge_occurrences = 0
    per_candidate_counts = Counter()

    for shard_path in tqdm(shard_paths, desc="Scanning shards", unit="shard"):
        tokens = load_data_shard(shard_path)
        total_tokens += len(tokens)

        bigram_counts, trigram_counts = build_ngram_counters(tokens)

        for ids in bigram_ids:
            count = bigram_counts.get(ids, 0)
            if count:
                per_candidate_counts[(2, ids)] += count
                total_naive_saved += count * 1

        for ids in trigram_ids:
            count = trigram_counts.get(ids, 0)
            if count:
                per_candidate_counts[(3, ids)] += count
                total_naive_saved += count * 2

        shard_saved, shard_merges = greedy_apply(tokens, bigram_set, trigram_set)
        total_greedy_saved += shard_saved
        total_merge_occurrences += shard_merges

    naive_pct = (total_naive_saved / total_tokens * 100.0) if total_tokens else 0.0
    greedy_pct = (total_greedy_saved / total_tokens * 100.0) if total_tokens else 0.0

    ranked = []
    for c in candidates:
        n = c["n"]
        ids = c["ids"]
        count = per_candidate_counts.get((n, ids), 0)
        ranked.append({
            "n": n,
            "ids": list(ids),
            "text": c["text"] or decode_safe(sp, ids),
            "candidate_freq_rank_source": c["freq"],
            "occurrences_in_sample": count,
            "naive_tokens_saved": count * (1 if n == 2 else 2),
        })

    ranked.sort(key=lambda x: x["naive_tokens_saved"], reverse=True)

    report = {
        "dataset_dir": str(dataset_dir),
        "num_shards": len(shard_paths),
        "total_tokens_scanned": total_tokens,
        "num_bigram_candidates": len(bigram_ids),
        "num_trigram_candidates": len(trigram_ids),
        "naive_tokens_saved": total_naive_saved,
        "naive_percent_saved": naive_pct,
        "greedy_tokens_saved": total_greedy_saved,
        "greedy_percent_saved": greedy_pct,
        "greedy_merge_occurrences": total_merge_occurrences,
        "top_candidates_by_sample_savings": ranked[:200],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- RESULTS ---")
    print(f"Total tokens scanned: {total_tokens:,}")
    print(f"Naive tokens saved:  {total_naive_saved:,} ({naive_pct:.4f}%)")
    print(f"Greedy tokens saved: {total_greedy_saved:,} ({greedy_pct:.4f}%)")
    print(f"Greedy merge uses:   {total_merge_occurrences:,}")
    print(f"Saved report to:     {output_path}")

    print("\nTop sample candidates:")
    for row in ranked[:15]:
        print(f"{row['n']}-gram {row['text']!r:16} occ={row['occurrences_in_sample']:<8} saved={row['naive_tokens_saved']}")


if __name__ == "__main__":
    main()