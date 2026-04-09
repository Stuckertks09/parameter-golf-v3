from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank vocab candidate score reports from analysis_scores/"
    )
    p.add_argument(
        "--scores-dir",
        default="/workspace/parameter-golf/analysis_scores",
        help="Directory containing *.score.json files",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="How many rows to print",
    )
    p.add_argument(
        "--pattern",
        default="*.score.json",
        help="Glob pattern for score files",
    )
    return p.parse_args()


def load_summary(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary", {})
    config = obj.get("config", {})
    return {
        "file": path.name,
        "path": str(path),
        "vocab_jsonl": config.get("vocab_jsonl", ""),
        "docs_analyzed": summary.get("docs_analyzed", 0),
        "scalar_score": float(summary.get("scalar_score_higher_is_better", float("-inf"))),
        "token_ratio": float(summary.get("token_ratio_custom_over_baseline", float("inf"))),
        "fallback_share": float(summary.get("fallback_token_share_of_custom", float("inf"))),
        "mean_delta": float(summary.get("mean_delta", float("inf"))),
        "median_delta": float(summary.get("median_delta", float("inf"))),
        "p95_delta": float(summary.get("p95_delta", float("inf"))),
        "p99_delta": float(summary.get("p99_delta", float("inf"))),
        "worst100_mean_delta": float(summary.get("worst100_mean_delta", float("inf"))),
        "improved_doc_share": float(summary.get("improved_doc_share", 0.0)),
        "worse_doc_share": float(summary.get("worse_doc_share", 1.0)),
        "baseline_total_tokens": int(summary.get("baseline_total_tokens", 0)),
        "custom_total_tokens": int(summary.get("custom_total_tokens", 0)),
        "token_delta_total": int(summary.get("token_delta_total", 0)),
    }


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)

    if not scores_dir.is_dir():
        raise FileNotFoundError(f"Scores dir not found: {scores_dir}")

    rows = []
    for path in sorted(scores_dir.glob(args.pattern)):
        try:
            rows.append(load_summary(path))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not rows:
        print("No score files found.")
        return

    rows.sort(
        key=lambda r: (
            -r["scalar_score"],
            r["token_ratio"],
            r["fallback_share"],
            r["p95_delta"],
            r["p99_delta"],
            r["worst100_mean_delta"],
        )
    )

    print(
        f"{'rank':>4}  {'name':<38}  {'score':>10}  {'ratio':>10}  "
        f"{'fb_share':>10}  {'p95':>8}  {'p99':>8}  {'worst100':>10}  {'worse%':>8}"
    )
    print("-" * 120)

    for idx, row in enumerate(rows[: args.limit], start=1):
        name = Path(row["vocab_jsonl"]).stem if row["vocab_jsonl"] else row["file"]
        print(
            f"{idx:>4}  "
            f"{name[:38]:<38}  "
            f"{row['scalar_score']:>10.3f}  "
            f"{row['token_ratio']:>10.6f}  "
            f"{row['fallback_share']:>10.6f}  "
            f"{row['p95_delta']:>8.3f}  "
            f"{row['p99_delta']:>8.3f}  "
            f"{row['worst100_mean_delta']:>10.3f}  "
            f"{100.0 * row['worse_doc_share']:>7.2f}%"
        )

    print("\nTop candidate details:\n")
    for idx, row in enumerate(rows[: min(args.limit, 10)], start=1):
        print(f"[{idx}] {row['file']}")
        print(f"    vocab_jsonl:         {row['vocab_jsonl']}")
        print(f"    scalar_score:       {row['scalar_score']:.6f}")
        print(f"    token_ratio:        {row['token_ratio']:.9f}")
        print(f"    fallback_share:     {row['fallback_share']:.9f}")
        print(f"    mean_delta:         {row['mean_delta']:.6f}")
        print(f"    median_delta:       {row['median_delta']:.6f}")
        print(f"    p95_delta:          {row['p95_delta']:.6f}")
        print(f"    p99_delta:          {row['p99_delta']:.6f}")
        print(f"    worst100_mean:      {row['worst100_mean_delta']:.6f}")
        print(f"    improved_doc_share: {row['improved_doc_share']:.6f}")
        print(f"    worse_doc_share:    {row['worse_doc_share']:.6f}")
        print(f"    token_delta_total:  {row['token_delta_total']}")
        print()


if __name__ == "__main__":
    main()