#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=Path("data/analysis/fineweb10B_sp1024/shard_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/analysis/fineweb10B_sp1024/shard_scores.csv"))
    parser.add_argument("--top-order-file", type=Path, default=Path("data/top_first_order.txt"))
    parser.add_argument("--bottom-order-file", type=Path, default=Path("data/bottom_first_order.txt"))
    args = parser.parse_args()

    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = []

    for s in report["summaries"]:
        if s["split"] != "train":
            continue
        m = s["aggregated_metrics"]
        row = {
            "shard_name": s["shard_name"],
            "split": s["split"],
            "path": s["path"],
            "file_size_bytes": s["file_size_bytes"],
            "header_bytes": s["header_bytes"],
            "token_dtype": s["token_dtype"],
            "token_count": s["token_count"],
            "min_token": s["min_token"],
            "max_token": s["max_token"],
            "mean_token": s["mean_token"],
            "std_token": s["std_token"],
            "unique_token_count": s["unique_token_count"],
            "unique_token_ratio": s["unique_token_ratio"],
            "sample_window_size": s["sample_window_size"],
            "sample_count": s["sample_count"],
            "preview_1": s["sample_previews"][0] if s["sample_previews"] else "",
            "preview_2": s["sample_previews"][1] if len(s["sample_previews"]) > 1 else "",
            "top_token_1_id": s["top_tokens"][0][0] if s["top_tokens"] else None,
            "top_token_1_freq": s["top_tokens"][0][1] if s["top_tokens"] else None,
            "top_token_2_id": s["top_tokens"][1][0] if len(s["top_tokens"]) > 1 else None,
            "top_token_2_freq": s["top_tokens"][1][1] if len(s["top_tokens"]) > 1 else None,
            "top_token_3_id": s["top_tokens"][2][0] if len(s["top_tokens"]) > 2 else None,
            "top_token_3_freq": s["top_tokens"][2][1] if len(s["top_tokens"]) > 2 else None,
        }
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)

    # crude but useful first-pass scoring
    df["score_diversity"] = (
        zscore(df["mean_unique_word_ratio"]) * 0.45 +
        zscore(df["mean_unique_words"]) * 0.25 +
        zscore(df["unique_token_count"]) * 0.15 +
        zscore(df["std_token"]) * 0.15
    )

    df["score_noise"] = (
        zscore(df["mean_url_hits"]) * 0.30 +
        zscore(df["mean_html_hits"]) * 0.30 +
        zscore(df["mean_repeated_char_hits"]) * 0.20 +
        zscore(df["mean_non_printable_ratio"]) * 0.10 +
        zscore(df["mean_digit_ratio"]) * 0.10
    )

    df["score_clean_prose"] = (
        zscore(df["mean_ascii_ratio"]) * 0.25 +
        zscore(df["mean_unique_word_ratio"]) * 0.25 +
        zscore(df["mean_unique_words"]) * 0.20 -
        zscore(df["mean_digit_ratio"]) * 0.10 -
        zscore(df["mean_url_hits"]) * 0.10 -
        zscore(df["mean_html_hits"]) * 0.10
    )

    df["score_priority"] = (
        df["score_clean_prose"] * 0.55 +
        df["score_diversity"] * 0.45 -
        df["score_noise"] * 0.35
    )

    for col in [
        "score_diversity",
        "score_noise",
        "score_clean_prose",
        "score_priority",
        "mean_unique_word_ratio",
        "mean_unique_words",
        "mean_ascii_ratio",
        "unique_token_count",
        "std_token",
    ]:
        df[f"rank_{col}_desc"] = df[col].rank(method="min", ascending=False)

    for col in [
        "mean_url_hits",
        "mean_html_hits",
        "mean_repeated_char_hits",
        "mean_non_printable_ratio",
        "mean_digit_ratio",
    ]:
        df[f"rank_{col}_asc"] = df[col].rank(method="min", ascending=True)

    df = df.sort_values("score_priority", ascending=False).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    args.top_order_file.write_text(
        "\n".join(df["shard_name"].tolist()) + "\n",
        encoding="utf-8",
    )

    args.bottom_order_file.write_text(
        "\n".join(df.sort_values("score_priority", ascending=True)["shard_name"].tolist()) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.top_order_file}")
    print(f"Wrote: {args.bottom_order_file}")
    print("\nTop 20 by score_priority:")
    print(df[["shard_name", "score_priority"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()