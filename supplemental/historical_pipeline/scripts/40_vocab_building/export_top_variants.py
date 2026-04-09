#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-csv", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--docs-jsonl", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    with args.variant_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["ratio_custom_over_sp"]))

    for row in rows[: args.top_n]:
        variant = row["variant"]
        vocab_path = Path("/workspace/parameter-golf/vocab_variants") / variant
        out_dir = args.output_root / vocab_path.stem
        cmd = (
            f"python -u scripts/export_custom_dp_dataset_v3.py "
            f"--docs-jsonl {args.docs_jsonl} "
            f"--vocab-jsonl {vocab_path} "
            f"--output-dir {out_dir} "
            f"--workers 16 "
            f"--batch-docs 512 "
            f"--max-train-shards 80"
        )
        print(cmd)


if __name__ == "__main__":
    main()