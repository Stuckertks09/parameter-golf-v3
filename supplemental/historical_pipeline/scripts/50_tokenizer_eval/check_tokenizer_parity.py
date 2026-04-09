from __future__ import annotations

import json
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import json

from export_custom_dp_dataset import CustomDPTokenizer as OldTokenizer, load_jsonl as load_old_jsonl
from export_custom_dp_dataset_mp import FastCustomDPTokenizer as NewTokenizer


DOCS_PATH = Path("/workspace/parameter-golf/data/docs_selected.jsonl")
VOCAB_PATH = Path("/workspace/parameter-golf/vocab/b650_p36_w32_bs-priority_ps-combined_ws-combined_pin-none_ms10.jsonl")
NUM_DOCS = 2000  # bump to 5000 if you want


def iter_docs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_DOCS:
                break
            obj = json.loads(line)
            yield i, obj["text"]


def main():
    vocab_rows = load_old_jsonl(VOCAB_PATH)
    old_tok = OldTokenizer(vocab_rows)
    new_tok = NewTokenizer(vocab_rows)

    mismatches = 0
    total_old = 0
    total_new = 0

    for i, text in iter_docs(DOCS_PATH):
        a = old_tok.encode_dp(text)
        b = new_tok.encode_dp(text)

        total_old += len(a)
        total_new += len(b)

        if a != b:
            mismatches += 1
            print(f"\nMISMATCH at doc {i}")
            print(f"old_len={len(a)} new_len={len(b)}")
            print(f"text[:300]={repr(text[:300])}")
            print(f"old[:80]={a[:80]}")
            print(f"new[:80]={b[:80]}")
            break

        if (i + 1) % 200 == 0:
            print(
                f"checked={i+1} mismatches={mismatches} "
                f"old_tokens={total_old} new_tokens={total_new}",
                flush=True,
            )

    print("\nDONE")
    print(f"checked_docs={i+1}")
    print(f"mismatches={mismatches}")
    print(f"total_old_tokens={total_old}")
    print(f"total_new_tokens={total_new}")
    print(f"delta={total_new - total_old}")


if __name__ == "__main__":
    main()