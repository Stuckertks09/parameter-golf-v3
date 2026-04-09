from __future__ import annotations

import json
from pathlib import Path
import sentencepiece as spm

from export_custom_dp_dataset_mp import FastCustomDPTokenizer, load_jsonl

DOCS_PATH = Path("/workspace/parameter-golf/data/docs_selected.jsonl")
VOCAB_PATH = Path("/workspace/parameter-golf/vocab/b650_p36_w32_bs-priority_ps-combined_ws-combined_pin-none_ms10.jsonl")

SP_MODEL = "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

NUM_DOCS = 10000


def iter_docs(path, n):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield i, json.loads(line)["text"]


def main():
    print("loading tokenizers...")

    # baseline
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)

    # custom
    vocab = load_jsonl(VOCAB_PATH)
    custom = FastCustomDPTokenizer(vocab)

    base_total = 0
    custom_total = 0

    worst = []

    for i, text in iter_docs(DOCS_PATH, NUM_DOCS):
        base_ids = sp.encode(text)
        custom_ids = custom.encode_dp(text)

        b = len(base_ids)
        c = len(custom_ids)

        base_total += b
        custom_total += c

        diff = c - b
        worst.append((diff, i, b, c, text[:200]))

        if (i + 1) % 1000 == 0:
            print(
                f"docs={i+1} base={base_total} custom={custom_total} "
                f"ratio={custom_total/base_total:.3f}",
                flush=True
            )

    worst.sort(reverse=True)

    print("\n=== FINAL ===")
    print("baseline:", base_total)
    print("custom:", custom_total)
    print("ratio:", custom_total / base_total)

    print("\nTop offenders:")
    for diff, i, b, c, preview in worst[:10]:
        print(f"\ndoc={i} diff={diff} baseline={b} custom={c}")
        print(preview)


if __name__ == "__main__":
    main()