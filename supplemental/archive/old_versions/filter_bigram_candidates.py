import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


BAD_SUBSTRINGS = ["�", "://", "www.", "http", "•"]
ALLOWED_EXACT = {"n't", "n’t", "I'm", "I’m", "It's", "It’s"}
WORD_RE = re.compile(r"[A-Za-z]+")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)?$")
PHRASE_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)? [A-Za-z]+(?:[’'][A-Za-z]+)?$")
SAFE_TEXT_RE = re.compile(r"^[A-Za-z .'\-’]+$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return "reject_empty"
    if t in ALLOWED_EXACT:
        return "keep_contraction"
    if any(x in t for x in BAD_SUBSTRINGS):
        return "reject_bad_substring"
    if not any(ch.isalpha() for ch in t):
        return "reject_no_letters"
    digit_ratio = sum(ch.isdigit() for ch in t) / max(len(t), 1)
    punct_ratio = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t) / max(len(t), 1)
    if digit_ratio >= 0.25:
        return "reject_numeric_heavy"
    if punct_ratio >= 0.30:
        return "reject_punct_heavy"
    if len(t) < 2:
        return "reject_too_short"
    if FULL_WORD_RE.fullmatch(t):
        return "keep_word"
    if PHRASE_RE.fullmatch(t):
        return "keep_phrase"
    if SAFE_TEXT_RE.fullmatch(t):
        alpha_ratio = sum(ch.isalpha() for ch in t) / max(len(t), 1)
        wc = len(WORD_RE.findall(t))
        if wc == 1 and alpha_ratio >= 0.75:
            return "keep_subword"
    return "reject_other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="ngram/filtered_bigram_candidates_v1.jsonl")
    parser.add_argument("--output-csv", type=str, default="ngram/filtered_bigram_candidates_v1_preview.csv")
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    seen_ids = set()
    seen_text = set()
    stats = {}

    with open(args.input, "r") as f:
        for line in f:
            row = json.loads(line)
            ids = tuple(row["ids"])
            text = row.get("text", "")
            norm = normalize_text(text).lower()
            label = classify(text)
            stats[label] = stats.get(label, 0) + 1

            if not label.startswith("keep_"):
                continue
            if ids in seen_ids or norm in seen_text:
                continue

            seen_ids.add(ids)
            seen_text.add(norm)
            kept.append({
                "n": 2,
                "ids": list(ids),
                "freq": row["freq"],
                "text": text,
                "normalized_text": norm,
                "class": label,
            })
            if len(kept) >= args.top_n:
                break

    with open(out_jsonl, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "text", "normalized_text", "freq", "class", "ids"])
        for i, row in enumerate(kept, start=1):
            writer.writerow([i, row["text"], row["normalized_text"], row["freq"], row["class"], row["ids"]])

    print(f"Kept {len(kept)} bigram candidates")
    print(f"Saved JSONL to {out_jsonl}")
    print(f"Saved CSV to {out_csv}")
    print("\nPreview:")
    for row in kept[:30]:
        print(f"{row['text']!r:20} freq={row['freq']} class={row['class']}")


if __name__ == "__main__":
    main()