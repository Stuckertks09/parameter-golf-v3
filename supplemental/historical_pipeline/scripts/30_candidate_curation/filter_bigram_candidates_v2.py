import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


BAD_SUBSTRINGS = ["�", "://", "www.", "http", "•"]

ALLOWED_EXACT = {
    "n't", "n’t", "I'm", "I’m", "It's", "It’s",
    "of the", "in the", "to the", "on the", "for the",
    "to be", "it is", "as a", "is a", "was a", "are the",
}

WORD_RE = re.compile(r"[A-Za-z]+")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)?$")
PHRASE_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)? [A-Za-z]+(?:[’'][A-Za-z]+)?$")
SAFE_TEXT_RE = re.compile(r"^[A-Za-z .'\-’]+$")

# Common bad boundary-artifact patterns
BAD_SECOND_WORDS = {
    "s", "c", "p", "m", "d", "t", "w", "f", "r", "b", "l", "g", "n"
}
BAD_FIRST_WORDS = {
    "s", "c", "p", "m", "d", "t", "w", "f", "r", "b", "l", "g", "n"
}

# Good function-word phrases that are worth keeping
GOOD_PHRASES = {
    "of the", "in the", "to the", "on the", "for the", "to be", "it is",
    "is a", "was a", "as a", "at the", "by the", "from the", "with the",
    "that the", "and the", "or the", "will be", "can be", "more than",
    "one of", "such as", "going to", "a lot", "a few",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(ch.isdigit() for ch in text) / len(text)


def punct_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / len(text)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalpha() for ch in text) / len(text)


def classify_bigram(text: str) -> str:
    t = normalize_text(text)
    tl = t.lower()

    if not t:
        return "reject_empty"

    if t in ALLOWED_EXACT or tl in GOOD_PHRASES:
        return "keep_allowlist"

    if any(x in t for x in BAD_SUBSTRINGS):
        return "reject_bad_substring"

    if not any(ch.isalpha() for ch in t):
        return "reject_no_letters"

    if digit_ratio(t) >= 0.20:
        return "reject_numeric_heavy"

    if punct_ratio(t) >= 0.25:
        return "reject_punct_heavy"

    wc = word_count(t)
    if wc == 0:
        return "reject_no_words"
    if wc > 2:
        return "reject_too_many_words"

    # Single-token bigram decode: likely subword/full-word
    if wc == 1:
        if not FULL_WORD_RE.fullmatch(t):
            return "reject_non_clean_single"
        if len(t) < 3:
            return "reject_too_short_single"
        return "keep_single_word"

    # Two-token phrase logic
    if wc == 2:
        if not PHRASE_RE.fullmatch(t):
            return "reject_non_clean_phrase"

        words = tl.split()
        w1, w2 = words[0], words[1]

        # Kill classic boundary junk like "the s", "the c"
        if w1 in BAD_FIRST_WORDS or w2 in BAD_SECOND_WORDS:
            return "reject_boundary_artifact"

        # kill tiny word tails unless explicitly allowed
        if len(w1) == 1 or len(w2) == 1:
            return "reject_single_char_word"

        # prefer clean phrase-like units
        if len(w1) >= 2 and len(w2) >= 2:
            return "keep_phrase"

    # fallback: only allow very clean alphabetic text
    if SAFE_TEXT_RE.fullmatch(t) and alpha_ratio(t) >= 0.75:
        return "keep_fallback_clean"

    return "reject_other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="ngram/filtered_bigram_candidates_v2.jsonl")
    parser.add_argument("--output-csv", type=str, default="ngram/filtered_bigram_candidates_v2_preview.csv")
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    seen_ids = set()
    seen_norm = set()
    stats = {}
    total = 0

    with open(args.input, "r") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            ids = tuple(row["ids"])
            text = row.get("text", "")
            norm = normalize_text(text).lower()
            label = classify_bigram(text)
            stats[label] = stats.get(label, 0) + 1

            if not label.startswith("keep_"):
                continue

            if ids in seen_ids or norm in seen_norm:
                stats["reject_duplicate"] = stats.get("reject_duplicate", 0) + 1
                continue

            seen_ids.add(ids)
            seen_norm.add(norm)

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

    print(f"Read {total} decoded bigram rows")
    print(f"Kept {len(kept)} bigram candidates")
    print(f"Saved JSONL to {out_jsonl}")
    print(f"Saved CSV to   {out_csv}")

    print("\nStats:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    print("\nPreview:")
    for row in kept[:40]:
        print(f"{row['text']!r:20} freq={row['freq']:<10} class={row['class']}")


if __name__ == "__main__":
    main()