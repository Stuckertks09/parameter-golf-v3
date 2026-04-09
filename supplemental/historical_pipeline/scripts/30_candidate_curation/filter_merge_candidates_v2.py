import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


BAD_SUBSTRINGS = [
    "�",
    "://",
    "www.",
    "http",
    "•",
]

# allow a small number of high-value contractions / discourse pieces
ALLOWED_EXACT = {
    "n't",
    "n’t",
    "I'm",
    "I’m",
    "It's",
    "It’s",
}

DISALLOWED_PUNCT_LED_PREFIXES = (
    ". ",
    ", ",
    "; ",
    ": ",
    "! ",
    "? ",
)

WORD_RE = re.compile(r"[A-Za-z]+")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)?$")
SHORT_PHRASE_RE = re.compile(r"^[A-Za-z]+(?:[’'][A-Za-z]+)?(?: [A-Za-z]+(?:[’'][A-Za-z]+)?){1,2}$")
SAFE_TEXT_RE = re.compile(r"^[A-Za-z .'\-’]+$")


def normalize_text_for_dedupe(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def has_bad_substring(text: str) -> bool:
    return any(x in text for x in BAD_SUBSTRINGS)


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(ch.isdigit() for ch in text) / len(text)


def punct_ratio(text: str) -> float:
    if not text:
        return 1.0
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    return punct / len(text)


def tokenish_count(text: str) -> int:
    return len([x for x in text.strip().split() if x])


def starts_with_discourse_punct(text: str) -> bool:
    return text.startswith(DISALLOWED_PUNCT_LED_PREFIXES)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalpha() for ch in text) / len(text)


def classify_candidate(text: str) -> str:
    raw = text
    text = text.strip()

    if not text:
        return "reject_empty"

    if text in ALLOWED_EXACT:
        return "keep_contraction"

    if has_bad_substring(text):
        return "reject_bad_substring"

    if not has_letters(text):
        return "reject_no_letters"

    if digit_ratio(text) >= 0.25:
        return "reject_numeric_heavy"

    if punct_ratio(text) >= 0.30:
        return "reject_punct_heavy"

    if len(text) < 3:
        return "reject_too_short"

    if tokenish_count(text) >= 4:
        return "reject_too_many_words"

    if starts_with_discourse_punct(text):
        return "reject_punct_led_phrase"

    # strong keep: full word or contraction
    if FULL_WORD_RE.fullmatch(text):
        return "keep_full_word"

    # strong keep: 2-3 word phrase
    if SHORT_PHRASE_RE.fullmatch(text):
        return "keep_short_phrase"

    # likely useful subword: mostly alpha, no weird chars
    if SAFE_TEXT_RE.fullmatch(text):
        ar = alpha_ratio(text)
        wc = word_count(text)

        # keep clean single-chunk subwords like "technolog", "govern", "market"
        if wc == 1 and ar >= 0.75:
            return "keep_subword"

        # allow short phrases if very alphabetic
        if wc in (2, 3) and ar >= 0.70:
            return "keep_phrase_like"

    return "reject_other"


def is_keep_label(label: str) -> bool:
    return label.startswith("keep_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="decoded_trigrams.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="ngram/filtered_trigram_candidates_v2.jsonl")
    parser.add_argument("--output-csv", type=str, default="ngram/filtered_trigram_candidates_v2_preview.csv")
    parser.add_argument("--top-n", type=int, default=250)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    seen_ids = set()
    seen_norm_text = set()
    stats = {}

    total_rows = 0

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            total_rows += 1

            ids = tuple(row["ids"])
            freq = row["freq"]
            text = row.get("text", "")
            norm_text = normalize_text_for_dedupe(text)

            label = classify_candidate(text)
            stats[label] = stats.get(label, 0) + 1

            if not is_keep_label(label):
                continue

            if ids in seen_ids:
                stats["reject_duplicate_ids"] = stats.get("reject_duplicate_ids", 0) + 1
                continue

            if norm_text in seen_norm_text:
                stats["reject_duplicate_normalized_text"] = stats.get("reject_duplicate_normalized_text", 0) + 1
                continue

            seen_ids.add(ids)
            seen_norm_text.add(norm_text)

            kept.append({
                "n": 3,
                "ids": list(ids),
                "freq": freq,
                "text": text,
                "normalized_text": norm_text,
                "class": label,
            })

            if len(kept) >= args.top_n:
                break

    with open(output_jsonl, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "text", "normalized_text", "freq", "class", "ids"])
        for i, row in enumerate(kept, start=1):
            writer.writerow([
                i,
                row["text"],
                row["normalized_text"],
                row["freq"],
                row["class"],
                row["ids"],
            ])

    print(f"Read {total_rows} decoded rows")
    print(f"Kept {len(kept)} filtered candidates")
    print(f"Saved JSONL to {output_jsonl}")
    print(f"Saved CSV to   {output_csv}")

    print("\nStats:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    print("\nPreview:")
    for row in kept[:30]:
        print(f"{row['text']!r:20} freq={row['freq']:<10} class={row['class']}")


if __name__ == "__main__":
    main()