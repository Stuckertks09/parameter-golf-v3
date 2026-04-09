import argparse
import json
import re
from pathlib import Path


BAD_SUBSTRINGS = [
    "�", "://", "www.", "http", "•"
]

ALLOWED_SHORT_EXACT = {
    "n't", "I’m", "I'm", "It’s", "It's", ". But", ". They", ". The", ". This", ". When", ". She"
}


def looks_numeric_heavy(text: str) -> bool:
    if not text:
        return True
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(len(text), 1) >= 0.4


def looks_punct_heavy(text: str) -> bool:
    if not text:
        return True
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    return punct / max(len(text), 1) >= 0.35


def has_bad_substring(text: str) -> bool:
    return any(x in text for x in BAD_SUBSTRINGS)


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def token_countish(text: str) -> int:
    return len([x for x in text.strip().split() if x])


def keep_candidate(text: str) -> bool:
    if not text:
        return False

    text = text.strip()
    if not text:
        return False

    if text in ALLOWED_SHORT_EXACT:
        return True

    if has_bad_substring(text):
        return False

    if not has_letters(text):
        return False

    if looks_numeric_heavy(text):
        return False

    if looks_punct_heavy(text):
        return False

    # reject extremely short fragments unless they contain letters and look normal
    if len(text) < 3:
        return False

    # reject odd leftovers that are mostly single-char chunks/spaces
    if token_countish(text) >= 4:
        return False

    # allow alphabetic words / subwords / short phrases
    if re.fullmatch(r"[A-Za-z]+(?:[’'][A-Za-z]+)?", text):
        return True

    if re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){1,2}", text):
        return True

    if re.fullmatch(r"\.? ?[A-Za-z]+(?:[’'][A-Za-z]+)?", text):
        return True

    if re.fullmatch(r"[A-Za-z]+(?:\.[A-Za-z]+)?", text):
        return True

    # fallback: allow mostly alphabetic text with spaces/apostrophes/periods/hyphens
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'’-")
    if all(ch in allowed for ch in text):
        alpha = sum(ch.isalpha() for ch in text)
        if alpha / max(len(text), 1) >= 0.6:
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="decoded_trigrams.jsonl")
    parser.add_argument("--output", type=str, default="ngram/filtered_trigram_candidates.jsonl")
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    kept = []
    seen = set()
    total = 0

    with open(args.input, "r") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            ids = tuple(row["ids"])
            text = row.get("text", "")
            freq = row["freq"]

            if ids in seen:
                continue

            if keep_candidate(text):
                seen.add(ids)
                kept.append({
                    "n": 3,
                    "ids": list(ids),
                    "freq": freq,
                    "text": text
                })

            if len(kept) >= args.top_n:
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read {total} decoded rows")
    print(f"Kept {len(kept)} filtered candidates")
    print(f"Saved to {args.output}")

    print("\nPreview:")
    for row in kept[:25]:
        print(f"{row['text']!r:20} freq={row['freq']}")


if __name__ == "__main__":
    main()