import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


VOWELS = set("aeiou")

# fragments that often signal weak / accidental leftovers
BAD_PREFIXES = (
    "omething", "therwise", "ccess", "fficient", "iversity", "rogram",
    "arket", "ittle", "etween", "ifferent", "ross", "umber", "ording",
)
BAD_SUFFIXES = (
    "somet", "gener", "techn", "americ", "intere", "partic", "charact",
    "envir", "govern", "educ", "discus", "offici",
)

# these can be useful even if not full words
GOOD_STEMS = {
    "techn", "govern", "market", "program", "develop", "environ", "educ",
    "electr", "financ", "social", "commun", "differ", "between", "general",
    "public", "policy", "history", "culture", "research", "project", "support",
    "success", "include", "control", "system", "network", "available",
    "consider", "possible", "personal", "economic", "energy", "future",
    "question", "example", "quality", "country", "people", "family",
}

FULL_WORD_ALLOW = {
    "different", "between", "program", "little", "market", "open", "always",
    "available", "history", "future", "family", "people", "country",
    "quality", "control", "support", "public", "social", "system",
    "network", "energy", "policy", "research", "project", "possible",
    "personal", "question", "example", "success", "include", "consider",
}

WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vowel_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VOWELS for c in letters) / len(letters)


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def score_subword(text: str) -> tuple[int, list[str]]:
    t = normalize_text(text)
    tl = t.lower()
    reasons = []
    score = 0

    if not t:
        return -999, ["empty"]

    if not WORD_RE.fullmatch(t):
        return -999, ["non_alpha"]

    if len(t) < 4:
        return -999, ["too_short"]

    # strong positives
    if tl in FULL_WORD_ALLOW:
        score += 6
        reasons.append("full_word_allowlist")

    if tl in GOOD_STEMS:
        score += 5
        reasons.append("good_stem_allowlist")

    if len(t) >= 7:
        score += 2
        reasons.append("good_length")

    ar = alpha_ratio(t)
    if ar >= 0.95:
        score += 2
        reasons.append("clean_alpha")

    vr = vowel_ratio(t)
    if 0.20 <= vr <= 0.65:
        score += 2
        reasons.append("plausible_vowel_ratio")

    # looks like a real standalone word
    if tl.isalpha() and tl not in BAD_SUFFIXES and tl not in BAD_PREFIXES:
        score += 1
        reasons.append("clean_word_shape")

    # negatives
    if any(tl.startswith(x) for x in BAD_PREFIXES):
        score -= 5
        reasons.append("bad_prefix_shape")

    if tl in BAD_SUFFIXES or any(tl.endswith(x) for x in BAD_SUFFIXES):
        score -= 5
        reasons.append("bad_suffix_shape")

    # suspicious clipped heads/tails
    if tl.endswith(("ing", "ion", "ity", "ment", "ness", "able", "less", "ward", "ship")):
        score += 1
        reasons.append("useful_suffix_pattern")

    # very awkward endings / starts
    if tl.startswith(("meth", "somet", "omething")):
        score -= 4
        reasons.append("awkward_fragment_start")

    if tl.endswith(("ross", "ener", "omet", "meric")):
        score -= 4
        reasons.append("awkward_fragment_end")

    # medium-length fragments that are ambiguous
    if 4 <= len(t) <= 5 and tl not in FULL_WORD_ALLOW and tl not in GOOD_STEMS:
        score -= 1
        reasons.append("short_ambiguous_fragment")

    return score, reasons


def action_from_score(score: int) -> str:
    if score >= 5:
        return "keep"
    if score >= 2:
        return "review_keep"
    return "drop"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="ngram/merge_candidates_subwords_curated_v1.jsonl")
    parser.add_argument("--reject-jsonl", type=str, default="ngram/merge_candidates_subwords_rejected_v1.jsonl")
    parser.add_argument("--output-csv", type=str, default="ngram/subword_auto_curation_v1.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--keep-review", action="store_true", help="also keep review_keep rows")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    reject_jsonl = Path(args.reject_jsonl)
    output_csv = Path(args.output_csv)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reject_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            if len(rows) >= args.top_n:
                break

    kept = []
    rejected = []
    scored_rows = []

    for rank, row in enumerate(rows, start=1):
        text = row.get("text", "")
        score, reasons = score_subword(text)
        action = action_from_score(score)

        out = {
            "rank": rank,
            "n": row.get("n", 3),
            "ids": row.get("ids", []),
            "freq": row.get("freq", 0),
            "text": text,
            "normalized_text": normalize_text(text).lower(),
            "score": score,
            "action": action,
            "reasons": reasons,
        }
        scored_rows.append(out)

        if action == "keep" or (args.keep_review and action == "review_keep"):
            kept.append({
                "n": out["n"],
                "ids": out["ids"],
                "freq": out["freq"],
                "text": out["text"],
                "normalized_text": out["normalized_text"],
                "class": "curated_subword_auto",
                "score": out["score"],
                "reasons": out["reasons"],
            })
        else:
            rejected.append(out)

    with open(output_jsonl, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(reject_jsonl, "w") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "text", "normalized_text", "freq", "score", "action", "reasons", "ids"])
        for row in scored_rows:
            writer.writerow([
                row["rank"],
                row["text"],
                row["normalized_text"],
                row["freq"],
                row["score"],
                row["action"],
                ";".join(row["reasons"]),
                json.dumps(row["ids"]),
            ])

    print(f"Read {len(rows)} rows")
    print(f"Kept {len(kept)} rows")
    print(f"Rejected {len(rejected)} rows")
    print(f"Saved curated JSONL to {output_jsonl}")
    print(f"Saved rejected JSONL to {reject_jsonl}")
    print(f"Saved audit CSV to      {output_csv}")

    print("\nKept preview:")
    for row in kept[:25]:
        print(f"{row['text']!r:20} score={row['score']:<3} freq={row['freq']} reasons={','.join(row['reasons'])}")

    print("\nRejected preview:")
    for row in rejected[:20]:
        print(f"{row['text']!r:20} score={row['score']:<3} action={row['action']} reasons={','.join(row['reasons'])}")


if __name__ == "__main__":
    main()