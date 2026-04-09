import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


VOWELS = set("aeiou")

WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")

# Strong full words / clean standalone items
FULL_WORD_ALLOW = {
    "different", "between", "program", "little", "market", "open", "always",
    "available", "effect", "custom", "important", "direct", "problem", "number",
    "public", "social", "quality", "possible", "success", "support", "control",
    "system", "network", "energy", "policy", "research", "project", "history",
    "future", "family", "people", "country", "question", "example", "consider",
    "personal", "include"
}

# Good reusable root-like stems / left-side fragments
GOOD_PREFIX_STEMS = {
    "techn", "govern", "educ", "organiz", "profess", "individ", "develop",
    "environ", "electr", "financ", "commun", "differ", "general", "public",
    "social", "research", "project", "support", "success", "control", "system",
    "network", "policy", "quality", "possib", "person", "econom", "histor",
    "cultu", "quest", "exampl", "market", "program", "avail", "custom"
}

# Canonical suffix-like chunks that can still be defensible
GOOD_SUFFIX_STEMS = {
    "tion", "ment", "ness", "able", "less", "ship", "ward", "ally", "ical",
    "ious", "ence", "ance", "ity", "ism", "ist"
}

# Known bad chopped tails / ugly leftovers
BAD_FRAGMENTS = {
    "somet", "omething", "ross", "gener", "terest", "chnolog", "vernment",
    "owever", "alread", "umber", "ording", "meric", "ffectiv", "quired",
    "ticle", "lieve", "selv", "nology", "ological", "utiful", "ecut", "ccess"
}

# Starts that often indicate a chopped tail rather than a root
BAD_STARTS = (
    "ther", "omething", "ross", "umber", "ording", "lread", "owe", "ecut",
    "chn", "ver", "meric", "ticle", "ffect", "quired", "lieve", "selv"
)

# Ends that often indicate awkward clipped endings
BAD_ENDS = (
    "ments", "olog", "ever", "read", "ross", "meric", "ticle", "lieve", "selv"
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def vowel_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VOWELS for c in letters) / len(letters)


def is_prefix_like(t: str) -> bool:
    # Root/stem-like chunks often start cleanly and are 4-8 chars
    if not (4 <= len(t) <= 9):
        return False
    if not t[0].isalpha():
        return False
    if t.lower() in GOOD_PREFIX_STEMS:
        return True
    if any(t.lower().startswith(b) for b in BAD_STARTS):
        return False
    # Prefer things that look like beginnings of words, not endings
    return t[0].lower() not in {"h", "v"} and vowel_ratio(t) >= 0.20


def is_suffix_like(t: str) -> bool:
    tl = t.lower()
    return any(tl.endswith(s) for s in GOOD_SUFFIX_STEMS) or tl in GOOD_SUFFIX_STEMS


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

    if tl in BAD_FRAGMENTS:
        return -999, ["explicit_bad_fragment"]

    # Strong full words
    if tl in FULL_WORD_ALLOW:
        score += 7
        reasons.append("full_word_allowlist")

    # Strong approved roots
    if tl in GOOD_PREFIX_STEMS:
        score += 6
        reasons.append("good_prefix_stem_allowlist")

    # Canonical suffix chunks are allowed but lower-priority
    if tl in GOOD_SUFFIX_STEMS:
        score += 3
        reasons.append("good_suffix_stem_allowlist")

    # Clean shape
    if alpha_ratio(t) >= 0.98:
        score += 2
        reasons.append("clean_alpha")

    vr = vowel_ratio(t)
    if 0.22 <= vr <= 0.62:
        score += 2
        reasons.append("plausible_vowel_ratio")

    if len(t) >= 6:
        score += 1
        reasons.append("good_length")

    # Structural judgment: prefix-like good, suffix-like neutral-to-good
    if is_prefix_like(t):
        score += 3
        reasons.append("prefix_like")

    if is_suffix_like(t):
        score += 1
        reasons.append("suffix_like")

    # Penalties for awkward chopped tails
    if any(tl.startswith(b) for b in BAD_STARTS):
        score -= 5
        reasons.append("bad_start_shape")

    if any(tl.endswith(b) for b in BAD_ENDS):
        score -= 4
        reasons.append("bad_end_shape")

    # Penalize obvious tail-only look
    if tl.endswith(("ment", "tion", "ness", "able", "less")) and tl not in GOOD_SUFFIX_STEMS:
        score -= 2
        reasons.append("tail_only_feel")

    # Penalize ambiguous short chunks unless explicitly blessed
    if 4 <= len(t) <= 5 and tl not in FULL_WORD_ALLOW and tl not in GOOD_PREFIX_STEMS and tl not in GOOD_SUFFIX_STEMS:
        score -= 1
        reasons.append("short_ambiguous_fragment")

    return score, reasons


def action_from_score(score: int) -> str:
    if score >= 7:
        return "keep"
    if score >= 4:
        return "review_keep"
    return "drop"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="ngram/merge_candidates_subwords_curated_v2.jsonl")
    parser.add_argument("--reject-jsonl", type=str, default="ngram/merge_candidates_subwords_rejected_v2.jsonl")
    parser.add_argument("--output-csv", type=str, default="ngram/subword_auto_curation_v2.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--keep-review", action="store_true")
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
                "class": "curated_subword_auto_v2",
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