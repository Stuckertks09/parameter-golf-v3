import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z]+")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+$")
CONTRACTION_RE = re.compile(r"^[A-Za-z]+(?:['’][A-Za-z]+)+$")
PHRASE_RE = re.compile(r"^[A-Za-z]+(?:['’][A-Za-z]+)?(?: [A-Za-z]+(?:['’][A-Za-z]+)?){1,2}$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalpha() for ch in text) / len(text)


def classify_bucket(text: str) -> str | None:
    text = normalize_text(text)

    if not text:
        return None

    # contractions / apostrophe-bearing forms
    if "'" in text:
        if PHRASE_RE.fullmatch(text):
            return "contractions"
        if CONTRACTION_RE.fullmatch(text):
            return "contractions"

    # multi-word phrases
    if " " in text:
        if PHRASE_RE.fullmatch(text):
            return "phrases"
        return None

    # single-token items: full words or subwords
    if FULL_WORD_RE.fullmatch(text):
        # full clean words go with subwords for now
        return "subwords"

    # likely useful clipped subword/stem
    if word_count(text) == 1 and alpha_ratio(text) >= 0.75 and re.fullmatch(r"[A-Za-z]+", text):
        return "subwords"

    return None


def load_rows(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "text", "normalized_text", "freq", "class", "ids"])
        for i, row in enumerate(rows, start=1):
            writer.writerow([
                i,
                row.get("text", ""),
                row.get("normalized_text", normalize_text(row.get("text", ""))).lower(),
                row.get("freq", 0),
                row.get("class", ""),
                row.get("ids", []),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="ngram/filtered_trigram_candidates_v2.jsonl",
        help="Filtered candidate JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ngram",
        help="Directory to write split files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows = load_rows(input_path)

    phrases = []
    subwords = []
    contractions = []

    for row in rows:
        text = row.get("text", "")
        bucket = classify_bucket(text)
        if bucket == "phrases":
            phrases.append(row)
        elif bucket == "subwords":
            subwords.append(row)
        elif bucket == "contractions":
            contractions.append(row)

    # keep original ranking order as inherited from filtered file
    phrases_jsonl = output_dir / "merge_candidates_phrases.jsonl"
    subwords_jsonl = output_dir / "merge_candidates_subwords.jsonl"
    contractions_jsonl = output_dir / "merge_candidates_contractions.jsonl"

    phrases_csv = output_dir / "merge_candidates_phrases_preview.csv"
    subwords_csv = output_dir / "merge_candidates_subwords_preview.csv"
    contractions_csv = output_dir / "merge_candidates_contractions_preview.csv"

    write_jsonl(phrases_jsonl, phrases)
    write_jsonl(subwords_jsonl, subwords)
    write_jsonl(contractions_jsonl, contractions)

    write_csv(phrases_csv, phrases)
    write_csv(subwords_csv, subwords)
    write_csv(contractions_csv, contractions)

    print(f"Input rows:      {len(rows)}")
    print(f"Phrases:         {len(phrases)}")
    print(f"Subwords:        {len(subwords)}")
    print(f"Contractions:    {len(contractions)}")

    print("\nWrote:")
    print(f"  {phrases_jsonl}")
    print(f"  {subwords_jsonl}")
    print(f"  {contractions_jsonl}")
    print(f"  {phrases_csv}")
    print(f"  {subwords_csv}")
    print(f"  {contractions_csv}")

    print("\nPhrase preview:")
    for row in phrases[:15]:
        print(f"  {row.get('text', '')!r} freq={row.get('freq', 0)}")

    print("\nSubword preview:")
    for row in subwords[:15]:
        print(f"  {row.get('text', '')!r} freq={row.get('freq', 0)}")

    print("\nContraction preview:")
    for row in contractions[:15]:
        print(f"  {row.get('text', '')!r} freq={row.get('freq', 0)}")


if __name__ == "__main__":
    main()