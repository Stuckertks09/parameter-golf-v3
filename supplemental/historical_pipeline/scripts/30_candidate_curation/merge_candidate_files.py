import argparse
import json
import unicodedata
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def read_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files to merge in priority order",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ngram/merge_candidates_final_v1.jsonl",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    seen_norm_text = set()
    merged = []

    for input_file in args.inputs:
        path = Path(input_file)
        rows = read_jsonl(path)

        for row in rows:
            ids = tuple(row.get("ids", []))
            text = row.get("text", "")
            norm_text = normalize_text(text)

            if ids in seen_ids:
                continue
            if norm_text and norm_text in seen_norm_text:
                continue

            seen_ids.add(ids)
            if norm_text:
                seen_norm_text.add(norm_text)

            merged.append(row)

    with open(output_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(args.inputs)} files")
    print(f"Wrote {len(merged)} unique candidates to {output_path}")

    print("\nPreview:")
    for row in merged[:30]:
        print(f"{row.get('text', '')!r:20} freq={row.get('freq', 0)}")


if __name__ == "__main__":
    main()