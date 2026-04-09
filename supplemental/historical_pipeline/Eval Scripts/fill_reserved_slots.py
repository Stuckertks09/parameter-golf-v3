import json
from pathlib import Path

SRC = Path("workspace/parameter-golf/vocab/vocab_best_v3.jsonl")
OUT = Path("workspace/parameter-golf/vocab/vocab_best_v4_fill_reserved.jsonl")

TOP_10 = [
    "has been",
    "one of the",
    ", however",
    "is not",
    "I don't",
    "of this",
    "and a",
    "as the",
    "there is",
    "have a",
]

with SRC.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows.sort(key=lambda r: int(r["id"]))

reserved_idxs = [
    i for i, r in enumerate(rows)
    if r.get("kind") == "reserved" and r["piece"].startswith("<RESERVED_")
]

if len(reserved_idxs) < len(TOP_10):
    raise ValueError(
        f"Only found {len(reserved_idxs)} empty reserved slots, need {len(TOP_10)}"
    )

existing_pieces = {r["piece"] for r in rows}
dupes = [p for p in TOP_10 if p in existing_pieces]
if dupes:
    raise ValueError(f"These proposed pieces already exist in vocab: {dupes}")

for idx, new_piece in zip(reserved_idxs[:len(TOP_10)], TOP_10):
    old_piece = rows[idx]["piece"]
    rows[idx]["piece"] = new_piece
    rows[idx]["source"] = "reserved_fill_v4"
    rows[idx]["note"] = {
        "replaced": old_piece,
        "reason": "fill_empty_reserved_slot_from_worst_docs",
    }

with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote: {OUT}")
print("Filled reserved slots:")
for idx in reserved_idxs[:len(TOP_10)]:
    row = rows[idx]
    print(f"id={row['id']} piece={row['piece']}")