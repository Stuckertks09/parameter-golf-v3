import json
import os
import re
import sentencepiece as spm

BASE_MODEL = "data/tokenizers/fineweb_1024_bpe.model"
OUTPUT_ALL = "analysis/baseline_vocab_all.jsonl"
OUTPUT_CLEAN = "analysis/baseline_vocab_clean.jsonl"

os.makedirs("analysis", exist_ok=True)

BYTE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")
SPECIAL = {"<pad>", "<s>", "</s>", "<unk>"}


def normalize_piece(piece: str) -> str:
    # Convert SentencePiece whitespace marker
    return piece.replace("▁", " ")


def is_byte_or_special(piece: str) -> bool:
    return piece in SPECIAL or BYTE_RE.match(piece) is not None


# Load model
sp = spm.SentencePieceProcessor()
ok = sp.load(BASE_MODEL)
if not ok:
    raise FileNotFoundError(f"Could not load baseline model: {BASE_MODEL}")

rows_all = []
rows_clean = []

vocab_size = sp.get_piece_size()

for i in range(vocab_size):
    raw_piece = sp.id_to_piece(i)

    row = {
        "id": i,
        "raw_piece": raw_piece,
        "piece": normalize_piece(raw_piece),
        "is_special_or_byte": is_byte_or_special(raw_piece),
    }

    rows_all.append(row)

    if not row["is_special_or_byte"]:
        # Skip empty results after normalization
        if row["piece"] != "":
            rows_clean.append({
                "id": i,
                "piece": row["piece"],
                "priority": vocab_size - i,
            })


# Save full dump (for debugging)
with open(OUTPUT_ALL, "w", encoding="utf-8") as f:
    for row in rows_all:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Save clean usable tokens
with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
    for row in rows_clean:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


print(f"Wrote {len(rows_all)} total pieces -> {OUTPUT_ALL}")
print(f"Wrote {len(rows_clean)} clean pieces -> {OUTPUT_CLEAN}")

print("\nTop 100 CLEAN baseline pieces:")
for row in rows_clean[:100]:
    print(f'{row["id"]:4d}  {repr(row["piece"])}')