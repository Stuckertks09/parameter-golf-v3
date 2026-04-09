import sentencepiece as spm
from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases

BASE = "data/tokenizers/fineweb_1024_bpe.model"
NEW = "tokenizers/merge_spm.model"
INPUT = "sample_text_large.txt"

base_sp = spm.SentencePieceProcessor()
base_sp.load(BASE)

new_sp = spm.SentencePieceProcessor()
new_sp.load(NEW)

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

base_total = 0
new_total = 0
lines = 0

examples = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        text = line.rstrip("\n")

        base_ids = base_sp.encode(text, out_type=int)

        transformed = replace_phrases(text, compiled)
        new_ids = new_sp.encode(transformed, out_type=int)

        base_total += len(base_ids)
        new_total += len(new_ids)
        lines += 1

        if len(examples) < 10 and len(new_ids) < len(base_ids):
            examples.append({
                "text": text,
                "transformed": transformed,
                "base_len": len(base_ids),
                "new_len": len(new_ids),
                "base_pieces": base_sp.encode(text, out_type=str),
                "new_pieces": new_sp.encode(transformed, out_type=str),
            })

print("LINES:", lines)
print("BASE TOTAL:", base_total)
print("NEW TOTAL:", new_total)
print("ABS SAVED:", base_total - new_total)
print("PCT SAVED:", (base_total - new_total) / base_total if base_total else 0.0)

print("\nEXAMPLES:")
for ex in examples:
    print("=" * 80)
    print("TEXT:       ", ex["text"])
    print("XFORM:      ", ex["transformed"])
    print("BASE LEN:   ", ex["base_len"])
    print("NEW LEN:    ", ex["new_len"])
    print("BASE PIECES:", ex["base_pieces"])
    print("NEW PIECES: ", ex["new_pieces"])