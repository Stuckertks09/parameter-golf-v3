import sentencepiece as spm
from collections import Counter
from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases

MODEL = "tokenizers/merge_spm.model"
INPUT = "sample_text_large.txt"

sp = spm.SentencePieceProcessor()
sp.load(MODEL)

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

placeholder_to_text = {row["placeholder"]: row["text"] for row in rows}
counts = Counter()
total_tokens = 0
lines = 0

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        text = line.rstrip("\n")
        transformed = replace_phrases(text, compiled)
        pieces = sp.encode(transformed, out_type=str)

        for p in pieces:
            total_tokens += 1
            if p in placeholder_to_text:
                counts[p] += 1

        lines += 1

print("LINES:", lines)
print("TOTAL TOKENS:", total_tokens)
print("PHRASE TOKENS USED:", sum(counts.values()))
print()

print("TOP USED PHRASES:")
for placeholder, count in counts.most_common(100):
    print(f"{count:8d}  {placeholder:30s}  ->  {placeholder_to_text[placeholder]}")