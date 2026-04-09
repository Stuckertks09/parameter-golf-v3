import sentencepiece as spm
from collections import defaultdict
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

phrase_stats = defaultdict(lambda: {"count": 0, "saved": 0})

for row in rows:
    phrase_stats[row["placeholder"]]["text"] = row["text"]

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        text = line.rstrip("\n")

        base_ids = base_sp.encode(text, out_type=int)

        transformed = replace_phrases(text, compiled)
        new_ids = new_sp.encode(transformed, out_type=int)
        new_pieces = new_sp.encode(transformed, out_type=str)

        base_len = len(base_ids)
        new_len = len(new_ids)

        for p in new_pieces:
            if p in phrase_stats:
                phrase_stats[p]["count"] += 1

        saved = base_len - new_len

        if saved > 0:
            # distribute savings to phrases in this line
            used_phrases = [p for p in new_pieces if p in phrase_stats]
            if used_phrases:
                per = saved / len(used_phrases)
                for p in used_phrases:
                    phrase_stats[p]["saved"] += per

print("PHRASE EFFICIENCY:\n")

results = []
for k, v in phrase_stats.items():
    if v["count"] > 0:
        efficiency = v["saved"] / v["count"]
        results.append((efficiency, v["count"], k, v["text"]))

results.sort(reverse=True)

for eff, count, placeholder, text in results[:50]:
    print(f"{eff:.3f}  {count:6d}  {placeholder:30s} -> {text}")