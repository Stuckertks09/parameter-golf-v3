import json
import os

OUTPUT = "analysis/custom_vocab.jsonl"
os.makedirs("analysis", exist_ok=True)

vocab = []

def add(piece: str, kind: str, priority: int):
    vocab.append({
        "piece": piece,
        "kind": kind,
        "priority": priority,
    })

# High-value phrases
phrases = [
    "of the", "in the", "to the", "on the", "and the", "for the",
    "from the", "by the", "with the", "that the", "is a", "it is",
    "to be", "will be", "can be", "such as", "as well as",
    "the same", "one of the", "going to",
]

for i, p in enumerate(phrases):
    add(p, "phrase", 1000 - i)

# Strong words
words = [
    "different", "between", "program", "important", "number",
    "problem", "public", "market", "custom", "available",
    "effect", "direct", "system", "process", "health",
]

for i, w in enumerate(words):
    add(w, "word", 900 - i)

# Productive subwords
subwords = [
    "ing", "tion", "ment", "able", "inter", "pro", "con",
    "ly", "ed", "er", "est", "al", "ity", "ous",
]

for i, s in enumerate(subwords):
    add(s, "subword", 800 - i)

# Basic chars / punctuation / digits
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
punct = [" ", ".", ",", ":", ";", "?", "!", "-", "(", ")", "[", "]", "{", "}", "'", '"', "/", "\\", "\n"]

for i, c in enumerate(chars):
    add(c, "char", 500 - i)

for i, p in enumerate(punct):
    add(p, "punct", 400 - i)

# Deduplicate, keep highest priority
best = {}
for row in vocab:
    piece = row["piece"]
    if piece not in best or row["priority"] > best[piece]["priority"]:
        best[piece] = row

final = sorted(best.values(), key=lambda x: -x["priority"])

with open(OUTPUT, "w", encoding="utf-8") as f:
    for row in final:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(final)} draft entries to {OUTPUT}")