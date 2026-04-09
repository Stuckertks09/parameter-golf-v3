import json
import os
import re

INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"
PHRASE_OUTPUT = "analysis/forced_phrases.jsonl"
BOOST_OUTPUT = "analysis/boost_terms.txt"

os.makedirs("analysis", exist_ok=True)

KEEP_PHRASES = {
    "the same",
    "as well",
    "such as",
    "by the",
    "to get",
    "one of the",
    "able to",
    "the end",
    "There are",
    "the world",
    "as well as",
    "from the",
    "it is",
    "want to",
    "will be",
    "as a",
    "that the",
    "for the",
    "of the",
    "to the",
    "in the",
    "and the",
    "has been",
    "is the",
    "you can",
    "If you",
    "is a",
    "at the",
    "to be",
    "can be",
    "with the",
    "on the",
    "going to",
    "a few",
    "one of",
    "lot of",
    "a lot",
    "as the",
}

BOOST_CLASSES = {
    "curated_subword_auto_v2",
    "keep_full_word",
    "keep_contraction",
    "keep_allowlist",
    "keep_phrase",
    "keep_short_phrase",
}

MAX_BOOST = 300

def is_clean(text: str) -> bool:
    if not text or "  " in text:
        return False
    if len(text.strip()) < 2:
        return False
    return True

def make_placeholder(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")
    return f"PHRASE_{cleaned}"

forced_final = []
boost_rows = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        text = row["text"].strip()
        cls = row.get("class", "")
        freq = int(row.get("freq", 0))

        if not is_clean(text):
            continue

        if text in KEEP_PHRASES:
            forced_final.append({
                "text": text,
                "placeholder": make_placeholder(text),
                "freq": freq,
                "class": cls,
            })

        if cls in BOOST_CLASSES:
            if cls == "curated_subword_auto_v2":
                score = int(row.get("score", 0))
                if score < 10:
                    continue
            if len(text) < 4 and " " not in text:
                continue
            boost_rows.append((text, freq, cls))

boost_rows.sort(key=lambda x: -x[1])

boost_seen = set()
boost_final = []
for text, freq, cls in boost_rows:
    if text not in boost_seen:
        boost_final.append(text)
        boost_seen.add(text)
    if len(boost_final) >= MAX_BOOST:
        break

forced_final.sort(key=lambda x: -x["freq"])

with open(PHRASE_OUTPUT, "w", encoding="utf-8") as f:
    for row in forced_final:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(BOOST_OUTPUT, "w", encoding="utf-8") as f:
    for item in boost_final:
        f.write(item + "\n")

print(f"Forced phrases: {len(forced_final)} -> {PHRASE_OUTPUT}")
print(f"Boost terms:    {len(boost_final)} -> {BOOST_OUTPUT}")