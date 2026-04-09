import json
import os
import re

BASELINE_INPUT = "analysis/baseline_vocab_clean.jsonl"
MERGE_INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"
OUTPUT = "analysis/hybrid_vocab_1024.jsonl"

TARGET_VOCAB_SIZE = 1024
BYTE_FALLBACK_COUNT = 256
USABLE_BUDGET = TARGET_VOCAB_SIZE - BYTE_FALLBACK_COUNT  # 768

BASELINE_KEEP = 620
PHRASE_KEEP = 80
WORD_SUBWORD_KEEP = 120

BAD_FRAGMENTS = {
    "ed to", "ing to", "ing the", "ed in", "ed by", "ed the",
    "es and", "es of", "ing in", "the st", "ation of", "ed with",
    "es to", "ing for", "the re", "ion of", "to re", "es in",
    "ed and", "er to", "ed for", "the pro", "ed on",
}

ALLOWED_PHRASE_CLASSES = {
    "keep_allowlist",
    "keep_short_phrase",
    "keep_full_word",
    "keep_contraction",
}

ALLOWED_WORD_CLASSES = {
    "curated_subword_auto_v2",
    "keep_full_word",
    "keep_contraction",
}

MIN_SUBWORD_SCORE = 10


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_clean_piece(piece: str) -> bool:
    if piece is None:
        return False
    if piece == "":
        return False
    if "\x00" in piece:
        return False
    return True


def looks_good_phrase(piece: str) -> bool:
    if piece in BAD_FRAGMENTS:
        return False
    if " " not in piece:
        return False
    if len(piece.strip()) < 4:
        return False
    return True


def looks_good_word(piece: str, score: int) -> bool:
    if piece in BAD_FRAGMENTS:
        return False
    if " " in piece:
        return False
    if len(piece) < 3:
        return False
    if score < MIN_SUBWORD_SCORE:
        return False
    return True


def add_piece(pool, piece: str, kind: str, priority: float):
    if not is_clean_piece(piece):
        return
    if piece not in pool:
        pool[piece] = {
            "piece": piece,
            "kind": kind,
            "priority": float(priority),
        }


def main():
    os.makedirs("analysis", exist_ok=True)
    pool = {}

    # 1. Baseline anchors
    baseline_rows = load_jsonl(BASELINE_INPUT)
    baseline_added = 0
    for row in baseline_rows:
        if baseline_added >= BASELINE_KEEP:
            break
        piece = row["piece"]
        add_piece(pool, piece, "baseline_anchor", 1_000_000 - row["id"])
        if piece in pool:
            baseline_added += 1

    # 2. Merge candidates
    merge_rows = load_jsonl(MERGE_INPUT)

    phrase_candidates = []
    word_candidates = []

    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        freq = int(row.get("freq", 0))
        score = int(row.get("score", 0))

        if cls in ALLOWED_PHRASE_CLASSES and looks_good_phrase(piece):
            phrase_candidates.append((freq, piece, cls))

        if cls in ALLOWED_WORD_CLASSES and looks_good_word(piece, score):
            word_candidates.append((freq, piece, cls))

    phrase_candidates.sort(reverse=True)
    word_candidates.sort(reverse=True)

    phrase_added = 0
    for freq, piece, cls in phrase_candidates:
        if phrase_added >= PHRASE_KEEP:
            break
        if piece not in pool:
            add_piece(pool, piece, "phrase_merge", 800_000 + freq)
            if piece in pool:
                phrase_added += 1

    word_added = 0
    for freq, piece, cls in word_candidates:
        if word_added >= WORD_SUBWORD_KEEP:
            break
        if piece not in pool:
            add_piece(pool, piece, "word_subword_merge", 700_000 + freq)
            if piece in pool:
                word_added += 1

    # 3. Coverage chars / punctuation / whitespace
    coverage = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    ) + [
        " ", "\n", "\t",
        ".", ",", ":", ";", "?", "!", "-", "_",
        "(", ")", "[", "]", "{", "}",
        "'", '"', "/", "\\", "@", "#", "$", "%", "&", "*", "+", "=",
        "<", ">", "|", "~", "^", "`",
    ]

    for i, piece in enumerate(coverage):
        add_piece(pool, piece, "coverage_char", 100_000 - i)

    # 4. Reserve fill inside usable budget if still short
    reserve_id = 0
    while len(pool) < USABLE_BUDGET:
        add_piece(pool, f"<RESERVED_{reserve_id}>", "reserved", 1_000 - reserve_id)
        reserve_id += 1

    # 5. Sort usable rows and truncate to usable budget
    usable_rows = list(pool.values())
    usable_rows.sort(key=lambda x: (-x["priority"], len(x["piece"]), x["piece"]))
    usable_rows = usable_rows[:USABLE_BUDGET]

    # 6. Add ALL byte fallbacks
    byte_rows = []
    for b in range(256):
        byte_rows.append({
            "piece": f"<0x{b:02X}>",
            "kind": "byte_fallback",
            "priority": 10_000 - b,
        })

    final_rows = usable_rows + byte_rows

    if len(final_rows) != TARGET_VOCAB_SIZE:
        raise ValueError(f"Expected {TARGET_VOCAB_SIZE} pieces, got {len(final_rows)}")

    # 7. Assign IDs
    for i, row in enumerate(final_rows):
        row["id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_rows)} pieces -> {OUTPUT}")

    counts = {}
    for row in final_rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    print("\nKind breakdown:")
    for kind, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{kind:20s} {count:4d}")

    print("\nTop 120 pieces:")
    for row in final_rows[:120]:
        print(f'{row["id"]:4d}  {row["kind"]:20s}  {repr(row["piece"])}')

    print("\nLast 20 pieces:")
    for row in final_rows[-20:]:
        print(f'{row["id"]:4d}  {row["kind"]:20s}  {repr(row["piece"])}')


if __name__ == "__main__":
    main()