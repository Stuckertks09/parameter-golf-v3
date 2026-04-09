import json
import os
import re

BASELINE_INPUT = "analysis/baseline_vocab_clean.jsonl"
MERGE_INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"

TARGET_VOCAB_SIZE = 1024

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASELINE_KEEP = 640
PHRASE_KEEP = 40
WORD_SUBWORD_KEEP = 30

BASELINE_SORT_MODE = "priority"   # priority | id | priority_len
PHRASE_SORT_MODE = "combined"     # freq | score | combined
WORD_SORT_MODE = "combined"       # freq | score | combined

RUN_TAG = f"b-{BASELINE_SORT_MODE}_p-{PHRASE_SORT_MODE}_w-{WORD_SORT_MODE}"
OUTPUT = f"analysis/custom_vocab_full_{RUN_TAG}.jsonl"

UNICODE_MUST_HAVE = [
    "\u2019", "\u2018", "\u201c", "\u201d",
    "\u2014", "\u2013", "\u2026",
    "\u2019s", "n\u2019t", "\u2019re", "\u2019ve", "\u2019ll", "\u2019d", "\u2019m",
    ". ", ", ", ": ", "? ", "! ", "; ",
]

PIN_CUSTOM_PIECES = [
    "There are",
    "such as",
    "as well as",
    "able to",
    "public",
    "quality",
    "the same",
    "the world",
    "a lot",
    "a few",
    "going to",
    "I’m",
    "social",
    "success",
    "personal",
    "organiz",
    "govern",
    "profess",
]

ALLOWED_PHRASE_CLASSES = {
    "phrase",
    "keep_short_phrase",
    "keep_allowlist",
}

ALLOWED_WORD_CLASSES = {
    "keep_full_word",
    "keep_contraction",
    "curated_subword_auto_v2",
}

MIN_SUBWORD_SCORE = 14
MAX_PHRASE_WORDS = 3

BAD_FRAGMENTS = {
    "ed to", "ing to", "ing the", "ed in", "ed by", "ed the",
    "es and", "es of", "ing in", "the st", "ation of", "ed with",
    "es to", "ing for", "the re", "ion of", "to re", "es in",
    "ed and", "er to", "ed for", "the pro", "ed on",
    "s of the", "ed by the", "ed in the", "umber of", "ording to",
    "a litt", "lot of",
}

BYTE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")
FULL_WORD_PHRASE_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+){1,2}$")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+$")
CONTRACTION_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+|\u2019[A-Za-z]+)$")
LETTER_RE = re.compile(r"[A-Za-z]")
SINGLE_ALPHA_RE = re.compile(r"^[A-Za-z]$")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_byte_piece(piece: str) -> bool:
    return BYTE_RE.match(piece) is not None


def is_clean_piece(piece: str) -> bool:
    return bool(piece) and "\x00" not in piece


def has_letter(piece: str) -> bool:
    return LETTER_RE.search(piece) is not None


def is_single_alpha(piece: str) -> bool:
    return SINGLE_ALPHA_RE.match(piece) is not None


def phrase_word_count(piece: str) -> int:
    return len([p for p in piece.split(" ") if p])


def has_bad_boundary_smell(piece: str) -> bool:
    if piece in BAD_FRAGMENTS:
        return True

    bad_prefixes = ("ed ", "ing ", "es ", "er ", "ion ", "ation ", "s ")
    bad_suffixes = (" ed", " ing", " es", " er", " ion", " tion", " s")

    if piece.startswith(bad_prefixes):
        return True
    if piece.endswith(bad_suffixes):
        return True

    bad_chunks = ("umber ", "ording ", "ation ", "ition ", "the pro", "the re", "the st")
    return any(chunk in piece for chunk in bad_chunks)


def phrase_ok(piece: str) -> bool:
    if not is_clean_piece(piece):
        return False
    if piece in BAD_FRAGMENTS:
        return False
    if is_single_alpha(piece):
        return False
    if not FULL_WORD_PHRASE_RE.match(piece):
        return False
    if phrase_word_count(piece) > MAX_PHRASE_WORDS:
        return False
    if has_bad_boundary_smell(piece):
        return False
    return True


def word_or_subword_ok(piece: str, cls: str, score: int) -> bool:
    if not is_clean_piece(piece):
        return False
    if piece in BAD_FRAGMENTS:
        return False
    if is_single_alpha(piece):
        return False
    if has_bad_boundary_smell(piece):
        return False

    if cls == "keep_full_word":
        return FULL_WORD_RE.match(piece) is not None and len(piece) >= 3

    if cls == "keep_contraction":
        return CONTRACTION_RE.match(piece) is not None

    if cls == "curated_subword_auto_v2":
        if " " in piece:
            return False
        if len(piece) < 3:
            return False
        if score < MIN_SUBWORD_SCORE:
            return False
        if not has_letter(piece):
            return False
        return True

    return False


# --------------------------------------------------
# SORTERS
# --------------------------------------------------
def baseline_sort_key(row):
    priority = float(row.get("priority", 0))
    source_id = int(row.get("id", 10**9))
    piece = row.get("piece", "")

    if BASELINE_SORT_MODE == "priority":
        return (-priority, source_id)
    elif BASELINE_SORT_MODE == "id":
        return (source_id,)
    elif BASELINE_SORT_MODE == "priority_len":
        return (-priority, len(piece), piece)
    else:
        raise ValueError(f"Unknown BASELINE_SORT_MODE: {BASELINE_SORT_MODE}")


def phrase_combined_value(row):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")

    bonus = 0.0
    if piece in {
        "of the", "in the", "to the", "on the", "for the", "from the",
        "as a", "one of", "such as", "as well as", "There are",
        "a lot", "a few", "the same", "able to", "going to"
    }:
        bonus += 250000
    if piece in PIN_CUSTOM_PIECES:
        bonus += 500000
    if piece.islower():
        bonus += 1000

    return freq * 1000 + score + bonus


def phrase_sort_key(row):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")

    if PHRASE_SORT_MODE == "freq":
        return (-freq, len(piece), piece)
    elif PHRASE_SORT_MODE == "score":
        return (-score, len(piece), piece)
    elif PHRASE_SORT_MODE == "combined":
        return (-phrase_combined_value(row), len(piece), piece)
    else:
        raise ValueError(f"Unknown PHRASE_SORT_MODE: {PHRASE_SORT_MODE}")


def word_combined_value(row):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    cls = row.get("class", "")
    piece = row.get("text", "")

    bonus = 0.0
    if cls == "keep_full_word":
        bonus += 50000
    elif cls == "keep_contraction":
        bonus += 30000
    elif cls == "curated_subword_auto_v2":
        bonus += 10000

    if FULL_WORD_RE.match(piece):
        bonus += 5000
    if piece in PIN_CUSTOM_PIECES:
        bonus += 500000

    return freq * 1000 + score + bonus


def word_sort_key(row):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")

    if WORD_SORT_MODE == "freq":
        return (-freq, len(piece), piece)
    elif WORD_SORT_MODE == "score":
        return (-score, len(piece), piece)
    elif WORD_SORT_MODE == "combined":
        return (-word_combined_value(row), len(piece), piece)
    else:
        raise ValueError(f"Unknown WORD_SORT_MODE: {WORD_SORT_MODE}")


def main():
    os.makedirs("analysis", exist_ok=True)

    final_rows = []
    used = set()

    def add_piece(piece, kind, source_priority, meta=None):
        if piece in used:
            return False
        if not is_clean_piece(piece):
            return False
        if kind not in {"unicode_required", "byte_fallback"} and is_single_alpha(piece):
            return False

        row = {
            "piece": piece,
            "kind": kind,
            "priority": float(source_priority),
        }
        if meta:
            row.update(meta)

        final_rows.append(row)
        used.add(piece)
        return True

    # 1. Unicode required
    unicode_added = 0
    for i, piece in enumerate(UNICODE_MUST_HAVE):
        if add_piece(piece, "unicode_required", 2_000_000 - i):
            unicode_added += 1

    # 2. Baseline anchors
    baseline_rows = load_jsonl(BASELINE_INPUT)
    baseline_rows = [r for r in baseline_rows if is_clean_piece(r.get("piece", ""))]
    baseline_rows.sort(key=baseline_sort_key)

    baseline_added = 0
    for row in baseline_rows:
        piece = row["piece"]
        if is_byte_piece(piece):
            continue
        if is_single_alpha(piece):
            continue

        if add_piece(
            piece,
            "baseline_anchor",
            row.get("priority", 0),
            meta={"source_id": row.get("id")}
        ):
            baseline_added += 1
            if baseline_added >= BASELINE_KEEP:
                break

    # 3. Merge rows
    merge_rows = load_jsonl(MERGE_INPUT)

    # 4. Pinned custom pieces first
    pin_map = {}
    for row in merge_rows:
        text = row.get("text", "")
        if text not in pin_map:
            pin_map[text] = row

    pinned_phrase_added = 0
    pinned_word_added = 0

    for piece in PIN_CUSTOM_PIECES:
        row = pin_map.get(piece)
        if not row:
            continue

        cls = row.get("class", "")
        score = int(row.get("score", 0))

        if cls in ALLOWED_PHRASE_CLASSES and phrase_ok(piece):
            if add_piece(
                piece,
                "phrase_merge",
                phrase_combined_value(row) + 1_000_000,
                meta={
                    "class": cls,
                    "freq": row.get("freq", 0),
                    "score": row.get("score", 0),
                    "pinned": True,
                },
            ):
                pinned_phrase_added += 1

        elif cls in ALLOWED_WORD_CLASSES and word_or_subword_ok(piece, cls, score):
            if add_piece(
                piece,
                "word_subword_merge",
                word_combined_value(row) + 1_000_000,
                meta={
                    "class": cls,
                    "freq": row.get("freq", 0),
                    "score": row.get("score", 0),
                    "pinned": True,
                },
            ):
                pinned_word_added += 1

    # 5. Remaining phrase merges
    phrase_candidates = []
    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        if cls not in ALLOWED_PHRASE_CLASSES:
            continue
        if not phrase_ok(piece):
            continue
        phrase_candidates.append(row)

    phrase_candidates.sort(key=phrase_sort_key)

    phrase_added = 0
    for row in phrase_candidates:
        if add_piece(
            row["text"],
            "phrase_merge",
            phrase_combined_value(row),
            meta={
                "class": row.get("class"),
                "freq": row.get("freq", 0),
                "score": row.get("score", 0),
                "pinned": False,
            },
        ):
            phrase_added += 1
            if phrase_added + pinned_phrase_added >= PHRASE_KEEP:
                break

    # 6. Remaining word/subword merges
    word_candidates = []
    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        score = int(row.get("score", 0))

        if cls not in ALLOWED_WORD_CLASSES:
            continue
        if not word_or_subword_ok(piece, cls, score):
            continue
        word_candidates.append(row)

    word_candidates.sort(key=word_sort_key)

    word_added = 0
    for row in word_candidates:
        if add_piece(
            row["text"],
            "word_subword_merge",
            word_combined_value(row),
            meta={
                "class": row.get("class"),
                "freq": row.get("freq", 0),
                "score": row.get("score", 0),
                "pinned": False,
            },
        ):
            word_added += 1
            if word_added + pinned_word_added >= WORD_SUBWORD_KEEP:
                break

    # 7. Full 256 byte fallbacks
    byte_added = 0
    for byte_id in range(256):
        piece = f"<0x{byte_id:02X}>"
        if add_piece(piece, "byte_fallback", -100000 - byte_id):
            byte_added += 1

    # 8. Fill remainder
    reserve_id = 0
    while len(final_rows) < TARGET_VOCAB_SIZE:
        piece = f"<RESERVED_{reserve_id}>"
        if add_piece(piece, "reserved", -200000 - reserve_id):
            reserve_id += 1

    final_rows = final_rows[:TARGET_VOCAB_SIZE]
    for i, row in enumerate(final_rows):
        row["id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_rows)} pieces -> {OUTPUT}")
    print()
    print("Config:")
    print(f"  BASELINE_SORT_MODE = {BASELINE_SORT_MODE}")
    print(f"  PHRASE_SORT_MODE   = {PHRASE_SORT_MODE}")
    print(f"  WORD_SORT_MODE     = {WORD_SORT_MODE}")
    print()
    print("Budget summary:")
    print(f"  unicode required   : {unicode_added}")
    print(f"  baseline anchors   : {baseline_added}")
    print(f"  pinned phrases     : {pinned_phrase_added}")
    print(f"  pinned words       : {pinned_word_added}")
    print(f"  extra phrases      : {phrase_added}")
    print(f"  extra words        : {word_added}")
    print(f"  byte fallbacks     : {byte_added}")
    print(f"  reserved           : {reserve_id}")
    print()

    kind_counts = {}
    for row in final_rows:
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1

    print("Kind breakdown:")
    for kind, count in sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{kind:20s} {count:4d}")

    print("\nTop 120 pieces:")
    for row in final_rows[:120]:
        print(f'{row["id"]:4d}  {row["kind"]:20s}  {repr(row["piece"])}')


if __name__ == "__main__":
    main()