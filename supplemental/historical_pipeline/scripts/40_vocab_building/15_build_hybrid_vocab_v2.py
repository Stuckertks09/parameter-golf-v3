import json
import os
import re

BASELINE_INPUT = "analysis/baseline_vocab_clean.jsonl"
MERGE_INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"
OUTPUT = "analysis/custom_vocab_full_v7.jsonl"

TARGET_VOCAB_SIZE = 1024

BASELINE_KEEP = 680
PHRASE_KEEP = 28
WORD_SUBWORD_KEEP = 24

# These MUST be in vocab — curly quotes and unicode apostrophes are everywhere in FineWeb
# Without these they explode to 3 bytes each
UNICODE_MUST_HAVE = [
    "\u2019",       # ' right single quote (most common apostrophe in FineWeb)
    "\u2018",       # ' left single quote
    "\u201c",       # " left double quote
    "\u201d",       # " right double quote
    "\u2014",       # — em dash
    "\u2013",       # – en dash
    "\u2026",       # … ellipsis
    "\u2019s",      # 's
    "n\u2019t",     # n't
    "\u2019re",     # 're
    "\u2019ve",     # 've
    "\u2019ll",     # 'll
    "\u2019d",      # 'd
    "\u2019m",      # 'm
    ". ",
    ", ",
    ": ",
    "? ",
    "! ",
    "; ",
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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_byte_piece(piece):
    return BYTE_RE.match(piece) is not None


def is_clean_piece(piece):
    if not piece:
        return False
    if "\x00" in piece:
        return False
    return True


def phrase_word_count(piece):
    return len([p for p in piece.split(" ") if p])


def has_letter(piece):
    return LETTER_RE.search(piece) is not None


def has_bad_boundary_smell(piece):
    if piece in BAD_FRAGMENTS:
        return True
    bad_prefixes = ("ed ", "ing ", "es ", "er ", "ion ", "ation ", "s ")
    bad_suffixes = (" ed", " ing", " es", " er", " ion", " tion", " s")
    if piece.startswith(bad_prefixes):
        return True
    if piece.endswith(bad_suffixes):
        return True
    bad_chunks = ("umber ", "ording ", "ation ", "ition ", "the pro", "the re", "the st")
    for chunk in bad_chunks:
        if chunk in piece:
            return True
    return False


def phrase_ok(piece):
    if not is_clean_piece(piece):
        return False
    if piece in BAD_FRAGMENTS:
        return False
    if not FULL_WORD_PHRASE_RE.match(piece):
        return False
    if phrase_word_count(piece) > MAX_PHRASE_WORDS:
        return False
    if has_bad_boundary_smell(piece):
        return False
    return True


def word_or_subword_ok(piece, cls, score):
    if not is_clean_piece(piece):
        return False
    if piece in BAD_FRAGMENTS:
        return False
    if has_bad_boundary_smell(piece):
        return False
    if cls == "keep_full_word":
        return FULL_WORD_RE.match(piece) is not None
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


def phrase_priority(row):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")
    bonus = 0.0
    if piece in {"of the", "in the", "to the", "on the", "for the", "from the", "as a", "one of"}:
        bonus += 100000
    if piece.islower():
        bonus += 1000
    return freq * 1000 + score + bonus


def word_priority(row):
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
    return freq * 1000 + score + bonus


def main():
    os.makedirs("analysis", exist_ok=True)

    final_rows = []
    used = set()

    def add_piece(piece, kind, source_priority, meta=None):
        if piece in used:
            return False
        if not is_clean_piece(piece):
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

    # --------------------------------------------------
    # 1. Unicode must-haves — guaranteed before anything else
    # --------------------------------------------------
    unicode_added = 0
    for piece in UNICODE_MUST_HAVE:
        if add_piece(piece, "unicode_required", 1_500_000):
            unicode_added += 1

    # --------------------------------------------------
    # 2. Baseline anchors sorted by original ID
    # --------------------------------------------------
    baseline_rows = load_jsonl(BASELINE_INPUT)
    baseline_rows = [r for r in baseline_rows if is_clean_piece(r.get("piece", ""))]
    baseline_rows.sort(key=lambda r: r.get("id", 9999))

    baseline_added = 0
    for row in baseline_rows:
        piece = row["piece"]
        if is_byte_piece(piece):
            continue
        if add_piece(piece, "baseline_anchor", 1_000_000 - row.get("id", 0),
                     meta={"source_id": row.get("id")}):
            baseline_added += 1
            if baseline_added >= BASELINE_KEEP:
                break

    # --------------------------------------------------
    # 3. Phrase merges
    # --------------------------------------------------
    merge_rows = load_jsonl(MERGE_INPUT)

    phrase_candidates = []
    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        if cls not in ALLOWED_PHRASE_CLASSES:
            continue
        if not phrase_ok(piece):
            continue
        phrase_candidates.append(row)

    phrase_candidates.sort(key=phrase_priority, reverse=True)

    phrase_added = 0
    for row in phrase_candidates:
        if add_piece(row["text"], "phrase_merge", phrase_priority(row),
                     meta={"class": row.get("class"), "freq": row.get("freq", 0), "score": row.get("score", 0)}):
            phrase_added += 1
            if phrase_added >= PHRASE_KEEP:
                break

    # --------------------------------------------------
    # 4. Word / subword merges
    # --------------------------------------------------
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

    word_candidates.sort(key=word_priority, reverse=True)

    word_added = 0
    for row in word_candidates:
        if add_piece(row["text"], "word_subword_merge", word_priority(row),
                     meta={"class": row.get("class"), "freq": row.get("freq", 0), "score": row.get("score", 0)}):
            word_added += 1
            if word_added >= WORD_SUBWORD_KEEP:
                break

    # --------------------------------------------------
    # 5. Byte fallback — all 256
    # --------------------------------------------------
    byte_added = 0
    for byte_id in range(256):
        piece = f"<0x{byte_id:02X}>"
        if add_piece(piece, "byte_fallback", -100000 - byte_id):
            byte_added += 1

    # --------------------------------------------------
    # 6. Fill remainder with reserved
    # --------------------------------------------------
    reserve_id = 0
    while len(final_rows) < TARGET_VOCAB_SIZE:
        piece = f"<RESERVED_{reserve_id}>"
        if add_piece(piece, "reserved", -200000 - reserve_id):
            reserve_id += 1

    # --------------------------------------------------
    # 7. Truncate and assign IDs
    # --------------------------------------------------
    final_rows = final_rows[:TARGET_VOCAB_SIZE]
    for i, row in enumerate(final_rows):
        row["id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_rows)} pieces -> {OUTPUT}")
    print()
    print("Budget summary:")
    print(f"  unicode required : {unicode_added}")
    print(f"  baseline anchors : {baseline_added}")
    print(f"  phrase merges    : {phrase_added}")
    print(f"  word/subwords    : {word_added}")
    print(f"  byte fallbacks   : {byte_added}")
    print(f"  reserved         : {reserve_id}")
    print()

    kind_counts = {}
    for row in final_rows:
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1

    print("Kind breakdown:")
    for kind, count in sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{kind:20s} {count:4d}")

    print("\nTop 30 pieces:")
    for row in final_rows[:30]:
        print(f'{row["id"]:4d}  {row["kind"]:20s}  {repr(row["piece"])}')

    print("\nPhrases added:")
    for row in final_rows:
        if row["kind"] == "phrase_merge":
            print(f'  {repr(row["piece"])}')


if __name__ == "__main__":
    main()