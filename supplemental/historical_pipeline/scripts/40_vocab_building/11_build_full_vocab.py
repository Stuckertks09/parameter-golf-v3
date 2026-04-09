import json
import os
import re
import string

DRAFT_INPUT = "analysis/custom_vocab.jsonl"
MERGE_INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"
OUTPUT = "analysis/custom_vocab_full.jsonl"

TARGET_VOCAB_SIZE = 1024

ALLOWED_MERGE_CLASSES = {
    "keep_allowlist",
    "keep_short_phrase",
    "keep_full_word",
    "keep_contraction",
    "curated_subword_auto_v2",
}

MIN_SUBWORD_SCORE = 14
MAX_PHRASE_TOKENS = 3

# Hard bans from prior bad runs
BAD_FRAGMENTS = {
    "ed to", "ing to", "ing the", "ed in", "ed by", "ed the",
    "es and", "es of", "ing in", "the st", "ation of", "ed with",
    "es to", "ing for", "the re", "ion of", "to re", "es in",
    "ed and", "er to", "ed for", "the pro", "ed on",
    "s of the", "ed by the", "ed in the", "umber of", "ording to",
    "a litt",
}

# Common Unicode punctuation that was likely exploding to byte fallback
UNICODE_COVERAGE = [
    "’", "‘", "“", "”",
    "–", "—", "…",
    "•", "·",
    "\u00A0",  # non-breaking space
]

# Useful punctuation chunks / contractions
COMMON_PUNCT_CHUNKS = [
    "’s", "n't", "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
    ".”", ",”", ".'", ",'", ". ", ", ", ": ", "; ", "? ", "! ",
    "— ", "– ",
]

BYTE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")

LOW_VALUE_PREFIXES = (
    "ed ", "ing ", "es ", "er ", "ly ", "ion ", "ation ", "s ",
)

LOW_VALUE_SUFFIXES = (
    " ed", " ing", " es", " er", " ly", " ion", " tion", " s",
)

MIDWORD_SPACE_PATTERNS = (
    "ation ",
    "ition ",
    "ment ",
    "ness ",
    "able ",
    "ible ",
    "umber ",
    "ording ",
)

LETTER_RE = re.compile(r"[A-Za-z]")
FULL_WORD_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+){0,2}$")
CONTRACTION_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+|’[A-Za-z]+)$")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_clean_piece(piece: str) -> bool:
    if not piece:
        return False
    if "\x00" in piece:
        return False
    return True


def is_byte_piece(piece: str) -> bool:
    return BYTE_RE.match(piece) is not None


def token_count(piece: str) -> int:
    return len([p for p in piece.split(" ") if p])


def has_letter(piece: str) -> bool:
    return LETTER_RE.search(piece) is not None


def has_midword_boundary_smell(piece: str) -> bool:
    # obvious partial-word junk from previous runs
    if piece.startswith(LOW_VALUE_PREFIXES):
        return True
    if piece.endswith(LOW_VALUE_SUFFIXES):
        return True

    for pat in MIDWORD_SPACE_PATTERNS:
        if pat in piece:
            return True

    # weird “half word + space + word” patterns
    # examples: "umber of", "ording to"
    parts = piece.split(" ")
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]

        if first and first.isalpha() and len(first) <= 5 and first.endswith(("ing", "ed", "es", "er", "ion")):
            return True
        if first in {"s", "ed", "ing", "es", "er", "ion", "ation"}:
            return True
        if len(first) <= 5 and first in {"umber", "ording", "ation", "ition"}:
            return True
        if len(last) <= 4 and last in {"re", "st", "pro", "tion"}:
            return True

    return False


def is_reasonable_phrase(piece: str) -> bool:
    # only allow clean full-word phrases up to 3 words
    if not FULL_WORD_RE.match(piece):
        return False
    if token_count(piece) > MAX_PHRASE_TOKENS:
        return False
    if has_midword_boundary_smell(piece):
        return False
    return True


def is_reasonable_contraction(piece: str) -> bool:
    return CONTRACTION_RE.match(piece) is not None


def is_reasonable_subword(piece: str, score: int) -> bool:
    if score < MIN_SUBWORD_SCORE:
        return False
    if " " in piece:
        return False
    if len(piece) < 3:
        return False
    if not has_letter(piece):
        return False
    return True


def looks_reasonable(piece: str, cls: str, score: int) -> bool:
    if piece in BAD_FRAGMENTS:
        return False
    if len(piece) < 2:
        return False
    if piece.strip() == "":
        return False

    # Avoid bringing byte fallback strings in from inputs
    if is_byte_piece(piece):
        return False

    if cls == "keep_short_phrase":
        return is_reasonable_phrase(piece)

    if cls == "keep_full_word":
        # full words or very clean phrases only
        return FULL_WORD_RE.match(piece) is not None and token_count(piece) <= MAX_PHRASE_TOKENS

    if cls == "keep_contraction":
        return is_reasonable_contraction(piece)

    if cls == "curated_subword_auto_v2":
        return is_reasonable_subword(piece, score)

    if cls == "keep_allowlist":
        # allowlist still gets filtered for obvious junk
        if has_midword_boundary_smell(piece):
            return False
        return True

    return False


def classify_bonus(piece: str, cls: str) -> float:
    """
    Small ranking nudges.
    We want:
    - clean full-word phrases > weird fragments
    - contractions and useful word pieces > marginal junk
    """
    bonus = 0.0

    if piece in UNICODE_COVERAGE:
        bonus += 1000000

    if piece in COMMON_PUNCT_CHUNKS:
        bonus += 500000

    if cls == "keep_short_phrase" and is_reasonable_phrase(piece):
        bonus += 20000

    if cls == "keep_full_word":
        if " " not in piece:
            bonus += 15000
        else:
            bonus += 10000

    if cls == "keep_contraction":
        bonus += 12000

    if cls == "curated_subword_auto_v2":
        bonus += 2000

    # prefer pieces that contain letters over pure punctuation noise
    if has_letter(piece):
        bonus += 500

    return bonus


def main():
    os.makedirs("analysis", exist_ok=True)

    pool = {}

    def add_piece(piece: str, kind: str, priority: float):
        if not is_clean_piece(piece):
            return
        if piece not in pool:
            pool[piece] = {
                "piece": piece,
                "kind": kind,
                "priority": float(priority),
            }

    # 1. Start with draft inventory
    draft_rows = load_jsonl(DRAFT_INPUT)
    for row in draft_rows:
        piece = row["piece"]
        kind = row.get("kind", "draft")
        priority = float(row.get("priority", 0))

        # keep draft pieces, but don't allow obvious junk or byte pieces to sneak in
        if piece in BAD_FRAGMENTS:
            continue
        if is_byte_piece(piece):
            continue
        if has_midword_boundary_smell(piece) and " " in piece:
            continue

        add_piece(piece, kind, priority)

    # 2. Add curated mined candidates with stronger filtering
    merge_rows = load_jsonl(MERGE_INPUT)
    candidates = []

    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        freq = int(row.get("freq", 0))
        score = int(row.get("score", 0))

        if cls not in ALLOWED_MERGE_CLASSES:
            continue
        if not looks_reasonable(piece, cls, score):
            continue

        priority = freq + classify_bonus(piece, cls)
        candidates.append((priority, piece, cls, freq, score))

    candidates.sort(reverse=True)

    for priority, piece, cls, freq, score in candidates:
        add_piece(piece, cls, priority)

    # 3. Add strong universal ASCII coverage
    required = []
    required.extend(list(string.ascii_lowercase))
    required.extend(list(string.ascii_uppercase))
    required.extend(list(string.digits))
    required.extend([
        " ", "\n", "\t",
        ".", ",", ":", ";", "?", "!",
        "-", "_",
        "(", ")", "[", "]", "{", "}",
        "'", '"', "/", "\\", "@", "#", "$", "%", "&", "*", "+", "=",
        "<", ">", "|", "~", "^", "`",
    ])

    # 4. Add explicit Unicode punctuation coverage
    required.extend(UNICODE_COVERAGE)

    # 5. Add useful punctuation chunks / contractions
    required.extend(COMMON_PUNCT_CHUNKS)

    for i, piece in enumerate(required):
        add_piece(piece, "required_char", 900000 - i)

    # 6. Fill with byte fallback pieces
    byte_id = 0
    while len(pool) < TARGET_VOCAB_SIZE and byte_id < 256:
        piece = f"<0x{byte_id:02X}>"
        add_piece(piece, "byte_fallback", -100000 - byte_id)
        byte_id += 1

    # 7. If still not full, add reserved pieces
    reserve_id = 0
    while len(pool) < TARGET_VOCAB_SIZE:
        piece = f"<RESERVED_{reserve_id}>"
        add_piece(piece, "reserved", -200000 - reserve_id)
        reserve_id += 1

    # 8. Sort and truncate
    rows = list(pool.values())
    rows.sort(key=lambda x: (-x["priority"], len(x["piece"]), x["piece"]))
    final_rows = rows[:TARGET_VOCAB_SIZE]

    for i, row in enumerate(final_rows):
        row["id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_rows)} pieces -> {OUTPUT}")

    kind_counts = {}
    for row in final_rows:
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1

    print("\nKind breakdown:")
    for kind, count in sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{kind:24s} {count:4d}")

    print("\nTop 80 pieces:")
    for row in final_rows[:80]:
        print(f'{row["id"]:4d}  {row["kind"]:24s}  {repr(row["piece"])}')


if __name__ == "__main__":
    main()