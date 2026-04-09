import json
import os
import re
from itertools import product

RESULTS_INPUT = "analysis/grid_vocabs/results.jsonl"
BASELINE_INPUT = "analysis/baseline_vocab_clean.jsonl"
MERGE_INPUT = "ngram/merge_candidates_bigram_plus_trigram_v2.jsonl"
OUTPUT_DIR = "analysis/local_search_vocabs"

TARGET_VOCAB_SIZE = 1024
BYTE_FALLBACK_COUNT = 256

TOP_N = 12

# Local search neighborhoods
BASELINE_DELTAS = [-20, -10, 0, 10, 20]
PHRASE_DELTAS = [-8, -4, 0, 4, 8]
WORD_DELTAS = [-8, -4, 0, 4, 8]

BASELINE_SORT_OPTIONS = ["priority", "id", "priority_len"]
PHRASE_SORT_OPTIONS = ["freq", "score", "combined"]
WORD_SORT_OPTIONS = ["freq", "score", "combined"]

# Allow a few alternate piece pools
PIN_MODE_OPTIONS = ["normal", "reduced", "none"]
MIN_SUBWORD_SCORE_OPTIONS = [12, 14, 16]

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


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def word_or_subword_ok(piece: str, cls: str, score: int, min_subword_score: int) -> bool:
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
        if score < min_subword_score:
            return False
        if not has_letter(piece):
            return False
        return True

    return False


def baseline_sort_key(row, mode):
    priority = float(row.get("priority", 0))
    source_id = int(row.get("id", 10**9))
    piece = row.get("piece", "")
    if mode == "priority":
        return (-priority, source_id)
    if mode == "id":
        return (source_id,)
    if mode == "priority_len":
        return (-priority, len(piece), piece)
    raise ValueError(f"Unknown baseline sort mode: {mode}")


def phrase_combined_value(row, pins):
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
    if piece in pins:
        bonus += 500000
    if piece.islower():
        bonus += 1000
    return freq * 1000 + score + bonus


def word_combined_value(row, pins):
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
    if piece in pins:
        bonus += 500000
    return freq * 1000 + score + bonus


def phrase_sort_key(row, mode, pins):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")
    if mode == "freq":
        return (-freq, len(piece), piece)
    if mode == "score":
        return (-score, len(piece), piece)
    if mode == "combined":
        return (-phrase_combined_value(row, pins), len(piece), piece)
    raise ValueError(f"Unknown phrase sort mode: {mode}")


def word_sort_key(row, mode, pins):
    freq = float(row.get("freq", 0))
    score = float(row.get("score", 0))
    piece = row.get("text", "")
    if mode == "freq":
        return (-freq, len(piece), piece)
    if mode == "score":
        return (-score, len(piece), piece)
    if mode == "combined":
        return (-word_combined_value(row, pins), len(piece), piece)
    raise ValueError(f"Unknown word sort mode: {mode}")


def pin_set_for_mode(mode):
    if mode == "normal":
        return set(PIN_CUSTOM_PIECES)
    if mode == "reduced":
        return {
            "There are",
            "such as",
            "as well as",
            "able to",
            "public",
            "quality",
            "going to",
            "the same",
            "a lot",
            "a few",
        }
    if mode == "none":
        return set()
    raise ValueError(f"Unknown pin mode: {mode}")


def build_vocab(
    baseline_rows,
    merge_rows,
    baseline_keep,
    phrase_keep,
    word_keep,
    baseline_sort_mode,
    phrase_sort_mode,
    word_sort_mode,
    pin_mode,
    min_subword_score,
):
    pins = pin_set_for_mode(pin_mode)

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

    unicode_added = 0
    for i, piece in enumerate(UNICODE_MUST_HAVE):
        if add_piece(piece, "unicode_required", 2_000_000 - i):
            unicode_added += 1

    baseline_candidates = [r for r in baseline_rows if is_clean_piece(r.get("piece", ""))]
    baseline_candidates.sort(key=lambda r: baseline_sort_key(r, baseline_sort_mode))

    baseline_added = 0
    for row in baseline_candidates:
        piece = row["piece"]
        if is_byte_piece(piece):
            continue
        if is_single_alpha(piece):
            continue
        if add_piece(piece, "baseline_anchor", row.get("priority", 0), meta={"source_id": row.get("id")}):
            baseline_added += 1
            if baseline_added >= baseline_keep:
                break

    pin_map = {}
    for row in merge_rows:
        text = row.get("text", "")
        if text not in pin_map:
            pin_map[text] = row

    pinned_phrase_added = 0
    pinned_word_added = 0

    for piece in sorted(pins):
        row = pin_map.get(piece)
        if not row:
            continue

        cls = row.get("class", "")
        score = int(row.get("score", 0))

        if cls in ALLOWED_PHRASE_CLASSES and phrase_ok(piece):
            if add_piece(
                piece,
                "phrase_merge",
                phrase_combined_value(row, pins) + 1_000_000,
                meta={"class": cls, "freq": row.get("freq", 0), "score": row.get("score", 0), "pinned": True},
            ):
                pinned_phrase_added += 1

        elif cls in ALLOWED_WORD_CLASSES and word_or_subword_ok(piece, cls, score, min_subword_score):
            if add_piece(
                piece,
                "word_subword_merge",
                word_combined_value(row, pins) + 1_000_000,
                meta={"class": cls, "freq": row.get("freq", 0), "score": row.get("score", 0), "pinned": True},
            ):
                pinned_word_added += 1

    phrase_candidates = []
    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        if cls not in ALLOWED_PHRASE_CLASSES:
            continue
        if not phrase_ok(piece):
            continue
        phrase_candidates.append(row)

    phrase_candidates.sort(key=lambda r: phrase_sort_key(r, phrase_sort_mode, pins))

    phrase_added = 0
    for row in phrase_candidates:
        if add_piece(
            row["text"],
            "phrase_merge",
            phrase_combined_value(row, pins),
            meta={"class": row.get("class"), "freq": row.get("freq", 0), "score": row.get("score", 0), "pinned": False},
        ):
            phrase_added += 1
            if phrase_added + pinned_phrase_added >= phrase_keep:
                break

    word_candidates = []
    for row in merge_rows:
        piece = row.get("text", "")
        cls = row.get("class", "")
        score = int(row.get("score", 0))
        if cls not in ALLOWED_WORD_CLASSES:
            continue
        if not word_or_subword_ok(piece, cls, score, min_subword_score):
            continue
        word_candidates.append(row)

    word_candidates.sort(key=lambda r: word_sort_key(r, word_sort_mode, pins))

    word_added = 0
    for row in word_candidates:
        if add_piece(
            row["text"],
            "word_subword_merge",
            word_combined_value(row, pins),
            meta={"class": row.get("class"), "freq": row.get("freq", 0), "score": row.get("score", 0), "pinned": False},
        ):
            word_added += 1
            if word_added + pinned_word_added >= word_keep:
                break

    byte_added = 0
    for byte_id in range(BYTE_FALLBACK_COUNT):
        piece = f"<0x{byte_id:02X}>"
        if add_piece(piece, "byte_fallback", -100000 - byte_id):
            byte_added += 1

    reserve_id = 0
    while len(final_rows) < TARGET_VOCAB_SIZE:
        piece = f"<RESERVED_{reserve_id}>"
        if add_piece(piece, "reserved", -200000 - reserve_id):
            reserve_id += 1

    final_rows = final_rows[:TARGET_VOCAB_SIZE]
    for i, row in enumerate(final_rows):
        row["id"] = i

    meta = {
        "baseline_keep": baseline_keep,
        "phrase_keep": phrase_keep,
        "word_keep": word_keep,
        "baseline_sort_mode": baseline_sort_mode,
        "phrase_sort_mode": phrase_sort_mode,
        "word_sort_mode": word_sort_mode,
        "pin_mode": pin_mode,
        "min_subword_score": min_subword_score,
        "unicode_added": unicode_added,
        "baseline_added": baseline_added,
        "pinned_phrase_added": pinned_phrase_added,
        "pinned_word_added": pinned_word_added,
        "extra_phrase_added": phrase_added,
        "extra_word_added": word_added,
        "byte_added": byte_added,
    }
    return final_rows, meta


def clamp_positive(x, minimum=0):
    return x if x >= minimum else minimum


def top_rows(results, n):
    valid = [r for r in results if r.get("abs_saved") is not None]
    valid.sort(
        key=lambda r: (
            -r["abs_saved"],
            r["cust_total"],
            r["worse"],
            -r["improved"],
        )
    )
    return valid[:n]


def local_variants_for_row(row):
    variants = []
    seen = set()

    base_b = int(row["baseline_keep"])
    base_p = int(row["phrase_keep"])
    base_w = int(row["word_keep"])
    base_bs = row["baseline_sort_mode"]
    base_ps = row["phrase_sort_mode"]
    base_ws = row["word_sort_mode"]

    for db, dp, dw, bs, ps, ws, pin_mode, min_subword_score in product(
        BASELINE_DELTAS,
        PHRASE_DELTAS,
        WORD_DELTAS,
        BASELINE_SORT_OPTIONS,
        PHRASE_SORT_OPTIONS,
        WORD_SORT_OPTIONS,
        PIN_MODE_OPTIONS,
        MIN_SUBWORD_SCORE_OPTIONS,
    ):
        b = clamp_positive(base_b + db)
        p = clamp_positive(base_p + dp)
        w = clamp_positive(base_w + dw)

        # Skip nonsensical tiny configs
        if b < 600 or p < 8 or w < 8:
            continue
        if b > 760 or p > 80 or w > 80:
            continue

        # Crude non-byte budget sanity. Unicode + bytes + a little slack.
        if b + p + w + len(UNICODE_MUST_HAVE) > 768:
            continue

        key = (b, p, w, bs, ps, ws, pin_mode, min_subword_score)
        if key in seen:
            continue
        seen.add(key)

        variants.append({
            "baseline_keep": b,
            "phrase_keep": p,
            "word_keep": w,
            "baseline_sort_mode": bs,
            "phrase_sort_mode": ps,
            "word_sort_mode": ws,
            "pin_mode": pin_mode,
            "min_subword_score": min_subword_score,
            "seed_tag": row["tag"],
        })

    return variants


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_jsonl(RESULTS_INPUT)
    best = top_rows(results, TOP_N)

    baseline_rows = load_jsonl(BASELINE_INPUT)
    merge_rows = load_jsonl(MERGE_INPUT)

    manifest = []
    all_variants = []
    seen_global = set()

    for row in best:
        local = local_variants_for_row(row)
        for variant in local:
            key = (
                variant["baseline_keep"],
                variant["phrase_keep"],
                variant["word_keep"],
                variant["baseline_sort_mode"],
                variant["phrase_sort_mode"],
                variant["word_sort_mode"],
                variant["pin_mode"],
                variant["min_subword_score"],
            )
            if key in seen_global:
                continue
            seen_global.add(key)
            all_variants.append(variant)

    print(f"Building {len(all_variants)} local-search vocab variants from top {len(best)} seeds...")

    for i, variant in enumerate(all_variants, start=1):
        rows, meta = build_vocab(
            baseline_rows=baseline_rows,
            merge_rows=merge_rows,
            baseline_keep=variant["baseline_keep"],
            phrase_keep=variant["phrase_keep"],
            word_keep=variant["word_keep"],
            baseline_sort_mode=variant["baseline_sort_mode"],
            phrase_sort_mode=variant["phrase_sort_mode"],
            word_sort_mode=variant["word_sort_mode"],
            pin_mode=variant["pin_mode"],
            min_subword_score=variant["min_subword_score"],
        )

        tag = (
            f'seed-{variant["seed_tag"]}'
            f'__b{variant["baseline_keep"]}'
            f'_p{variant["phrase_keep"]}'
            f'_w{variant["word_keep"]}'
            f'_bs-{variant["baseline_sort_mode"]}'
            f'_ps-{variant["phrase_sort_mode"]}'
            f'_ws-{variant["word_sort_mode"]}'
            f'_pin-{variant["pin_mode"]}'
            f'_ms{variant["min_subword_score"]}'
        )

        out_path = os.path.join(OUTPUT_DIR, f"{tag}.jsonl")
        write_jsonl(out_path, rows)

        manifest_row = {
            "tag": tag,
            "path": out_path,
            **meta,
            "seed_tag": variant["seed_tag"],
        }
        manifest.append(manifest_row)

        if i % 25 == 0 or i == len(all_variants):
            print(f"[{i}/{len(all_variants)}] wrote {out_path}")

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")
    write_jsonl(manifest_path, manifest)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()