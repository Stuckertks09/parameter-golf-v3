from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")
STRUCTURAL_LABELS = {
    "Chapter", "Section", "Volume", "Act", "Scene", "Article", "Part",
    "Book", "Page", "Terms", "Conditions", "Summary", "Introduction",
    "Appendix", "Table", "Figure", "Transcript", "Episode", "NEWS",
    "THE", "Text",
}
STRUCTURE_WORDS = {
    "Page", "Text", "Chapter", "Section", "Volume", "Article",
    "Act", "Scene", "Terms", "Conditions", "Transcript", "Summary",
    "Introduction", "Appendix", "Book", "Part", "Figure", "Table",
}
PROSE_STARTERS = {
    "you have", "you are", "you can", "have to", "want to",
    "that you", "i think", "if you", "we have", "it is",
    "this is", "there is", "there are",
}
COMMON_GLUE = {
    "the", "and", "is", "to", "of", "in", "on", "for", "a", "an", "or",
    "as", "at", "by", "be", "with", "from", "that", "this", "these",
    "those", "it", "its", "he", "she", "they", "we", "you",
}
COMMON_SUFFIXES = {"ing", "ed", "er", "ly", "s", "es", "ion", "tion", "ment"}


def is_byte_piece(piece: str) -> bool:
    return bool(BYTE_PIECE_RE.match(piece))


def load_vocab_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


def write_vocab_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_worst_doc_rows(path: Path, top_k: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if top_k is not None and i >= top_k:
                break
            row["doc_idx"] = int(row["doc_idx"])
            row["delta_tokens"] = int(row["delta_tokens"])
            row["bytes"] = int(row["bytes"])
            row["sp_tokens"] = int(row["sp_tokens"])
            row["custom_tokens"] = int(row["custom_tokens"])
            rows.append(row)
    return rows


def load_docs_by_ids(docs_jsonl: Path, wanted_ids: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in wanted_ids:
                obj = json.loads(line)
                out[idx] = obj["text"]
                if len(out) == len(wanted_ids):
                    break
    missing = wanted_ids - set(out.keys())
    if missing:
        raise ValueError(f"Missing doc ids in docs_selected.jsonl: {sorted(list(missing))[:10]}")
    return out


def build_piece_set(vocab_rows: list[dict]) -> set[str]:
    return {row["piece"] for row in vocab_rows}


def normalize_text(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def tokenize_line(line: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[,:;.!?()/%&]+", line)


def is_headerish(line: str) -> bool:
    s = line.strip()
    if len(s) < 4:
        return False
    if s.isupper():
        return True
    upper = sum(ch.isupper() for ch in s if ch.isalpha())
    alpha = sum(ch.isalpha() for ch in s)
    return alpha >= 4 and upper / max(alpha, 1) >= 0.55


def has_enough_signal(piece: str) -> bool:
    stripped = piece.strip()
    alpha = sum(ch.isalpha() for ch in stripped)
    digit = sum(ch.isdigit() for ch in stripped)
    return (alpha + digit) >= 4 and alpha >= 2


def is_newline_structural(piece: str) -> bool:
    if not piece.startswith("\n"):
        return False
    p = piece[1:].strip()
    toks = tokenize_line(p)
    if not toks:
        return False
    first = toks[0]
    if first in STRUCTURAL_LABELS:
        return True
    if first.isupper() and len(first) >= 3:
        return True
    return False


def is_sentence_starter(piece: str) -> bool:
    p = piece.strip()
    toks = tokenize_line(p)
    if not toks:
        return False
    joined = " ".join(toks[:2]).lower()
    bad_starts = {
        "this is", "there is", "there are", "if you", "i have", "we have",
        "it is", "when the", "what is", "how to", "in this",
    }
    return joined in bad_starts or (
        p.startswith("\n") and len(toks) >= 1 and toks[0] in {"What", "When", "There", "This", "How", "Why"}
    )


def is_probable_proper_noun(piece: str) -> bool:
    p = piece.strip()
    toks = tokenize_line(p)
    if not toks:
        return False
    if len(toks) == 1 and toks[0][:1].isupper() and toks[0] not in STRUCTURAL_LABELS:
        return True
    return False


def has_structure_word(piece: str) -> bool:
    toks = tokenize_line(piece.strip())
    return any(tok in STRUCTURE_WORDS for tok in toks)


def is_prose_phrase(piece: str) -> bool:
    p = " ".join(tokenize_line(piece.strip())[:2]).lower()
    return p in PROSE_STARTERS


def is_low_information(piece: str) -> bool:
    p = piece.strip()

    if len(p) < 4:
        return True
    if p.lower() in COMMON_GLUE:
        return True
    if p.lower() in COMMON_SUFFIXES:
        return True
    if re.fullmatch(r"[A-Za-z]\s[A-Za-z]", p):
        return True
    if re.fullmatch(r"[a-z]{1,3}", p):
        return True
    if re.fullmatch(r"[a-z]{1,2}\s[a-z]{1,2}", p):
        return True
    if p.endswith(" ") and p.strip().lower() in COMMON_GLUE:
        return True
    if p.startswith(" ") and len(p.strip()) < 4:
        return True
    if "  " in p:
        return True
    if not has_enough_signal(p):
        return True
    if is_sentence_starter(piece):
        return True
    if is_probable_proper_noun(piece):
        return True
    if len(tokenize_line(p)) >= 4 and not is_newline_structural(piece) and not is_headerish(p):
        return True

    return False


def is_good_candidate(piece: str, existing: set[str], min_len: int, max_len: int) -> bool:
    if piece in existing:
        return False
    if not piece or is_byte_piece(piece):
        return False
    if len(piece) < min_len or len(piece) > max_len:
        return False
    if piece.strip() == "":
        return False
    if is_low_information(piece):
        return False
    return True


def add_candidate(counts: Counter, cand: str, weight: int, existing: set[str], min_len: int, max_len: int) -> None:
    cand = cand[:max_len]
    if is_good_candidate(cand, existing, min_len, max_len):
        counts[cand] += weight


def extract_line_candidates(line: str, existing: set[str], min_len: int, max_len: int) -> Counter:
    counts = Counter()
    line = line.strip()
    if not line:
        return counts

    tokens = tokenize_line(line)
    if not tokens:
        return counts

    # prefixes
    for n in range(1, min(5, len(tokens)) + 1):
        cand = " ".join(tokens[:n]).strip()
        add_candidate(counts, cand, 5, existing, min_len, max_len)

    # internal spans
    for i in range(len(tokens)):
        for n in range(2, 5):
            if i + n > len(tokens):
                break
            cand = " ".join(tokens[i:i + n]).strip()
            add_candidate(counts, cand, 2, existing, min_len, max_len)

    # punctuation-aware connectors
    joined = " ".join(tokens)
    for m in re.finditer(r"(?:[,:;]\s+[A-Za-z][A-Za-z0-9'’\-]+(?:\s+[A-Za-z0-9'’\-]+){0,2})", joined):
        add_candidate(counts, m.group(0), 4, existing, min_len, max_len)

    return counts


def generate_candidates_from_text(text: str, existing: set[str], min_len: int, max_len: int) -> Counter:
    counts = Counter()
    text = normalize_text(text)
    lines = text.split("\n")

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue

        counts.update(extract_line_candidates(line, existing, min_len, max_len))

        stripped = line.strip()
        line_tokens = tokenize_line(stripped)
        if not line_tokens:
            continue

        # only structural newline spans
        if is_headerish(stripped) or line_tokens[0] in STRUCTURAL_LABELS or line_tokens[0].isupper():
            for n in range(1, min(4, len(line_tokens)) + 1):
                cand = "\n" + " ".join(line_tokens[:n])
                add_candidate(counts, cand, 10, existing, min_len, max_len)

        # reusable document-structure phrases
        for i in range(len(line_tokens)):
            for n in range(2, 4):
                if i + n > len(line_tokens):
                    break
                cand = " ".join(line_tokens[i:i + n]).strip()
                if has_structure_word(cand):
                    add_candidate(counts, cand, 8, existing, min_len, max_len)

        # known useful general phrases
        for n in range(2, min(4, len(line_tokens)) + 1):
            cand = " ".join(line_tokens[:n])
            if cand in {"is the", "with a", "have been", "into the", "through the", "of the", "and the", ", which", ". This"}:
                add_candidate(counts, cand, 5, existing, min_len, max_len)

    return counts


@dataclass
class CandidateScore:
    piece: str
    doc_hits: int
    raw_count: int
    est_savings: int
    weighted_score: float


def candidate_shape_features(piece: str) -> dict[str, bool]:
    stripped = piece.strip()
    return {
        "has_newline": "\n" in piece,
        "starts_upper": stripped[:1].isupper(),
        "has_multiword": len(stripped.split()) >= 2,
        "has_digit": any(ch.isdigit() for ch in stripped),
        "has_punct": any(ch in ":;,.!?()[]{}'\"/-" for ch in piece),
        "looks_header": is_headerish(stripped),
        "newline_structural": is_newline_structural(piece),
        "has_structure_word": has_structure_word(piece),
    }


def score_candidates(candidate_counts_by_doc: dict[int, Counter], min_doc_hits: int) -> list[CandidateScore]:
    agg_counts = Counter()
    doc_hits = Counter()

    for _, counts in candidate_counts_by_doc.items():
        for piece, c in counts.items():
            agg_counts[piece] += c
            doc_hits[piece] += 1

    scored = []
    for piece, raw_count in agg_counts.items():
        hits = doc_hits[piece]
        if hits < min_doc_hits:
            continue

        feats = candidate_shape_features(piece)
        est_savings = max(1, int(raw_count * max(len(piece) - 1, 1) / 10))

        score = 0.0
        score += est_savings
        score += 2.5 * hits
        score += 0.08 * len(piece)

        if feats["newline_structural"]:
            score += 14.0
        elif feats["has_newline"]:
            score -= 10.0

        if feats["looks_header"]:
            score += 9.0
        if feats["has_structure_word"]:
            score += 8.0
        if feats["starts_upper"] and not feats["newline_structural"]:
            score += 1.0
        if feats["has_multiword"]:
            score += 2.5
        if feats["has_digit"]:
            score += 1.5
        if feats["has_punct"]:
            score += 2.0

        stripped = piece.strip()
        if re.fullmatch(r"[a-z]{4,8}", stripped):
            score -= 6.0
        if stripped.lower() in COMMON_GLUE:
            score -= 12.0
        if is_sentence_starter(piece):
            score -= 22.0
        if is_probable_proper_noun(piece):
            score -= 20.0
        if is_prose_phrase(piece):
            score -= 18.0

        scored.append(
            CandidateScore(
                piece=piece,
                doc_hits=hits,
                raw_count=raw_count,
                est_savings=est_savings,
                weighted_score=score,
            )
        )

    scored.sort(key=lambda x: (x.weighted_score, x.est_savings, x.doc_hits, len(x.piece)), reverse=True)
    return scored


def choose_remove_candidates(vocab_rows: list[dict], num_remove: int) -> list[dict]:
    protected = {
        "ing", "ion", "tion", "ed", "er", "re", "th", "in",
        " the", " of", " and", ", and", " to", " is", "The", "This",
        " of the", " and the", "is the", "with a", "have been", "into the", "through the",
    }

    removable = []
    for row in vocab_rows:
        piece = row["piece"]
        stripped = piece.strip()

        if is_byte_piece(piece):
            continue
        if piece in protected:
            continue
        if "\n" in piece:
            continue
        if len(piece) <= 2:
            continue
        if re.fullmatch(r"[A-Za-z]{1,4}", stripped):
            continue

        score = len(piece) * 1.1
        score += 3.0 if " " in piece else 0.0
        score += 4.0 if len(stripped.split()) >= 2 else 0.0
        score += 1.0 if stripped[:1].islower() else 0.0
        score += 2.0 if len(piece) >= 8 else 0.0

        removable.append((score, row))

    removable.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in removable[:num_remove]]


def apply_swaps(vocab_rows: list[dict], add_scores: list[CandidateScore], remove_rows: list[dict], num_swaps: int):
    rows = [dict(r) for r in vocab_rows]
    remove_rows = remove_rows[:num_swaps]
    add_scores = add_scores[:num_swaps]

    swap_log = []
    for add, rem in zip(add_scores, remove_rows, strict=True):
        rid = int(rem["id"])
        old_piece = rows[rid]["piece"]
        rows[rid]["piece"] = add.piece
        rows[rid]["source"] = "worst_docs_fix_structured_bias"
        rows[rid]["note"] = {
            "replaced": old_piece,
            "candidate_score": add.weighted_score,
            "doc_hits": add.doc_hits,
            "raw_count": add.raw_count,
            "est_savings": add.est_savings,
        }
        swap_log.append(
            {
                "id": rid,
                "old_piece": old_piece,
                "new_piece": add.piece,
                "doc_hits": add.doc_hits,
                "raw_count": add.raw_count,
                "est_savings": add.est_savings,
                "weighted_score": add.weighted_score,
            }
        )
    return rows, swap_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst-docs-csv", type=Path, required=True)
    ap.add_argument("--docs-jsonl", type=Path, default=Path("data/docs_selected.jsonl"))
    ap.add_argument("--vocab-jsonl", type=Path, default=Path("vocab/vocab_best.jsonl"))
    ap.add_argument("--top-k-docs", type=int, default=200)
    ap.add_argument("--min-piece-len", type=int, default=4)
    ap.add_argument("--max-piece-len", type=int, default=24)
    ap.add_argument("--min-doc-hits", type=int, default=3)
    ap.add_argument("--num-swaps", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = ap.parse_args()

    vocab_rows = load_vocab_rows(args.vocab_jsonl)
    existing = {row["piece"] for row in vocab_rows}

    worst_rows = load_worst_doc_rows(args.worst_docs_csv, top_k=args.top_k_docs)
    worst_doc_ids = [r["doc_idx"] for r in worst_rows]
    docs = load_docs_by_ids(args.docs_jsonl, set(worst_doc_ids))

    candidate_counts_by_doc: dict[int, Counter] = {}
    for row in worst_rows:
        doc_id = row["doc_idx"]
        text = docs[doc_id]
        counts = generate_candidates_from_text(
            text=text,
            existing=existing,
            min_len=args.min_piece_len,
            max_len=args.max_piece_len,
        )

        weight = max(1, int(math.sqrt(max(row["delta_tokens"], 1))))
        weighted = Counter()
        for piece, c in counts.items():
            weighted[piece] = c * weight
        candidate_counts_by_doc[doc_id] = weighted

    scored = score_candidates(candidate_counts_by_doc, min_doc_hits=args.min_doc_hits)
    remove_rows = choose_remove_candidates(vocab_rows, num_remove=args.num_swaps * 3)
    new_rows, swap_log = apply_swaps(vocab_rows, scored, remove_rows, num_swaps=args.num_swaps)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    add_csv = out_dir / f"worst_docs_add_candidates_structured_bias_{ts}.csv"
    remove_csv = out_dir / f"worst_docs_remove_candidates_structured_bias_{ts}.csv"
    swaps_csv = out_dir / f"worst_docs_vocab_swaps_structured_bias_{ts}.csv"
    report_json = out_dir / f"worst_docs_vocab_report_structured_bias_{ts}.json"
    vocab_out = out_dir / f"vocab_best_worstdocs_fix_structured_bias_{ts}.jsonl"

    with add_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["piece", "doc_hits", "raw_count", "est_savings", "weighted_score"],
        )
        writer.writeheader()
        for row in scored[: max(args.num_swaps * 10, 100)]:
            writer.writerow(
                {
                    "piece": row.piece,
                    "doc_hits": row.doc_hits,
                    "raw_count": row.raw_count,
                    "est_savings": row.est_savings,
                    "weighted_score": f"{row.weighted_score:.4f}",
                }
            )

    with remove_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "piece"])
        writer.writeheader()
        for row in remove_rows[: max(args.num_swaps * 3, 60)]:
            writer.writerow({"id": int(row["id"]), "piece": row["piece"]})

    with swaps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "old_piece", "new_piece", "doc_hits", "raw_count", "est_savings", "weighted_score"],
        )
        writer.writeheader()
        for row in swap_log:
            out = dict(row)
            out["weighted_score"] = f"{out['weighted_score']:.4f}"
            writer.writerow(out)

    write_vocab_rows(vocab_out, new_rows)

    report = {
        "input_vocab": str(args.vocab_jsonl),
        "input_worst_docs_csv": str(args.worst_docs_csv),
        "top_k_docs": args.top_k_docs,
        "num_swaps": args.num_swaps,
        "min_piece_len": args.min_piece_len,
        "max_piece_len": args.max_piece_len,
        "min_doc_hits": args.min_doc_hits,
        "num_candidates_scored": len(scored),
        "outputs": {
            "add_candidates_csv": str(add_csv),
            "remove_candidates_csv": str(remove_csv),
            "swaps_csv": str(swaps_csv),
            "proposed_vocab_jsonl": str(vocab_out),
        },
        "swaps_preview": swap_log[:10],
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Scored candidates: {len(scored)}")
    print(f"Saved add candidates -> {add_csv}")
    print(f"Saved remove candidates -> {remove_csv}")
    print(f"Saved swaps -> {swaps_csv}")
    print(f"Saved proposed vocab -> {vocab_out}")
    print(f"Saved report -> {report_json}")


if __name__ == "__main__":
    main()