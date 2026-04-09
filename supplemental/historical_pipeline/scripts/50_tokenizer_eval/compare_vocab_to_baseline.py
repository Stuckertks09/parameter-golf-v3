#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sentencepiece as spm


# -----------------------------
# IO
# -----------------------------

def load_docs_jsonl(path: Path, limit: int) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(docs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) and "text" in obj else obj
            if isinstance(text, str) and text:
                docs.append(text)
    return docs


def load_custom_vocab_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


# -----------------------------
# SP helpers
# -----------------------------

def sp_id_to_piece_text(sp: spm.SentencePieceProcessor, tid: int) -> str:
    if sp.is_byte(tid):
        piece = sp.id_to_piece(tid)
        # SentencePiece byte pieces are already printable like <0x41> on many models.
        return piece
    piece = sp.id_to_piece(tid)
    # Convert ▁ prefix to literal leading space so it matches your custom dump convention.
    if piece.startswith("▁"):
        return " " + piece[1:]
    return piece


def build_sp_maps(sp: spm.SentencePieceProcessor) -> tuple[dict[int, str], dict[str, int]]:
    id_to_piece: dict[int, str] = {}
    piece_to_id: dict[str, int] = {}
    for tid in range(int(sp.vocab_size())):
        piece = sp_id_to_piece_text(sp, tid)
        id_to_piece[tid] = piece
        piece_to_id[piece] = tid
    return id_to_piece, piece_to_id


# -----------------------------
# Custom vocab helpers
# -----------------------------

def build_custom_maps(rows: list[dict]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_piece: dict[int, str] = {}
    piece_to_id: dict[str, int] = {}
    for row in rows:
        tid = int(row["id"])
        piece = row["piece"]
        id_to_piece[tid] = piece
        piece_to_id[piece] = tid
    return id_to_piece, piece_to_id


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def piece_bytes(piece: str) -> int:
    if is_byte_piece(piece):
        return 1
    return len(piece.encode("utf-8"))


def split_keep_spaces(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split(" ")
    if len(parts) == 1:
        return [text]
    out: list[str] = []
    first = parts[0]
    if first:
        out.append(first)
    else:
        for idx in range(1, len(parts)):
            if parts[idx]:
                out.append(" " + parts[idx])
                for rest in parts[idx + 1:]:
                    if rest:
                        out.append(" " + rest)
                return out
        return []
    for p in parts[1:]:
        if p:
            out.append(" " + p)
    return out


def build_candidates_by_first_char(piece_to_id: dict[str, int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in piece_to_id.keys():
        if not p or is_byte_piece(p):
            continue
        out.setdefault(p[0], []).append(p)
    for ch in out:
        out[ch].sort(key=len, reverse=True)
    return out


def build_byte_piece_map(piece_to_id: dict[str, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for b in range(256):
        p = f"<0x{b:02X}>"
        tid = piece_to_id.get(p)
        if tid is not None:
            out[b] = tid
    return out


def decompose_piece_greedy(
    piece: str,
    piece_to_id: dict[str, int],
    candidates_by_first_char: dict[str, list[str]],
    max_parts: int,
) -> Optional[list[int]]:
    if not piece:
        return None
    n = len(piece)
    i = 0
    out: list[int] = []
    while i < n and len(out) < max_parts:
        first = piece[i]
        candidates = candidates_by_first_char.get(first, [])
        matched = None
        for cand in candidates:
            if cand == piece:
                continue
            if piece.startswith(cand, i):
                matched = cand
                break
        if matched is None:
            return None
        out.append(piece_to_id[matched])
        i += len(matched)
    if i != n:
        return None
    return out if out else None


def decompose_piece_to_ids(
    piece: str,
    piece_to_id: dict[str, int],
    byte_piece_to_id: dict[int, int],
    candidates_by_first_char: dict[str, list[str]],
    max_parts: int = 16,
    allow_byte_fallback: bool = True,
) -> Optional[list[int]]:
    if not piece:
        return None

    split_parts = split_keep_spaces(piece)
    if len(split_parts) >= 2:
        ids: list[int] = []
        ok = True
        for p in split_parts:
            if p in piece_to_id and p != piece:
                ids.append(piece_to_id[p])
            else:
                sub = decompose_piece_greedy(p, piece_to_id, candidates_by_first_char, max_parts)
                if sub is None:
                    ok = False
                    break
                ids.extend(sub)
            if len(ids) > max_parts:
                ok = False
                break
        if ok and ids:
            return ids

    greedy_ids = decompose_piece_greedy(piece, piece_to_id, candidates_by_first_char, max_parts)
    if greedy_ids is not None and len(greedy_ids) <= max_parts:
        return greedy_ids

    if allow_byte_fallback:
        out: list[int] = []
        for b in piece.encode("utf-8"):
            tid = byte_piece_to_id.get(b)
            if tid is None:
                return None
            out.append(tid)
            if len(out) > max_parts:
                return None
        return out if out else None

    return None


# -----------------------------
# Tokenization
# -----------------------------

class CustomGreedyTokenizer:
    """
    Uses longest-match greedy over your custom vocab.
    This is good enough for inventory analysis / replacement pressure.
    If you want to mirror DP exactly later, swap tokenize().
    """
    def __init__(self, piece_to_id: dict[str, int]):
        self.piece_to_id = piece_to_id
        self.candidates_by_first_char = build_candidates_by_first_char(piece_to_id)

    def tokenize(self, text: str) -> list[int]:
        n = len(text)
        i = 0
        ids: list[int] = []

        while i < n:
            first = text[i]
            candidates = self.candidates_by_first_char.get(first, [])
            matched = None
            for cand in candidates:
                if text.startswith(cand, i):
                    matched = cand
                    break
            if matched is not None:
                ids.append(self.piece_to_id[matched])
                i += len(matched)
                continue

            # fallback to bytes
            b = text[i].encode("utf-8")
            for bb in b:
                piece = f"<0x{bb:02X}>"
                tid = self.piece_to_id.get(piece)
                if tid is None:
                    raise ValueError(f"Missing byte fallback piece {piece}")
                ids.append(tid)
            i += 1
        return ids


# -----------------------------
# Stats
# -----------------------------

@dataclass
class TokenStats:
    freq: int = 0
    doc_freq: int = 0
    left_diversity: int = 0
    right_diversity: int = 0
    avg_piece_bytes: float = 0.0
    compression_credit: float = 0.0
    token_count_contrib: int = 0


def finalize_stats(
    freq: Counter,
    doc_freq: Counter,
    left_neighbors: dict[int, set[int]],
    right_neighbors: dict[int, set[int]],
    id_to_piece: dict[int, str],
    compression_credit: Counter,
) -> dict[int, TokenStats]:
    out: dict[int, TokenStats] = {}
    for tid in id_to_piece:
        f = int(freq.get(tid, 0))
        piece = id_to_piece[tid]
        out[tid] = TokenStats(
            freq=f,
            doc_freq=int(doc_freq.get(tid, 0)),
            left_diversity=len(left_neighbors.get(tid, set())),
            right_diversity=len(right_neighbors.get(tid, set())),
            avg_piece_bytes=float(piece_bytes(piece)),
            compression_credit=float(compression_credit.get(tid, 0.0)),
            token_count_contrib=f,
        )
    return out


# -----------------------------
# Main analysis
# -----------------------------

def analyze(
    docs: list[str],
    sp_model_path: Path,
    custom_vocab_path: Path,
    out_dir: Path,
    max_decomp_parts: int = 16,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    sp_id_to_piece, sp_piece_to_id = build_sp_maps(sp)

    custom_rows = load_custom_vocab_rows(custom_vocab_path)
    custom_id_to_piece, custom_piece_to_id = build_custom_maps(custom_rows)
    custom_tok = CustomGreedyTokenizer(custom_piece_to_id)

    custom_candidates = build_candidates_by_first_char(custom_piece_to_id)
    custom_byte_map = build_byte_piece_map(custom_piece_to_id)

    sp_freq = Counter()
    sp_doc_freq = Counter()
    sp_left = defaultdict(set)
    sp_right = defaultdict(set)
    sp_credit = Counter()

    custom_freq = Counter()
    custom_doc_freq = Counter()
    custom_left = defaultdict(set)
    custom_right = defaultdict(set)
    custom_credit = Counter()

    doc_records = []
    total_sp_tokens = 0
    total_custom_tokens = 0

    for doc_idx, text in enumerate(docs):
        sp_ids = list(sp.encode(text, out_type=int))
        custom_ids = custom_tok.tokenize(text)

        total_sp_tokens += len(sp_ids)
        total_custom_tokens += len(custom_ids)

        # document-level credit: positive means custom saved tokens
        delta = len(sp_ids) - len(custom_ids)

        sp_seen = set(sp_ids)
        custom_seen = set(custom_ids)

        for tid in sp_seen:
            sp_doc_freq[tid] += 1
        for tid in custom_seen:
            custom_doc_freq[tid] += 1

        for i, tid in enumerate(sp_ids):
            sp_freq[tid] += 1
            if i > 0:
                sp_left[tid].add(sp_ids[i - 1])
            if i + 1 < len(sp_ids):
                sp_right[tid].add(sp_ids[i + 1])

        for i, tid in enumerate(custom_ids):
            custom_freq[tid] += 1
            if i > 0:
                custom_left[tid].add(custom_ids[i - 1])
            if i + 1 < len(custom_ids):
                custom_right[tid].add(custom_ids[i + 1])

        # rough token-level credit allocation
        if custom_ids:
            per_custom_credit = delta / len(custom_ids)
            for tid in custom_ids:
                custom_credit[tid] += per_custom_credit
        if sp_ids:
            per_sp_credit = (-delta) / len(sp_ids)
            for tid in sp_ids:
                sp_credit[tid] += per_sp_credit

        doc_records.append(
            {
                "doc_id": doc_idx,
                "chars": len(text),
                "sp_tokens": len(sp_ids),
                "custom_tokens": len(custom_ids),
                "delta_sp_minus_custom": delta,
            }
        )

    sp_stats = finalize_stats(sp_freq, sp_doc_freq, sp_left, sp_right, sp_id_to_piece, sp_credit)
    custom_stats = finalize_stats(custom_freq, custom_doc_freq, custom_left, custom_right, custom_id_to_piece, custom_credit)

    # custom decomposition info
    custom_decomp = {}
    for tid, piece in custom_id_to_piece.items():
        part_ids = decompose_piece_to_ids(
            piece,
            custom_piece_to_id,
            custom_byte_map,
            custom_candidates,
            max_parts=max_decomp_parts,
            allow_byte_fallback=True,
        )
        if part_ids is None:
            custom_decomp[tid] = {
                "num_parts": None,
                "non_byte_parts": None,
                "parts": None,
            }
        else:
            part_pieces = [custom_id_to_piece[p] for p in part_ids]
            custom_decomp[tid] = {
                "num_parts": len(part_ids),
                "non_byte_parts": sum(0 if is_byte_piece(p) else 1 for p in part_pieces),
                "parts": part_pieces,
            }

    # overlap + reporting
    overlap_pieces = set(sp_piece_to_id) & set(custom_piece_to_id)
    sp_only_pieces = set(sp_piece_to_id) - set(custom_piece_to_id)
    custom_only_pieces = set(custom_piece_to_id) - set(sp_piece_to_id)

    summary = {
        "num_docs": len(docs),
        "total_sp_tokens": total_sp_tokens,
        "total_custom_tokens": total_custom_tokens,
        "custom_vs_sp_ratio": (total_custom_tokens / total_sp_tokens) if total_sp_tokens else None,
        "token_delta_sp_minus_custom": total_sp_tokens - total_custom_tokens,
        "overlap_piece_count": len(overlap_pieces),
        "sp_only_piece_count": len(sp_only_pieces),
        "custom_only_piece_count": len(custom_only_pieces),
        "sp_vocab_size": len(sp_id_to_piece),
        "custom_vocab_size": len(custom_id_to_piece),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # unified token report
    token_report_path = out_dir / "token_report.csv"
    with token_report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source", "token_id", "piece", "in_sp", "in_custom",
            "freq", "doc_freq", "doc_coverage_ratio",
            "left_diversity", "right_diversity", "context_diversity",
            "piece_bytes", "compression_credit",
            "num_parts", "non_byte_parts", "decomp_parts_json",
            "verdict"
        ])

        # SP rows
        for tid, piece in sp_id_to_piece.items():
            st = sp_stats[tid]
            in_custom = piece in custom_piece_to_id
            verdict = "sp_overlap" if in_custom else "sp_only"
            w.writerow([
                "sp",
                tid,
                piece,
                1,
                1 if in_custom else 0,
                st.freq,
                st.doc_freq,
                (st.doc_freq / len(docs)) if docs else 0.0,
                st.left_diversity,
                st.right_diversity,
                st.left_diversity + st.right_diversity,
                st.avg_piece_bytes,
                st.compression_credit,
                "",
                "",
                "",
                verdict,
            ])

        # custom rows
        for tid, piece in custom_id_to_piece.items():
            st = custom_stats[tid]
            in_sp = piece in sp_piece_to_id
            de = custom_decomp[tid]
            verdict = classify_custom_token(
                piece=piece,
                freq=st.freq,
                doc_freq=st.doc_freq,
                num_docs=len(docs),
                compression_credit=st.compression_credit,
                num_parts=de["num_parts"],
                non_byte_parts=de["non_byte_parts"],
                context_diversity=st.left_diversity + st.right_diversity,
                in_sp=in_sp,
            )
            w.writerow([
                "custom",
                tid,
                piece,
                1 if in_sp else 0,
                1,
                st.freq,
                st.doc_freq,
                (st.doc_freq / len(docs)) if docs else 0.0,
                st.left_diversity,
                st.right_diversity,
                st.left_diversity + st.right_diversity,
                st.avg_piece_bytes,
                st.compression_credit,
                de["num_parts"],
                de["non_byte_parts"],
                json.dumps(de["parts"], ensure_ascii=False) if de["parts"] is not None else "",
                verdict,
            ])

    # SP-only useful pieces
    with (out_dir / "sp_only_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "token_id", "piece", "freq", "doc_freq", "doc_coverage_ratio",
            "context_diversity", "piece_bytes", "compression_credit", "priority_score"
        ])
        rows = []
        for piece in sp_only_pieces:
            tid = sp_piece_to_id[piece]
            st = sp_stats[tid]
            coverage = st.doc_freq / len(docs) if docs else 0.0
            ctx = st.left_diversity + st.right_diversity
            priority = (
                math.log1p(st.freq) * 1.5
                + coverage * 200.0
                + math.log1p(ctx) * 2.0
                + max(0.0, st.compression_credit) * 5.0
            )
            rows.append((priority, tid, piece, st, coverage, ctx))
        rows.sort(reverse=True)
        for priority, tid, piece, st, coverage, ctx in rows:
            w.writerow([tid, piece, st.freq, st.doc_freq, coverage, ctx, st.avg_piece_bytes, st.compression_credit, priority])

    # custom-only weak pieces
    with (out_dir / "custom_only_weak.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "token_id", "piece", "freq", "doc_freq", "doc_coverage_ratio",
            "context_diversity", "piece_bytes", "compression_credit",
            "num_parts", "non_byte_parts", "weakness_score"
        ])
        rows = []
        for piece in custom_only_pieces:
            tid = custom_piece_to_id[piece]
            st = custom_stats[tid]
            de = custom_decomp[tid]
            coverage = st.doc_freq / len(docs) if docs else 0.0
            ctx = st.left_diversity + st.right_diversity
            weakness = (
                (1.0 / max(1, st.freq)) * 5000.0
                + (1.0 - coverage) * 5.0
                + max(0.0, -st.compression_credit) * 5.0
                + (de["num_parts"] or 0) * 0.25
            )
            rows.append((weakness, tid, piece, st, coverage, ctx, de))
        rows.sort(reverse=True)
        for weakness, tid, piece, st, coverage, ctx, de in rows:
            w.writerow([
                tid, piece, st.freq, st.doc_freq, coverage, ctx, st.avg_piece_bytes,
                st.compression_credit, de["num_parts"], de["non_byte_parts"], weakness
            ])

    # replacement map: SP-only strong candidates vs custom-only weak candidates
    sp_candidates = read_csv_rows(out_dir / "sp_only_candidates.csv")
    custom_weak = read_csv_rows(out_dir / "custom_only_weak.csv")
    with (out_dir / "replacement_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "remove_custom_token_id", "remove_custom_piece",
            "add_sp_token_id", "add_sp_piece",
            "remove_freq", "add_freq",
            "remove_doc_freq", "add_doc_freq",
            "remove_compression_credit", "add_compression_credit"
        ])
        n = min(200, len(sp_candidates), len(custom_weak))
        for i in range(n):
            rm = custom_weak[i]
            add = sp_candidates[i]
            w.writerow([
                rm["token_id"], rm["piece"],
                add["token_id"], add["piece"],
                rm["freq"], add["freq"],
                rm["doc_freq"], add["doc_freq"],
                rm["compression_credit"], add["compression_credit"],
            ])

    # doc-level file
    with (out_dir / "doc_level.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(doc_records[0].keys()) if doc_records else [])
        if doc_records:
            w.writeheader()
            w.writerows(doc_records)


def classify_custom_token(
    *,
    piece: str,
    freq: int,
    doc_freq: int,
    num_docs: int,
    compression_credit: float,
    num_parts: Optional[int],
    non_byte_parts: Optional[int],
    context_diversity: int,
    in_sp: bool,
) -> str:
    if in_sp:
        return "overlap_keep"
    coverage = (doc_freq / num_docs) if num_docs else 0.0

    if freq < 20 and coverage < 0.002:
        return "weak_remove"

    if compression_credit < 0 and coverage < 0.01:
        return "weak_remove"

    if (num_parts or 0) >= 3 and (non_byte_parts or 0) >= 2 and freq < 200:
        return "possibly_overmerged"

    if compression_credit > 0 and coverage > 0.01 and context_diversity > 20:
        return "custom_keep"

    if piece.startswith(" ") and freq > 100 and coverage > 0.005:
        return "custom_keep"

    return "review"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True, help="docs_selected.jsonl or equivalent")
    ap.add_argument("--sp-model", type=Path, required=True, help="SentencePiece .model")
    ap.add_argument("--custom-vocab", type=Path, required=True, help="custom vocab jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-docs", type=int, default=50000)
    args = ap.parse_args()

    docs = load_docs_jsonl(args.docs, args.num_docs)
    analyze(
        docs=docs,
        sp_model_path=args.sp_model,
        custom_vocab_path=args.custom_vocab,
        out_dir=args.out_dir,
    )
    print(f"Wrote report to: {args.out_dir}")


if __name__ == "__main__":
    main()