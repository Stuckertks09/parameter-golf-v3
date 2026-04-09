#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import difflib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sentencepiece as spm


# ----------------------------
# JSON / text loading
# ----------------------------

TEXT_KEYS = [
    "text",
    "content",
    "raw_text",
    "raw_content",
    "document",
    "doc",
    "body",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_text(row: dict) -> str:
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    raise ValueError(f"Could not find text field in row keys={list(row.keys())[:20]}")


# ----------------------------
# Custom vocab loading
# ----------------------------

def load_vocab_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def parse_byte_piece(piece: str) -> Optional[int]:
    if not is_byte_piece(piece):
        return None
    try:
        return int(piece[3:-1], 16)
    except Exception:
        return None


@dataclass
class TokenMatch:
    piece: str
    tid: int
    end: int


class TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.terminal: list[tuple[str, int]] = []  # (piece, tid)


class CustomDPTokenizer:
    """
    Exact min-token DP tokenizer over vocab pieces.
    Falls back to byte pieces when no string piece matches.
    Suitable for inspection/debugging, not bulk export.
    """

    def __init__(self, vocab_rows: list[dict]) -> None:
        self.rows = vocab_rows
        self.id_to_piece = {int(r["id"]): r["piece"] for r in vocab_rows}
        self.piece_to_id = {r["piece"]: int(r["id"]) for r in vocab_rows}

        self.root = TrieNode()
        self.byte_tid_by_val: dict[int, int] = {}

        for row in vocab_rows:
            tid = int(row["id"])
            piece = row["piece"]

            b = parse_byte_piece(piece)
            if b is not None:
                self.byte_tid_by_val[b] = tid
                continue

            node = self.root
            for ch in piece:
                node = node.children.setdefault(ch, TrieNode())
            node.terminal.append((piece, tid))

    def _all_matches(self, text: str, start: int) -> list[TokenMatch]:
        node = self.root
        out: list[TokenMatch] = []

        j = start
        while j < len(text):
            ch = text[j]
            node = node.children.get(ch)
            if node is None:
                break
            j += 1
            if node.terminal:
                for piece, tid in node.terminal:
                    out.append(TokenMatch(piece=piece, tid=tid, end=j))
        return out

    def encode(self, text: str) -> tuple[list[int], list[str]]:
        n = len(text)
        INF = 10**15

        dp = [INF] * (n + 1)
        nxt: list[Optional[tuple[int, int, str]]] = [None] * (n + 1)
        dp[n] = 0

        # backward DP
        for i in range(n - 1, -1, -1):
            matches = self._all_matches(text, i)

            # prefer fewer tokens; on ties, prefer longer piece
            best_cost = INF
            best_step: Optional[tuple[int, int, str]] = None

            for m in matches:
                cost = 1 + dp[m.end]
                span = m.end - i
                if cost < best_cost:
                    best_cost = cost
                    best_step = (m.end, m.tid, m.piece)
                elif cost == best_cost and best_step is not None:
                    best_span = best_step[0] - i
                    if span > best_span:
                        best_step = (m.end, m.tid, m.piece)

            if best_step is None:
                # byte fallback on current UTF-8 bytes
                b = text[i].encode("utf-8")
                if not b:
                    raise ValueError("Empty UTF-8 encoding unexpectedly encountered")
                if any(byte not in self.byte_tid_by_val for byte in b):
                    missing = [byte for byte in b if byte not in self.byte_tid_by_val]
                    raise ValueError(f"Missing byte pieces for bytes={missing}")

                # emit first byte here, remainder handled by subsequent positions? No.
                # To keep DP exact, fallback consumes the full character as a sequence of byte tokens.
                # We treat this as len(b) tokens advancing one character.
                cost = len(b) + dp[i + 1]
                best_cost = cost
                # special marker; decode later by re-emitting bytes for char
                best_step = (i + 1, -1, text[i])

            dp[i] = best_cost
            nxt[i] = best_step

        ids: list[int] = []
        pieces: list[str] = []
        i = 0
        while i < n:
            step = nxt[i]
            if step is None:
                raise RuntimeError(f"DP reconstruction failed at position {i}")

            end, tid, piece = step
            if tid != -1:
                ids.append(tid)
                pieces.append(piece)
            else:
                # byte fallback for one Unicode character
                for byte in piece.encode("utf-8"):
                    btid = self.byte_tid_by_val[byte]
                    ids.append(btid)
                    pieces.append(self.id_to_piece[btid])
            i = end

        return ids, pieces


# ----------------------------
# Diff / display helpers
# ----------------------------

def short_text(s: str, max_len: int = 220) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def piece_counts_used(pieces: list[str], target_set: set[str]) -> Counter:
    c = Counter()
    for p in pieces:
        if p in target_set:
            c[p] += 1
    return c


def seq_diff_blocks(a: list[str], b: list[str], max_blocks: int = 8) -> list[tuple[str, list[str], list[str]]]:
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append((tag, a[i1:i2], b[j1:j2]))
        if len(out) >= max_blocks:
            break
    return out


def token_windows(
    pieces: list[str],
    targets: set[str],
    max_hits: int = 5,
    radius: int = 3,
) -> list[str]:
    out = []
    for i, p in enumerate(pieces):
        if p not in targets:
            continue
        lo = max(0, i - radius)
        hi = min(len(pieces), i + radius + 1)
        window = pieces[lo:hi]
        marked = []
        for j, q in enumerate(window, start=lo):
            if j == i:
                marked.append(f"[{q}]")
            else:
                marked.append(q)
        out.append(" | ".join(marked))
        if len(out) >= max_hits:
            break
    return out


# ----------------------------
# Main inspection logic
# ----------------------------

def build_replacement_sets(base_vocab: list[dict], variant_vocab: list[dict]) -> tuple[set[str], set[str], list[tuple[int, str, str]]]:
    if len(base_vocab) != len(variant_vocab):
        raise ValueError(f"Vocab size mismatch: base={len(base_vocab)} variant={len(variant_vocab)}")

    removed = set()
    added = set()
    changed = []

    for brow, vrow in zip(base_vocab, variant_vocab, strict=True):
        bid = int(brow["id"])
        vid = int(vrow["id"])
        if bid != vid:
            raise ValueError(f"ID mismatch at base_id={bid}, variant_id={vid}")

        bp = brow["piece"]
        vp = vrow["piece"]
        if bp != vp:
            removed.add(bp)
            added.add(vp)
            changed.append((bid, bp, vp))

    return removed, added, changed


def inspect_doc(
    doc_id: int,
    text: str,
    sp,
    custom_tok: CustomDPTokenizer,
    removed_pieces: set[str],
    added_pieces: set[str],
    max_diff_blocks: int,
    max_context_hits: int,
) -> None:
    sp_ids = sp.encode(text, out_type=int)
    sp_pieces = sp.encode(text, out_type=str)

    custom_ids, custom_pieces = custom_tok.encode(text)

    sp_n = len(sp_ids)
    custom_n = len(custom_ids)
    delta = custom_n - sp_n

    used_removed = piece_counts_used(sp_pieces, removed_pieces)
    used_added = piece_counts_used(custom_pieces, added_pieces)

    unused_added = sorted([p for p in added_pieces if used_added[p] == 0])

    print("=" * 120)
    print(f"DOC {doc_id}")
    print("-" * 120)
    print(f"text preview: {short_text(text, 320)}")
    print(f"sp_tokens={sp_n} custom_tokens={custom_n} delta_custom_minus_sp={delta}")
    print()

    print("top removed SP pieces that WERE used in this doc:")
    if used_removed:
        for piece, cnt in used_removed.most_common(12):
            print(f"  {piece!r}: {cnt}")
    else:
        print("  (none)")
    print()

    print("top added variant pieces that fired in this doc:")
    if used_added:
        for piece, cnt in used_added.most_common(12):
            print(f"  {piece!r}: {cnt}")
    else:
        print("  (none)")
    print()

    print(f"added pieces not used in this doc: {len(unused_added)}")
    if unused_added:
        print("  preview:", ", ".join(repr(x) for x in unused_added[:20]))
    print()

    print("segmentation diff blocks (SP -> variant):")
    diff_blocks = seq_diff_blocks(sp_pieces, custom_pieces, max_blocks=max_diff_blocks)
    if diff_blocks:
        for idx, (tag, left, right) in enumerate(diff_blocks, start=1):
            print(f"  block {idx} [{tag}]")
            print(f"    SP:      {left}")
            print(f"    variant: {right}")
    else:
        print("  (no differences)")
    print()

    print("contexts where REMOVED SP pieces appeared:")
    removed_windows = token_windows(sp_pieces, removed_pieces, max_hits=max_context_hits, radius=3)
    if removed_windows:
        for w in removed_windows:
            print(f"  {w}")
    else:
        print("  (none)")
    print()

    print("contexts where ADDED variant pieces fired:")
    added_windows = token_windows(custom_pieces, added_pieces, max_hits=max_context_hits, radius=3)
    if added_windows:
        for w in added_windows:
            print(f"  {w}")
    else:
        print("  (none)")
    print()

    print("SP first 80 pieces:")
    print("  ", sp_pieces[:80])
    print()
    print("variant first 80 pieces:")
    print("  ", custom_pieces[:80])
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--sp-model", type=Path, required=True)
    ap.add_argument("--base-vocab", type=Path, required=True)
    ap.add_argument("--variant-vocab", type=Path, required=True)
    ap.add_argument("--doc-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-diff-blocks", type=int, default=8)
    ap.add_argument("--max-context-hits", type=int, default=8)
    args = ap.parse_args()

    docs = load_jsonl(args.docs)
    max_doc_id = len(docs) - 1

    for doc_id in args.doc_ids:
        if doc_id < 0 or doc_id > max_doc_id:
            raise ValueError(f"doc_id {doc_id} out of range [0, {max_doc_id}]")

    sp = spm.SentencePieceProcessor(model_file=str(args.sp_model))
    base_vocab = load_vocab_jsonl(args.base_vocab)
    variant_vocab = load_vocab_jsonl(args.variant_vocab)
    removed_pieces, added_pieces, changed = build_replacement_sets(base_vocab, variant_vocab)

    print(f"changed_token_slots={len(changed)}")
    print("replacement preview:")
    for tid, old_piece, new_piece in changed[:20]:
        print(f"  id={tid}: {old_piece!r} -> {new_piece!r}")
    print()

    custom_tok = CustomDPTokenizer(variant_vocab)

    for doc_id in args.doc_ids:
        row = docs[doc_id]
        text = extract_text(row)
        inspect_doc(
            doc_id=doc_id,
            text=text,
            sp=sp,
            custom_tok=custom_tok,
            removed_pieces=removed_pieces,
            added_pieces=added_pieces,
            max_diff_blocks=args.max_diff_blocks,
            max_context_hits=args.max_context_hits,
        )


if __name__ == "__main__":
    main()