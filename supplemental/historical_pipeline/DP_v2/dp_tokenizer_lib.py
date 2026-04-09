#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DPVocab:
    rows: list[dict]
    id_to_piece: dict[int, str]
    piece_to_id: dict[str, int]
    byte_piece_to_id: dict[int, int]
    candidates_by_first_char: dict[str, list[str]]
    vocab_size: int


@dataclass
class EncodeResult:
    ids: list[int]
    token_count: int
    fallback_count: int
    fallback_runs: int
    boundary_fallback_count: int
    score: tuple[int, int, int, int, int]


def load_vocab_rows(path: str | Path) -> list[dict]:
    path = Path(path)
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


def build_piece_maps(rows: list[dict]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_piece: dict[int, str] = {}
    piece_to_id: dict[str, int] = {}
    for row in rows:
        tid = int(row["id"])
        piece = row["piece"]
        id_to_piece[tid] = piece
        piece_to_id[piece] = tid
    return id_to_piece, piece_to_id


def build_byte_piece_map(piece_to_id: dict[str, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for b in range(256):
        piece = f"<0x{b:02X}>"
        tid = piece_to_id.get(piece)
        if tid is not None:
            out[b] = tid
    return out


def build_candidates_by_first_char(piece_to_id: dict[str, int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for piece in piece_to_id.keys():
        if not piece or is_byte_piece(piece):
            continue
        out.setdefault(piece[0], []).append(piece)
    for ch in out:
        out[ch].sort(key=len, reverse=True)
    return out


def load_dp_vocab(path: str | Path) -> DPVocab:
    rows = load_vocab_rows(path)
    id_to_piece, piece_to_id = build_piece_maps(rows)
    byte_piece_to_id = build_byte_piece_map(piece_to_id)
    candidates_by_first_char = build_candidates_by_first_char(piece_to_id)

    max_id = max(id_to_piece) if id_to_piece else -1
    if max_id + 1 != len(rows):
        raise ValueError(f"Vocab IDs must be contiguous from 0..N-1, got max_id={max_id}, rows={len(rows)}")

    missing_bytes = [b for b in range(256) if b not in byte_piece_to_id]
    if missing_bytes:
        raise ValueError(
            f"Missing byte fallback pieces for bytes: {missing_bytes[:10]}"
            f"{'...' if len(missing_bytes) > 10 else ''}"
        )

    return DPVocab(
        rows=rows,
        id_to_piece=id_to_piece,
        piece_to_id=piece_to_id,
        byte_piece_to_id=byte_piece_to_id,
        candidates_by_first_char=candidates_by_first_char,
        vocab_size=len(rows),
    )


def is_boundary_char(ch: str) -> bool:
    return ch.isspace() or ch in {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'"}


def byte_fallback_ids_for_char(ch: str, vocab: DPVocab) -> list[int]:
    return [vocab.byte_piece_to_id[b] for b in ch.encode("utf-8")]


def boundary_penalty_for_fallback(ch: str) -> int:
    if ch == "\n":
        return 4
    if ch == "\t":
        return 3
    if ch == " ":
        return 3
    if is_boundary_char(ch):
        return 2
    return 0


def encode_dp_ids_min_tokens(text: str, vocab: DPVocab) -> list[int]:
    n = len(text)
    if n == 0:
        return []

    inf = 10**9
    best_cost = [inf] * (n + 1)
    best_next = [-1] * (n + 1)
    best_tid = [-1] * (n + 1)
    best_span = [-1] * (n + 1)

    best_cost[n] = 0

    for i in range(n - 1, -1, -1):
        first = text[i]
        candidates = vocab.candidates_by_first_char.get(first, [])

        for piece in candidates:
            if text.startswith(piece, i):
                j = i + len(piece)
                cost = 1 + best_cost[j]
                span = len(piece)
                if cost < best_cost[i] or (cost == best_cost[i] and span > best_span[i]):
                    best_cost[i] = cost
                    best_next[i] = j
                    best_tid[i] = vocab.piece_to_id[piece]
                    best_span[i] = span

        ch_bytes = text[i].encode("utf-8")
        byte_cost = len(ch_bytes) + best_cost[i + 1]
        if byte_cost < best_cost[i]:
            best_cost[i] = byte_cost
            best_next[i] = i + 1
            best_tid[i] = -2 if len(ch_bytes) > 1 else vocab.byte_piece_to_id[ch_bytes[0]]
            best_span[i] = 1

    if best_cost[0] >= inf:
        raise RuntimeError("DP encode failed to find a valid path")

    out: list[int] = []
    i = 0
    while i < n:
        tid = best_tid[i]
        j = best_next[i]
        if j < 0:
            raise RuntimeError(f"Broken DP backpointer at position {i}")
        if tid == -2:
            out.extend(byte_fallback_ids_for_char(text[i], vocab))
        else:
            out.append(tid)
        i = j
    return out


def encode_dp_ids_boundary(text: str, vocab: DPVocab) -> tuple[list[int], tuple[int, int, int, int, int]]:
    """
    Lexicographic objective:
      1. minimize token count
      2. minimize fallback token count
      3. minimize fallback runs
      4. minimize boundary fallback penalty
      5. prefer longer first piece
    """
    n = len(text)
    if n == 0:
        return [], (0, 0, 0, 0, 0)

    inf = 10**9
    sentinel = (inf, inf, inf, inf, 0)

    best_score = [sentinel for _ in range(n + 1)]
    best_next = [-1] * (n + 1)
    best_tid = [-1] * (n + 1)
    best_kind = [""] * (n + 1)

    best_score[n] = (0, 0, 0, 0, 0)
    best_kind[n] = "end"

    for i in range(n - 1, -1, -1):
        first = text[i]
        candidates = vocab.candidates_by_first_char.get(first, [])

        for piece in candidates:
            if text.startswith(piece, i):
                j = i + len(piece)
                tail = best_score[j]
                score = (
                    1 + tail[0],
                    tail[1],
                    tail[2],
                    tail[3],
                    -len(piece),
                )
                if score < best_score[i]:
                    best_score[i] = score
                    best_next[i] = j
                    best_tid[i] = vocab.piece_to_id[piece]
                    best_kind[i] = "piece"

        ch = text[i]
        ch_bytes = ch.encode("utf-8")
        j = i + 1
        tail = best_score[j]
        next_is_fallback = best_kind[j] == "bytes"
        new_run = 0 if next_is_fallback else 1
        fallback_len = len(ch_bytes)
        score = (
            fallback_len + tail[0],
            fallback_len + tail[1],
            new_run + tail[2],
            boundary_penalty_for_fallback(ch) + tail[3],
            -1,
        )
        if score < best_score[i]:
            best_score[i] = score
            best_next[i] = j
            best_tid[i] = -2 if fallback_len > 1 else vocab.byte_piece_to_id[ch_bytes[0]]
            best_kind[i] = "bytes"

    if best_next[0] < 0:
        raise RuntimeError("DP encode failed to find a valid path")

    out: list[int] = []
    i = 0
    while i < n:
        tid = best_tid[i]
        j = best_next[i]
        if j < 0:
            raise RuntimeError(f"Broken DP backpointer at position {i}")
        if tid == -2:
            out.extend(byte_fallback_ids_for_char(text[i], vocab))
        else:
            out.append(tid)
        i = j

    return out, best_score[0]


def decode_ids(ids: list[int], vocab: DPVocab) -> str:
    out_parts: list[str] = []
    byte_buf = bytearray()

    def flush_bytes() -> None:
        nonlocal byte_buf
        if byte_buf:
            try:
                out_parts.append(byte_buf.decode("utf-8"))
            except UnicodeDecodeError:
                out_parts.append(byte_buf.decode("utf-8", errors="replace"))
            byte_buf = bytearray()

    for tid in ids:
        piece = vocab.id_to_piece[tid]
        if is_byte_piece(piece):
            byte_val = int(piece[3:5], 16)
            byte_buf.append(byte_val)
        else:
            flush_bytes()
            out_parts.append(piece)

    flush_bytes()
    return "".join(out_parts)


def count_fallback_ids(ids: list[int], vocab: DPVocab) -> int:
    return sum(1 for tid in ids if is_byte_piece(vocab.id_to_piece[tid]))


def count_fallback_runs(ids: list[int], vocab: DPVocab) -> int:
    runs = 0
    prev_byte = False
    for tid in ids:
        is_byte = is_byte_piece(vocab.id_to_piece[tid])
        if is_byte and not prev_byte:
            runs += 1
        prev_byte = is_byte
    return runs


def count_boundary_fallback_ids(ids: list[int], vocab: DPVocab) -> int:
    total = 0
    for tid in ids:
        piece = vocab.id_to_piece[tid]
        if not is_byte_piece(piece):
            continue
        b = int(piece[3:5], 16)
        ch = bytes([b]).decode("utf-8", errors="ignore")
        if ch and is_boundary_char(ch):
            total += 1
    return total


def ids_to_pieces(ids: list[int], vocab: DPVocab) -> list[str]:
    return [vocab.id_to_piece[tid] for tid in ids]


def token_bytes_for_ids(ids: list[int], vocab: DPVocab) -> int:
    total = 0
    for tid in ids:
        piece = vocab.id_to_piece[tid]
        if is_byte_piece(piece):
            total += 1
        else:
            total += len(piece.encode("utf-8"))
    return total


def encode_dp(text: str, vocab: DPVocab, mode: str = "boundary") -> EncodeResult:
    if mode == "min_tokens":
        ids = encode_dp_ids_min_tokens(text, vocab)
        score = (len(ids), count_fallback_ids(ids, vocab), count_fallback_runs(ids, vocab), 0, 0)
    elif mode == "boundary":
        ids, score = encode_dp_ids_boundary(text, vocab)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return EncodeResult(
        ids=ids,
        token_count=len(ids),
        fallback_count=count_fallback_ids(ids, vocab),
        fallback_runs=count_fallback_runs(ids, vocab),
        boundary_fallback_count=count_boundary_fallback_ids(ids, vocab),
        score=score,
    )


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-jsonl", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--mode", choices=["min_tokens", "boundary"], default="boundary")
    args = ap.parse_args()

    vocab = load_dp_vocab(args.vocab_jsonl)
    result = encode_dp(args.text, vocab, mode=args.mode)
    pieces = ids_to_pieces(result.ids, vocab)

    print("MODE:   ", args.mode)
    print("TEXT:   ", repr(args.text))
    print("IDS:    ", result.ids)
    print("PIECES: ", pieces)
    print("TOKENS: ", result.token_count)
    print("FALLBK: ", result.fallback_count)
    print("RUNS:   ", result.fallback_runs)
    print("BNDRY:  ", result.boundary_fallback_count)
    print("SCORE:  ", result.score)
    print("DECODE: ", repr(decode_ids(result.ids, vocab)))


if __name__ == "__main__":
    _main()
