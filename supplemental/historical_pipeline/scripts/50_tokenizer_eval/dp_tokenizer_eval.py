import json
import math
from collections import defaultdict, Counter

import sentencepiece as spm

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL = "data/tokenizers/fineweb_1024_bpe.model"
VOCAB_PATH = "analysis/grid_vocabs_v2/b680_p36_w32_bs-priority_ps-combined_ws-combined_pin-none_ms10.jsonl"
SAMPLE_TEXT = "sample_text_large.txt"

MAX_LINE_REPORT = 40
MAX_FIXED_LINE_REPORT = 40


# =========================================================
# HELPERS
# =========================================================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def top_counter(counter, n=40):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


# =========================================================
# TOKENIZER
# =========================================================
class CustomTokenizer:
    """
    Supports:
    - greedy longest-match tokenization
    - DP minimum-token-count tokenization
    - UTF-8 byte fallback via <0xXX> pieces
    """

    def __init__(self, vocab_rows):
        self.rows = vocab_rows
        self.id_to_piece_map = {row["id"]: row["piece"] for row in vocab_rows}
        self.piece_to_id_map = {row["piece"]: row["id"] for row in vocab_rows}

        if not self.piece_to_id_map:
            raise ValueError("Empty vocab")

        self.max_piece_len = max(len(piece) for piece in self.piece_to_id_map)

        self.pieces_by_len = defaultdict(dict)
        self.lengths_desc = set()
        for piece, pid in self.piece_to_id_map.items():
            L = len(piece)
            self.pieces_by_len[L][piece] = pid
            self.lengths_desc.add(L)

        self.lengths_desc = sorted(self.lengths_desc, reverse=True)

        # Precompute byte fallback ids
        self.byte_piece_to_id = {}
        for b in range(256):
            piece = f"<0x{b:02X}>"
            if piece in self.piece_to_id_map:
                self.byte_piece_to_id[b] = self.piece_to_id_map[piece]

    def id_to_piece(self, idx):
        return self.id_to_piece_map[idx]

    def ids_to_pieces(self, ids):
        return [self.id_to_piece(i) for i in ids]

    def _byte_fallback_ids_for_char(self, ch: str):
        out = []
        for b in ch.encode("utf-8"):
            if b not in self.byte_piece_to_id:
                piece = f"<0x{b:02X}>"
                raise ValueError(f"Missing byte fallback piece for char={repr(ch)} byte={piece}")
            out.append(self.byte_piece_to_id[b])
        return out

    def encode_greedy(self, text: str):
        ids = []
        i = 0
        n = len(text)

        while i < n:
            matched = False
            max_len = min(self.max_piece_len, n - i)

            for L in self.lengths_desc:
                if L > max_len:
                    continue
                chunk = text[i:i+L]
                pid = self.pieces_by_len[L].get(chunk)
                if pid is not None:
                    ids.append(pid)
                    i += L
                    matched = True
                    break

            if matched:
                continue

            ids.extend(self._byte_fallback_ids_for_char(text[i]))
            i += 1

        return ids

    def encode_dp(self, text: str):
        """
        DP for minimum token count.
        Tie-breakers:
        1. fewer total tokens
        2. larger first piece length
        """
        n = len(text)

        # dp_cost[i] = minimum number of tokens needed to encode text[i:]
        dp_cost = [math.inf] * (n + 1)
        dp_next = [None] * (n + 1)
        dp_cost[n] = 0

        for i in range(n - 1, -1, -1):
            best_cost = math.inf
            best_choice = None

            max_len = min(self.max_piece_len, n - i)

            # Try all normal pieces
            for L in self.lengths_desc:
                if L > max_len:
                    continue
                chunk = text[i:i+L]
                pid = self.pieces_by_len[L].get(chunk)
                if pid is None:
                    continue

                cand_cost = 1 + dp_cost[i + L]
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_choice = ("piece", pid, L)
                elif cand_cost == best_cost and best_choice is not None:
                    # tie-break: prefer longer piece
                    if L > best_choice[2]:
                        best_choice = ("piece", pid, L)

            # Byte fallback for one Unicode char
            fallback_ids = self._byte_fallback_ids_for_char(text[i])
            fallback_cost = len(fallback_ids) + dp_cost[i + 1]

            if fallback_cost < best_cost:
                best_cost = fallback_cost
                best_choice = ("bytes", fallback_ids, 1)
            elif fallback_cost == best_cost and best_choice is None:
                best_choice = ("bytes", fallback_ids, 1)

            dp_cost[i] = best_cost
            dp_next[i] = best_choice

        # Reconstruct
        ids = []
        i = 0
        while i < n:
            choice = dp_next[i]
            if choice is None:
                raise RuntimeError(f"DP reconstruction failed at position {i}")

            kind, payload, advance = choice
            if kind == "piece":
                ids.append(payload)
            else:
                ids.extend(payload)

            i += advance

        return ids


# =========================================================
# MAIN
# =========================================================
def main():
    lines = load_lines(SAMPLE_TEXT)
    vocab_rows = load_jsonl(VOCAB_PATH)

    tok = CustomTokenizer(vocab_rows)

    sp = spm.SentencePieceProcessor()
    ok = sp.load(BASE_MODEL)
    if not ok:
        raise FileNotFoundError(f"Could not load baseline model: {BASE_MODEL}")

    total_base = 0
    total_greedy = 0
    total_dp = 0

    greedy_better_than_base = 0
    greedy_worse_than_base = 0
    greedy_same_as_base = 0

    dp_better_than_base = 0
    dp_worse_than_base = 0
    dp_same_as_base = 0

    dp_better_than_greedy = 0
    dp_worse_than_greedy = 0
    dp_same_as_greedy = 0

    fixed_lines = []
    made_worse_lines = []
    same_lines = []

    piece_gain_counter = Counter()
    piece_loss_counter = Counter()

    for idx, text in enumerate(lines):
        base_ids = sp.encode(text, out_type=int)
        greedy_ids = tok.encode_greedy(text)
        dp_ids = tok.encode_dp(text)

        base_len = len(base_ids)
        greedy_len = len(greedy_ids)
        dp_len = len(dp_ids)

        total_base += base_len
        total_greedy += greedy_len
        total_dp += dp_len

        # greedy vs baseline
        if greedy_len < base_len:
            greedy_better_than_base += 1
        elif greedy_len > base_len:
            greedy_worse_than_base += 1
        else:
            greedy_same_as_base += 1

        # dp vs baseline
        if dp_len < base_len:
            dp_better_than_base += 1
        elif dp_len > base_len:
            dp_worse_than_base += 1
        else:
            dp_same_as_base += 1

        # dp vs greedy
        record = {
            "line_idx": idx,
            "text": text,
            "base_len": base_len,
            "greedy_len": greedy_len,
            "dp_len": dp_len,
            "greedy_delta_vs_base": greedy_len - base_len,
            "dp_delta_vs_base": dp_len - base_len,
            "dp_minus_greedy": dp_len - greedy_len,
            "greedy_pieces": tok.ids_to_pieces(greedy_ids),
            "dp_pieces": tok.ids_to_pieces(dp_ids),
        }

        if dp_len < greedy_len:
            dp_better_than_greedy += 1
            fixed_lines.append(record)

            greedy_set = Counter(record["greedy_pieces"])
            dp_set = Counter(record["dp_pieces"])

            for piece, count in dp_set.items():
                if greedy_set[piece] < count:
                    piece_gain_counter[piece] += count - greedy_set[piece]
            for piece, count in greedy_set.items():
                if dp_set[piece] < count:
                    piece_loss_counter[piece] += count - dp_set[piece]

        elif dp_len > greedy_len:
            dp_worse_than_greedy += 1
            made_worse_lines.append(record)
        else:
            dp_same_as_greedy += 1
            same_lines.append(record)

    print("CONFIG")
    print(f"BASE_MODEL: {BASE_MODEL}")
    print(f"VOCAB_PATH: {VOCAB_PATH}")
    print(f"SAMPLE_TEXT: {SAMPLE_TEXT}")
    print()

    print("TOTALS")
    print(f"LINES:        {len(lines)}")
    print(f"BASE TOTAL:   {total_base}")
    print(f"GREEDY TOTAL: {total_greedy}")
    print(f"DP TOTAL:     {total_dp}")
    print()

    print("VS BASELINE")
    print(f"GREEDY abs_saved: {total_base - total_greedy}")
    print(f"GREEDY pct_saved: {(total_base - total_greedy) / total_base:.12f}")
    print(f"GREEDY improved / worse / same: {greedy_better_than_base} / {greedy_worse_than_base} / {greedy_same_as_base}")
    print()
    print(f"DP abs_saved:     {total_base - total_dp}")
    print(f"DP pct_saved:     {(total_base - total_dp) / total_base:.12f}")
    print(f"DP improved / worse / same:     {dp_better_than_base} / {dp_worse_than_base} / {dp_same_as_base}")
    print()

    print("DP VS GREEDY")
    print(f"DP better lines: {dp_better_than_greedy}")
    print(f"DP worse lines:  {dp_worse_than_greedy}")
    print(f"DP same lines:   {dp_same_as_greedy}")
    print(f"Token gain from DP over greedy: {total_greedy - total_dp}")
    print()

    fixed_lines.sort(key=lambda r: (r["dp_minus_greedy"], r["dp_delta_vs_base"]))
    made_worse_lines.sort(key=lambda r: (-r["dp_minus_greedy"], r["dp_delta_vs_base"]))

    print(f"TOP {MAX_FIXED_LINE_REPORT} LINES WHERE DP BEATS GREEDY")
    for r in fixed_lines[:MAX_FIXED_LINE_REPORT]:
        print("-" * 100)
        print(
            f'line={r["line_idx"]} '
            f'base={r["base_len"]} '
            f'greedy={r["greedy_len"]} '
            f'dp={r["dp_len"]} '
            f'dp_minus_greedy={r["dp_minus_greedy"]}'
        )
        print(repr(r["text"]))
        print("GREEDY:", r["greedy_pieces"])
        print("DP:    ", r["dp_pieces"])

    if made_worse_lines:
        print()
        print(f"TOP {MAX_LINE_REPORT} LINES WHERE DP IS WORSE THAN GREEDY")
        for r in made_worse_lines[:MAX_LINE_REPORT]:
            print("-" * 100)
            print(
                f'line={r["line_idx"]} '
                f'base={r["base_len"]} '
                f'greedy={r["greedy_len"]} '
                f'dp={r["dp_len"]} '
                f'dp_minus_greedy={r["dp_minus_greedy"]}'
            )
            print(repr(r["text"]))
            print("GREEDY:", r["greedy_pieces"])
            print("DP:    ", r["dp_pieces"])

    print()
    print("TOP PIECES GAINED BY DP")
    for piece, count in top_counter(piece_gain_counter, 50):
        print(f"{count:6d}  {repr(piece)}")

    print()
    print("TOP PIECES REDUCED BY DP")
    for piece, count in top_counter(piece_loss_counter, 50):
        print(f"{count:6d}  {repr(piece)}")


if __name__ == "__main__":
    main()