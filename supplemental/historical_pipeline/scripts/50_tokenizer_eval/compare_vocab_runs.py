import json
import sentencepiece as spm
from collections import Counter, defaultdict

BASE_MODEL = "data/tokenizers/fineweb_1024_bpe.model"
VOCAB_V6 = "analysis/custom_vocab_full_v6.jsonl"
VOCAB_V7 = "analysis/custom_vocab_full_v7.jsonl"
SAMPLE_TEXT = "sample_text_large.txt"
MAX_LINE_REPORT = 40


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class GreedyTokenizer:
    def __init__(self, vocab_rows):
        self.rows = vocab_rows
        self.id_to_piece_map = {row["id"]: row["piece"] for row in vocab_rows}
        self.piece_to_id_map = {row["piece"]: row["id"] for row in vocab_rows}

        self.max_piece_len = max(len(p) for p in self.piece_to_id_map)
        self.pieces_by_len = defaultdict(dict)
        for piece, pid in self.piece_to_id_map.items():
            self.pieces_by_len[len(piece)][piece] = pid

    def id_to_piece(self, idx):
        return self.id_to_piece_map[idx]

    def encode(self, text):
        ids = []
        i = 0
        n = len(text)

        while i < n:
            matched = False

            max_len = min(self.max_piece_len, n - i)
            for L in range(max_len, 0, -1):
                chunk = text[i:i+L]
                if chunk in self.pieces_by_len[L]:
                    ids.append(self.pieces_by_len[L][chunk])
                    i += L
                    matched = True
                    break

            if matched:
                continue

            # UTF-8 byte fallback
            b = text[i].encode("utf-8")
            for byte in b:
                piece = f"<0x{byte:02X}>"
                if piece not in self.piece_to_id_map:
                    raise ValueError(f"Missing byte fallback piece for {repr(text[i])} -> {piece}")
                ids.append(self.piece_to_id_map[piece])
            i += 1

        return ids


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def top_counter(counter, n=30):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main():
    # baseline
    sp = spm.SentencePieceProcessor()
    if not sp.load(BASE_MODEL):
        raise FileNotFoundError(f"Could not load baseline model: {BASE_MODEL}")

    # vocabs
    v6_rows = load_jsonl(VOCAB_V6)
    v7_rows = load_jsonl(VOCAB_V7)

    tok_v6 = GreedyTokenizer(v6_rows)
    tok_v7 = GreedyTokenizer(v7_rows)

    vocab_v6 = {row["piece"] for row in v6_rows}
    vocab_v7 = {row["piece"] for row in v7_rows}

    only_v6 = vocab_v6 - vocab_v7
    only_v7 = vocab_v7 - vocab_v6
    both = vocab_v6 & vocab_v7

    print("VOCAB DIFF")
    print(f"v6 only: {len(only_v6)}")
    print(f"v7 only: {len(only_v7)}")
    print(f"shared : {len(both)}")
    print()

    print("Top v6-only pieces:")
    for p in sorted(list(only_v6))[:80]:
        print(repr(p))
    print()

    print("Top v7-only pieces:")
    for p in sorted(list(only_v7))[:80]:
        print(repr(p))
    print()

    lines = load_lines(SAMPLE_TEXT)

    total_base = 0
    total_v6 = 0
    total_v7 = 0

    v6_better_than_v7 = []
    v7_better_than_v6 = []
    ties = 0

    v6_unique_piece_usage_on_wins = Counter()
    v7_unique_piece_usage_on_wins = Counter()

    v6_unique_piece_usage_on_losses = Counter()
    v7_unique_piece_usage_on_losses = Counter()

    for idx, text in enumerate(lines):
        base_ids = sp.encode(text, out_type=int)
        v6_ids = tok_v6.encode(text)
        v7_ids = tok_v7.encode(text)

        base_len = len(base_ids)
        v6_len = len(v6_ids)
        v7_len = len(v7_ids)

        total_base += base_len
        total_v6 += v6_len
        total_v7 += v7_len

        v6_pieces = [tok_v6.id_to_piece(i) for i in v6_ids]
        v7_pieces = [tok_v7.id_to_piece(i) for i in v7_ids]

        diff = v6_len - v7_len  # negative means v6 better

        record = {
            "line_idx": idx,
            "text": text,
            "base_len": base_len,
            "v6_len": v6_len,
            "v7_len": v7_len,
            "v6_delta_vs_base": v6_len - base_len,
            "v7_delta_vs_base": v7_len - base_len,
            "v6_vs_v7": diff,
            "v6_pieces": v6_pieces,
            "v7_pieces": v7_pieces,
        }

        v6_unique_used = [p for p in v6_pieces if p in only_v6]
        v7_unique_used = [p for p in v7_pieces if p in only_v7]

        if diff < 0:
            v6_better_than_v7.append(record)
            v6_unique_piece_usage_on_wins.update(v6_unique_used)
            v7_unique_piece_usage_on_losses.update(v7_unique_used)
        elif diff > 0:
            v7_better_than_v6.append(record)
            v7_unique_piece_usage_on_wins.update(v7_unique_used)
            v6_unique_piece_usage_on_losses.update(v6_unique_used)
        else:
            ties += 1

    print("TOTALS")
    print(f"BASE TOTAL: {total_base}")
    print(f"V6 TOTAL:   {total_v6}  delta={total_v6 - total_base}")
    print(f"V7 TOTAL:   {total_v7}  delta={total_v7 - total_base}")
    print()

    print("HEAD-TO-HEAD")
    print(f"V6 better: {len(v6_better_than_v7)}")
    print(f"V7 better: {len(v7_better_than_v6)}")
    print(f"Ties:      {ties}")
    print()

    v6_better_than_v7.sort(key=lambda r: (r["v6_vs_v7"], r["v6_delta_vs_base"] ))
    v7_better_than_v6.sort(key=lambda r: (-r["v6_vs_v7"], r["v7_delta_vs_base"] ))

    print(f"TOP {MAX_LINE_REPORT} LINES WHERE V6 BEATS V7")
    for r in v6_better_than_v7[:MAX_LINE_REPORT]:
        print("-" * 80)
        print(f'line={r["line_idx"]} base={r["base_len"]} v6={r["v6_len"]} v7={r["v7_len"]} diff={r["v6_vs_v7"]}')
        print(repr(r["text"]))
        print("V6:", r["v6_pieces"])
        print("V7:", r["v7_pieces"])

    print()
    print(f"TOP {MAX_LINE_REPORT} LINES WHERE V7 BEATS V6")
    for r in v7_better_than_v6[:MAX_LINE_REPORT]:
        print("-" * 80)
        print(f'line={r["line_idx"]} base={r["base_len"]} v6={r["v6_len"]} v7={r["v7_len"]} diff={r["v6_vs_v7"]}')
        print(repr(r["text"]))
        print("V6:", r["v6_pieces"])
        print("V7:", r["v7_pieces"])

    print()
    print("TOP V6-ONLY PIECES USED ON V6 WINNING LINES")
    for piece, count in top_counter(v6_unique_piece_usage_on_wins, 40):
        print(f"{count:6d}  {repr(piece)}")

    print()
    print("TOP V7-ONLY PIECES USED ON V7 WINNING LINES")
    for piece, count in top_counter(v7_unique_piece_usage_on_wins, 40):
        print(f"{count:6d}  {repr(piece)}")

    print()
    print("TOP V6-ONLY PIECES USED ON V6 LOSING LINES")
    for piece, count in top_counter(v6_unique_piece_usage_on_losses, 40):
        print(f"{count:6d}  {repr(piece)}")

    print()
    print("TOP V7-ONLY PIECES USED ON V7 LOSING LINES")
    for piece, count in top_counter(v7_unique_piece_usage_on_losses, 40):
        print(f"{count:6d}  {repr(piece)}")


if __name__ == "__main__":
    main()