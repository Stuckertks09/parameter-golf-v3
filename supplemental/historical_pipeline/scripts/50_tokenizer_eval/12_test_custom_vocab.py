import json
import re
from collections import defaultdict

import sentencepiece as spm

VOCAB_PATH = "analysis/grid_vocabs_v2/b650_p36_w32_bs-priority_ps-combined_ws-combined_pin-none_ms10.jsonl"
INPUT = "sample_text_large.txt"
BASE_MODEL = "data/tokenizers/fineweb_1024_bpe.model"


def load_vocab(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda x: x["id"])
    return rows


class GreedyVocabTokenizer:
    BYTE_RE = re.compile(r"^<0x([0-9A-F]{2})>$")

    def __init__(self, rows):
        self.rows = rows
        self.id_to_piece = {row["id"]: row["piece"] for row in rows}
        self.piece_to_id = {row["piece"]: row["id"] for row in rows}

        # Longest match first, then lower id for determinism.
        self.sorted_pieces = sorted(
            [(row["piece"], row["id"]) for row in rows],
            key=lambda x: (-len(x[0]), x[1]),
        )

        # Index by first character for speed.
        first_char_index = defaultdict(list)
        for piece, piece_id in self.sorted_pieces:
            if piece:
                first_char_index[piece[0]].append((piece, piece_id))
        self.first_char_index = dict(first_char_index)

    def encode(self, text: str):
        out = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            matched = False

            for piece, piece_id in self.first_char_index.get(ch, []):
                if text.startswith(piece, i):
                    out.append(piece_id)
                    i += len(piece)
                    matched = True
                    break

            if matched:
                continue

            # UTF-8 byte fallback for any unmatched character.
            utf8_bytes = text[i].encode("utf-8")
            for b in utf8_bytes:
                piece = f"<0x{b:02X}>"
                if piece not in self.piece_to_id:
                    raise ValueError(f"Missing byte fallback piece for byte {b}")
                out.append(self.piece_to_id[piece])

            i += 1

        return out

    def decode(self, ids):
        out_bytes = bytearray()

        for token_id in ids:
            piece = self.id_to_piece[token_id]
            m = self.BYTE_RE.match(piece)

            if m:
                out_bytes.append(int(m.group(1), 16))
            else:
                out_bytes.extend(piece.encode("utf-8"))

        return out_bytes.decode("utf-8", errors="strict")


def main():
    rows = load_vocab(VOCAB_PATH)
    tok = GreedyVocabTokenizer(rows)

    base_sp = spm.SentencePieceProcessor()
    ok = base_sp.load(BASE_MODEL)
    if not ok:
        raise FileNotFoundError(f"Could not load baseline model: {BASE_MODEL}")

    sanity = [
        "of the",
        "in the",
        "going to",
        "as well as",
        "this is a test of the system",
        "Hello, world!",
        "Price is $19.99\nNext line.",
    ]

    print("SANITY CHECKS\n")
    for s in sanity:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        pieces = [tok.id_to_piece[i] for i in ids[:20]]

        print("TEXT:   ", repr(s))
        print("IDS:    ", ids[:20], "..." if len(ids) > 20 else "")
        print("PIECES: ", pieces, "..." if len(ids) > 20 else "")
        print("OK:     ", decoded == s)
        print()

        if decoded != s:
            raise ValueError(f"Roundtrip failed: {repr(s)} -> {repr(decoded)}")

    total_lines = 0
    base_total = 0
    custom_total = 0
    improved = 0
    worse = 0
    same = 0
    examples_better = []
    examples_worse = []

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")

            base_ids = base_sp.encode(text, out_type=int)
            custom_ids = tok.encode(text)
            decoded = tok.decode(custom_ids)

            if decoded != text:
                raise ValueError(f"Roundtrip mismatch on line: {repr(text[:200])}")

            b = len(base_ids)
            c = len(custom_ids)

            total_lines += 1
            base_total += b
            custom_total += c

            if c < b:
                improved += 1
                if len(examples_better) < 10:
                    examples_better.append({
                        "text": text,
                        "base_len": b,
                        "custom_len": c,
                        "custom_pieces": [tok.id_to_piece[i] for i in custom_ids[:40]],
                    })
            elif c > b:
                worse += 1
                if len(examples_worse) < 10:
                    examples_worse.append({
                        "text": text,
                        "base_len": b,
                        "custom_len": c,
                        "custom_pieces": [tok.id_to_piece[i] for i in custom_ids[:40]],
                    })
            else:
                same += 1

    debug_texts = [
        "of the United States",
        "going to be a great",
        "This is a simple test",
        "the government of the",
    ]

    print("\nDEBUG COMPARISON\n")
    for text in debug_texts:
        base_ids = base_sp.encode(text, out_type=int)
        custom_ids = tok.encode(text)
        
        base_pieces = [base_sp.id_to_piece(i) for i in base_ids]
        custom_pieces = [tok.id_to_piece[i] for i in custom_ids]
        
        print("TEXT:          ", repr(text))
        print("BASE pieces:   ", base_pieces)
        print("CUSTOM pieces: ", custom_pieces)
        print("BASE len:      ", len(base_ids))
        print("CUSTOM len:    ", len(custom_ids))
        print()

    print("\nRESULTS\n")
    print("LINES:      ", total_lines)

    print("\nRESULTS\n")
    print("LINES:      ", total_lines)
    print("BASE TOTAL: ", base_total)
    print("CUST TOTAL: ", custom_total)
    print("ABS SAVED:  ", base_total - custom_total)
    print("PCT SAVED:  ", (base_total - custom_total) / base_total if base_total else 0.0)
    print("IMPROVED:   ", improved)
    print("WORSE:      ", worse)
    print("SAME:       ", same)

    print("\nBETTER EXAMPLES\n")
    for ex in examples_better:
        print("=" * 100)
        print("TEXT:       ", ex["text"])
        print("BASE LEN:   ", ex["base_len"])
        print("CUST LEN:   ", ex["custom_len"])
        print("CUST PIECES:", ex["custom_pieces"])

    print("\nWORSE EXAMPLES\n")
    for ex in examples_worse:
        print("=" * 100)
        print("TEXT:       ", ex["text"])
        print("BASE LEN:   ", ex["base_len"])
        print("CUST LEN:   ", ex["custom_len"])
        print("CUST PIECES:", ex["custom_pieces"])


if __name__ == "__main__":
    main()