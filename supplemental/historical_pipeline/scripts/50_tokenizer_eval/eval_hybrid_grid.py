import csv
import json
import os
from collections import defaultdict

import sentencepiece as spm

BASE_MODEL = "data/tokenizers/fineweb_1024_bpe.model"
MANIFEST_PATH = "analysis/grid_vocabs_v2/manifest.jsonl"
SAMPLE_TEXT = "sample_text_large.txt"

RESULTS_JSONL = "analysis/grid_vocabs_v2/results.jsonl"
RESULTS_CSV = "analysis/grid_vocabs_v2/results.csv"


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


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class GreedyTokenizer:
    def __init__(self, vocab_rows):
        self.rows = vocab_rows
        self.id_to_piece_map = {row["id"]: row["piece"] for row in vocab_rows}
        self.piece_to_id_map = {row["piece"]: row["id"] for row in vocab_rows}

        self.max_piece_len = max(len(p) for p in self.piece_to_id_map)
        self.pieces_by_len = defaultdict(dict)
        for piece, pid in self.piece_to_id_map.items():
            self.pieces_by_len[len(piece)][piece] = pid

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
                    raise ValueError(
                        f"Missing byte fallback piece for char={repr(text[i])} byte={piece}"
                    )
                ids.append(self.piece_to_id_map[piece])
            i += 1

        return ids


def eval_one_vocab(sp, lines, manifest_row):
    vocab_path = manifest_row["path"]
    vocab_rows = load_jsonl(vocab_path)
    tok = GreedyTokenizer(vocab_rows)

    total_base = 0
    total_custom = 0
    improved = 0
    worse = 0
    same = 0

    for text in lines:
        base_ids = sp.encode(text, out_type=int)
        custom_ids = tok.encode(text)

        b = len(base_ids)
        c = len(custom_ids)

        total_base += b
        total_custom += c

        if c < b:
            improved += 1
        elif c > b:
            worse += 1
        else:
            same += 1

    abs_saved = total_base - total_custom
    pct_saved = abs_saved / total_base if total_base else 0.0

    result = dict(manifest_row)
    result.update({
        "lines": len(lines),
        "base_total": total_base,
        "cust_total": total_custom,
        "abs_saved": abs_saved,
        "pct_saved": pct_saved,
        "improved": improved,
        "worse": worse,
        "same": same,
    })
    return result


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = load_jsonl(MANIFEST_PATH)
    lines = load_lines(SAMPLE_TEXT)

    sp = spm.SentencePieceProcessor()
    ok = sp.load(BASE_MODEL)
    if not ok:
        raise FileNotFoundError(f"Could not load baseline model: {BASE_MODEL}")

    print(f"Evaluating {len(manifest)} vocab variants against {len(lines)} lines...")

    results = []
    for i, row in enumerate(manifest, start=1):
        try:
            result = eval_one_vocab(sp, lines, row)
            results.append(result)
            print(
                f"[{i}/{len(manifest)}] "
                f'{row["tag"]}  cust={result["cust_total"]}  '
                f'abs_saved={result["abs_saved"]}  worse={result["worse"]}'
            )
        except Exception as e:
            error_row = dict(row)
            error_row.update({
                "lines": len(lines),
                "base_total": None,
                "cust_total": None,
                "abs_saved": None,
                "pct_saved": None,
                "improved": None,
                "worse": None,
                "same": None,
                "error": str(e),
            })
            results.append(error_row)
            print(f'[{i}/{len(manifest)}] {row["tag"]}  ERROR: {e}')

    # sort best first: highest abs_saved, then lower cust_total, then fewer worse
    sortable = [r for r in results if r.get("abs_saved") is not None]
    sortable.sort(
        key=lambda r: (
            -r["abs_saved"],
            r["cust_total"],
            r["worse"],
            -r["improved"],
        )
    )

    errored = [r for r in results if r.get("abs_saved") is None]
    final_results = sortable + errored

    write_jsonl(RESULTS_JSONL, final_results)
    write_csv(RESULTS_CSV, final_results)

    print(f"\nWrote results -> {RESULTS_JSONL}")
    print(f"Wrote csv     -> {RESULTS_CSV}")

    print("\nTOP 20 CONFIGS")
    for row in final_results[:20]:
        if row.get("abs_saved") is None:
            continue
        print(
            f'{row["tag"]:<60} '
            f'cust={row["cust_total"]:<8} '
            f'abs_saved={row["abs_saved"]:<8} '
            f'pct={row["pct_saved"]:.6f} '
            f'improved={row["improved"]:<5} '
            f'worse={row["worse"]:<5} '
            f'same={row["same"]:<5}'
        )


if __name__ == "__main__":
    main()