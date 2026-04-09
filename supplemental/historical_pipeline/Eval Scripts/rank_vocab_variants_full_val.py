from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import sentencepiece as spm


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def load_vocab_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["id"]))
    return rows


class CustomDPTokenizer:
    def __init__(self, vocab_rows: list[dict]):
        self.id_to_piece = {int(r["id"]): r["piece"] for r in vocab_rows}
        self.piece_to_id = {r["piece"]: int(r["id"]) for r in vocab_rows}

        self.byte_piece_to_id: dict[int, int] = {}
        for b in range(256):
            piece = f"<0x{b:02X}>"
            tid = self.piece_to_id.get(piece)
            if tid is not None:
                self.byte_piece_to_id[b] = tid

        missing = [b for b in range(256) if b not in self.byte_piece_to_id]
        if missing:
            raise ValueError(f"Missing byte pieces for bytes: {missing[:10]}")

        self.candidates_by_first_char: dict[str, list[tuple[str, int, int]]] = {}
        for piece, tid in self.piece_to_id.items():
            if not piece or is_byte_piece(piece):
                continue
            self.candidates_by_first_char.setdefault(piece[0], []).append((piece, tid, len(piece)))

        for ch in self.candidates_by_first_char:
            self.candidates_by_first_char[ch].sort(key=lambda x: x[2], reverse=True)

        self._char_fallback_cache: dict[str, list[int]] = {}

    def _byte_ids_for_char(self, ch: str) -> list[int]:
        cached = self._char_fallback_cache.get(ch)
        if cached is not None:
            return cached
        out = [self.byte_piece_to_id[b] for b in ch.encode("utf-8")]
        self._char_fallback_cache[ch] = out
        return out

    def encode_dp(self, text: str) -> list[int]:
        n = len(text)
        if n == 0:
            return []

        inf = 10**9
        sentinel = (inf, inf, inf, inf, 0)

        best_score = [sentinel for _ in range(n + 1)]
        best_next: list[tuple[str, int | list[int], int] | None] = [None] * (n + 1)
        best_kind = [""] * (n + 1)

        best_score[n] = (0, 0, 0, 0, 0)
        best_kind[n] = "end"

        def boundary_penalty(ch: str) -> int:
            if ch == "\n":
                return 4
            if ch == "\t":
                return 3
            if ch == " ":
                return 3
            if ch.isspace() or ch in {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'"}:
                return 2
            return 0

        for i in range(n - 1, -1, -1):
            first = text[i]
            candidates = self.candidates_by_first_char.get(first, ())

            for piece, tid, L in candidates:
                if text.startswith(piece, i):
                    tail = best_score[i + L]
                    score = (1 + tail[0], tail[1], tail[2], tail[3], -L)
                    if score < best_score[i]:
                        best_score[i] = score
                        best_next[i] = ("piece", tid, L)
                        best_kind[i] = "piece"

            fallback_ids = self._byte_ids_for_char(first)
            tail = best_score[i + 1]
            next_is_fallback = best_kind[i + 1] == "bytes"
            new_run = 0 if next_is_fallback else 1
            score = (
                len(fallback_ids) + tail[0],
                len(fallback_ids) + tail[1],
                new_run + tail[2],
                boundary_penalty(first) + tail[3],
                -1,
            )
            if score < best_score[i]:
                best_score[i] = score
                best_next[i] = ("bytes", fallback_ids, 1)
                best_kind[i] = "bytes"

        out: list[int] = []
        i = 0
        while i < n:
            choice = best_next[i]
            if choice is None:
                raise RuntimeError(f"DP reconstruction failed at position {i}")
            kind, payload, advance = choice
            if kind == "piece":
                out.append(int(payload))
            else:
                out.extend(payload)  # type: ignore[arg-type]
            i += advance

        return out

    def count_fallback_tokens(self, ids: list[int]) -> int:
        return sum(1 for tid in ids if is_byte_piece(self.id_to_piece[tid]))

    def count_fallback_runs(self, ids: list[int]) -> int:
        runs = 0
        prev = False
        for tid in ids:
            cur = is_byte_piece(self.id_to_piece[tid])
            if cur and not prev:
                runs += 1
            prev = cur
        return runs


def iter_val_docs(docs_jsonl: Path, num_val_docs: int):
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_val_docs:
                break
            obj = json.loads(line)
            yield i, obj["text"]


def evaluate_variant(vocab_path: Path, docs_jsonl: Path, num_val_docs: int, sp) -> dict:
    tokenizer = CustomDPTokenizer(load_vocab_rows(vocab_path))
    totals = {"docs": 0, "bytes": 0, "sp_tokens": 0, "custom_tokens": 0, "custom_fallback_tokens": 0, "custom_fallback_runs": 0}

    for _, text in iter_val_docs(docs_jsonl, num_val_docs):
        byte_len = len(text.encode("utf-8"))
        sp_ids = sp.encode(text, out_type=int)
        custom_ids = tokenizer.encode_dp(text)

        totals["docs"] += 1
        totals["bytes"] += byte_len
        totals["sp_tokens"] += len(sp_ids)
        totals["custom_tokens"] += len(custom_ids)
        totals["custom_fallback_tokens"] += tokenizer.count_fallback_tokens(custom_ids)
        totals["custom_fallback_runs"] += tokenizer.count_fallback_runs(custom_ids)

    sp_tpb = totals["sp_tokens"] / totals["bytes"]
    custom_tpb = totals["custom_tokens"] / totals["bytes"]

    return {
        "variant_path": str(vocab_path),
        "variant_name": vocab_path.name,
        "docs": totals["docs"],
        "bytes": totals["bytes"],
        "sp_tokens": totals["sp_tokens"],
        "custom_tokens": totals["custom_tokens"],
        "delta_tokens": totals["custom_tokens"] - totals["sp_tokens"],
        "sp_tokens_per_byte": sp_tpb,
        "custom_tokens_per_byte": custom_tpb,
        "delta_tpb": custom_tpb - sp_tpb,
        "sp_avg_bytes_per_token": totals["bytes"] / totals["sp_tokens"],
        "custom_avg_bytes_per_token": totals["bytes"] / totals["custom_tokens"],
        "custom_fallback_tokens": totals["custom_fallback_tokens"],
        "custom_fallback_runs": totals["custom_fallback_runs"],
    }


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-jsonl", type=Path, default=Path("data/docs_selected.jsonl"))
    ap.add_argument("--num-val-docs", type=int, default=50_000)
    ap.add_argument("--sp-model", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    ap.add_argument("--variant-dir", type=Path, required=True)
    ap.add_argument("--glob", type=str, default="*.jsonl")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    ok = sp.load(str(args.sp_model))
    if not ok:
        raise FileNotFoundError(f"Could not load SP model: {args.sp_model}")

    variant_paths = sorted(args.variant_dir.glob(args.glob))
    if not variant_paths:
        raise FileNotFoundError(f"No vocab variants found in {args.variant_dir} matching {args.glob}")

    results = []
    for i, vocab_path in enumerate(variant_paths, start=1):
        print(f"[{i}/{len(variant_paths)}] Evaluating {vocab_path.name} ...", flush=True)
        try:
            result = evaluate_variant(vocab_path=vocab_path, docs_jsonl=args.docs_jsonl, num_val_docs=args.num_val_docs, sp=sp)
            results.append(result)
            print(f"  delta_tpb={result['delta_tpb']:.9f} delta_tokens={result['delta_tokens']:,} fallback={result['custom_fallback_tokens']:,}", flush=True)
        except Exception as e:
            results.append({"variant_path": str(vocab_path), "variant_name": vocab_path.name, "error": str(e)})
            print(f"  ERROR: {e}", flush=True)

    good_results = [r for r in results if "error" not in r]
    good_results.sort(key=lambda r: (r["delta_tpb"], r["delta_tokens"]))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"rank_vocab_variants_{ts}.json"
    csv_path = out_dir / f"rank_vocab_variants_{ts}.csv"

    write_json(json_path, results)

    if good_results:
        write_csv(
            csv_path,
            good_results,
            [
                "variant_name", "variant_path", "docs", "bytes", "sp_tokens", "custom_tokens", "delta_tokens",
                "sp_tokens_per_byte", "custom_tokens_per_byte", "delta_tpb", "sp_avg_bytes_per_token",
                "custom_avg_bytes_per_token", "custom_fallback_tokens", "custom_fallback_runs",
            ],
        )

    print("\n=== TOP VARIANTS ===")
    for row in good_results[:10]:
        print(f"{row['variant_name']}: delta_tpb={row['delta_tpb']:.9f} delta_tokens={row['delta_tokens']:,} fallback={row['custom_fallback_tokens']:,}")

    print(f"\nSaved JSON -> {json_path}")
    if good_results:
        print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
