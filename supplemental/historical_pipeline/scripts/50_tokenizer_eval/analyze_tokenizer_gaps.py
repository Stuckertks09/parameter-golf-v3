from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from export_custom_dp_dataset_mp import FastCustomDPTokenizer, load_jsonl


DOCS_PATH = Path("/workspace/parameter-golf/data/docs_selected.jsonl")
VOCAB_PATH = Path("/workspace/parameter-golf/vocab/vocab_2.jsonl")
SP_MODEL_PATH = Path("/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")

NUM_DOCS = 10000
TOP_DOCS = 50
TOP_SPANS = 100
TOP_CHARS = 100


@dataclass
class DocResult:
    doc_id: int
    baseline_len: int
    custom_len: int
    delta: int
    fallback_token_count: int
    fallback_char_count: int
    fallback_span_count: int
    fallback_ratio_tokens: float
    preview: str


def iter_docs(path: Path, num_docs: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_docs:
                break
            obj = json.loads(line)
            yield i, obj["text"]


def analyze_custom_encoding(tokenizer: FastCustomDPTokenizer, text: str) -> dict:
    """
    Re-run DP, but keep track of which positions fell back to byte tokens.
    This mirrors the current custom tokenizer logic closely enough for analysis.
    """
    n = len(text)
    dp_cost = [math.inf] * (n + 1)
    dp_next: list[tuple[str, object, int] | None] = [None] * (n + 1)
    dp_cost[n] = 0

    for i in range(n - 1, -1, -1):
        best_cost = math.inf
        best_choice: tuple[str, object, int] | None = None

        first_char = text[i]
        candidates = tokenizer.candidates_by_first_char.get(first_char, ())
        if candidates:
            remaining = n - i
            for piece, pid, L in candidates:
                if L > remaining:
                    continue
                if text.startswith(piece, i):
                    cand_cost = 1 + dp_cost[i + L]
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_choice = ("piece", (piece, pid), L)
                    elif cand_cost == best_cost and best_choice is not None:
                        if L > best_choice[2]:
                            best_choice = ("piece", (piece, pid), L)

        fallback_ids = tokenizer._byte_fallback_ids_for_char(first_char)
        fallback_cost = len(fallback_ids) + dp_cost[i + 1]
        if fallback_cost < best_cost:
            best_cost = fallback_cost
            best_choice = ("bytes", fallback_ids, 1)
        elif fallback_cost == best_cost and best_choice is None:
            best_choice = ("bytes", fallback_ids, 1)

        dp_cost[i] = best_cost
        dp_next[i] = best_choice

    ids: list[int] = []
    fallback_token_count = 0
    fallback_char_count = 0
    fallback_spans: list[str] = []
    fallback_chars: list[str] = []

    i = 0
    current_span_chars: list[str] = []

    while i < n:
        choice = dp_next[i]
        if choice is None:
            raise RuntimeError(f"DP reconstruction failed at position {i}")

        kind, payload, advance = choice

        if kind == "piece":
            if current_span_chars:
                fallback_spans.append("".join(current_span_chars))
                current_span_chars = []
            piece, pid = payload  # type: ignore[misc]
            ids.append(int(pid))
        else:
            ch = text[i]
            byte_ids = payload  # type: ignore[assignment]
            ids.extend(byte_ids)
            fallback_token_count += len(byte_ids)
            fallback_char_count += 1
            fallback_chars.append(ch)
            current_span_chars.append(ch)

        i += advance

    if current_span_chars:
        fallback_spans.append("".join(current_span_chars))

    return {
        "ids": ids,
        "fallback_token_count": fallback_token_count,
        "fallback_char_count": fallback_char_count,
        "fallback_spans": fallback_spans,
        "fallback_chars": fallback_chars,
    }


def safe_preview(text: str, max_len: int = 220) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text[:max_len]


def main() -> None:
    if not DOCS_PATH.is_file():
        raise FileNotFoundError(f"Missing docs file: {DOCS_PATH}")
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(f"Missing vocab file: {VOCAB_PATH}")
    if not SP_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing sentencepiece model: {SP_MODEL_PATH}")

    print("Loading baseline SentencePiece...")
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))

    print("Loading custom tokenizer vocab...")
    vocab_rows = load_jsonl(VOCAB_PATH)
    custom = FastCustomDPTokenizer(vocab_rows)

    baseline_total = 0
    custom_total = 0
    fallback_token_total = 0
    fallback_char_total = 0

    results: list[DocResult] = []
    span_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    span_docs: dict[str, list[int]] = defaultdict(list)

    print(f"Analyzing first {NUM_DOCS:,} docs...")
    for doc_id, text in iter_docs(DOCS_PATH, NUM_DOCS):
        baseline_ids = sp.encode(text)
        custom_info = analyze_custom_encoding(custom, text)

        baseline_len = len(baseline_ids)
        custom_len = len(custom_info["ids"])
        delta = custom_len - baseline_len

        baseline_total += baseline_len
        custom_total += custom_len
        fallback_token_total += custom_info["fallback_token_count"]
        fallback_char_total += custom_info["fallback_char_count"]

        for span in custom_info["fallback_spans"]:
            span_counter[span] += 1
            if len(span_docs[span]) < 5:
                span_docs[span].append(doc_id)

        for ch in custom_info["fallback_chars"]:
            char_counter[ch] += 1

        fallback_ratio_tokens = (
            custom_info["fallback_token_count"] / custom_len if custom_len > 0 else 0.0
        )

        results.append(
            DocResult(
                doc_id=doc_id,
                baseline_len=baseline_len,
                custom_len=custom_len,
                delta=delta,
                fallback_token_count=custom_info["fallback_token_count"],
                fallback_char_count=custom_info["fallback_char_count"],
                fallback_span_count=len(custom_info["fallback_spans"]),
                fallback_ratio_tokens=fallback_ratio_tokens,
                preview=safe_preview(text),
            )
        )

        if (doc_id + 1) % 1000 == 0:
            ratio = custom_total / baseline_total if baseline_total else float("nan")
            fb_ratio = fallback_token_total / custom_total if custom_total else 0.0
            print(
                f"docs={doc_id+1} "
                f"baseline={baseline_total} "
                f"custom={custom_total} "
                f"ratio={ratio:.6f} "
                f"fallback_tokens={fallback_token_total} "
                f"fallback_share={fb_ratio:.6f}",
                flush=True,
            )

    results_by_delta = sorted(results, key=lambda r: (r.delta, r.fallback_token_count), reverse=True)
    results_by_fallback = sorted(results, key=lambda r: (r.fallback_token_count, r.delta), reverse=True)

    final_ratio = custom_total / baseline_total if baseline_total else float("nan")
    final_fb_ratio = fallback_token_total / custom_total if custom_total else 0.0

    out_path = Path("/workspace/parameter-golf/tokenizer_gap_report.txt")
    with out_path.open("w", encoding="utf-8") as out:
        out.write("=== SUMMARY ===\n")
        out.write(f"docs_analyzed: {NUM_DOCS}\n")
        out.write(f"baseline_total_tokens: {baseline_total}\n")
        out.write(f"custom_total_tokens: {custom_total}\n")
        out.write(f"custom_over_baseline_ratio: {final_ratio:.9f}\n")
        out.write(f"token_delta_total: {custom_total - baseline_total}\n")
        out.write(f"fallback_token_total: {fallback_token_total}\n")
        out.write(f"fallback_char_total: {fallback_char_total}\n")
        out.write(f"fallback_token_share_of_custom: {final_fb_ratio:.9f}\n\n")

        out.write("=== TOP DOCS BY TOKEN DELTA (custom - baseline) ===\n")
        for r in results_by_delta[:TOP_DOCS]:
            out.write(
                f"\ndoc={r.doc_id} delta={r.delta} baseline={r.baseline_len} custom={r.custom_len} "
                f"fallback_tokens={r.fallback_token_count} fallback_chars={r.fallback_char_count} "
                f"fallback_spans={r.fallback_span_count} fallback_ratio_tokens={r.fallback_ratio_tokens:.6f}\n"
            )
            out.write(f"preview={r.preview}\n")

        out.write("\n=== TOP DOCS BY FALLBACK TOKEN COUNT ===\n")
        for r in results_by_fallback[:TOP_DOCS]:
            out.write(
                f"\ndoc={r.doc_id} fallback_tokens={r.fallback_token_count} delta={r.delta} "
                f"baseline={r.baseline_len} custom={r.custom_len} "
                f"fallback_chars={r.fallback_char_count} fallback_spans={r.fallback_span_count} "
                f"fallback_ratio_tokens={r.fallback_ratio_tokens:.6f}\n"
            )
            out.write(f"preview={r.preview}\n")

        out.write("\n=== TOP FALLBACK SPANS ===\n")
        for span, count in span_counter.most_common(TOP_SPANS):
            out.write(
                f"count={count} span={repr(span)} sample_docs={span_docs[span]}\n"
            )

        out.write("\n=== TOP FALLBACK CHARS ===\n")
        for ch, count in char_counter.most_common(TOP_CHARS):
            out.write(f"count={count} char={repr(ch)} ord={ord(ch)}\n")

    print("\n=== FINAL ===")
    print(f"baseline_total_tokens: {baseline_total}")
    print(f"custom_total_tokens:   {custom_total}")
    print(f"ratio:                 {final_ratio:.9f}")
    print(f"token_delta_total:     {custom_total - baseline_total}")
    print(f"fallback_token_total:  {fallback_token_total}")
    print(f"fallback_char_total:   {fallback_char_total}")
    print(f"fallback_token_share:  {final_fb_ratio:.9f}")
    print(f"report_written_to:     {out_path}")


if __name__ == "__main__":
    main()