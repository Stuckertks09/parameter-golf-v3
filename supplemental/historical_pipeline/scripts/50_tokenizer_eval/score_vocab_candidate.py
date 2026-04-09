from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from export_custom_dp_dataset_mp import FastCustomDPTokenizer, load_jsonl


@dataclass
class DocMetrics:
    doc_id: int
    baseline_len: int
    custom_len: int
    delta: int
    ratio: float
    fallback_token_count: int
    fallback_char_count: int
    fallback_span_count: int
    fallback_token_share: float
    preview: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score a custom vocab against SP1024 using token count, fallback, and tail penalties."
    )
    p.add_argument(
        "--docs-jsonl",
        default="/workspace/parameter-golf/data/docs_selected.jsonl",
        help="Path to docs_selected.jsonl",
    )
    p.add_argument(
        "--vocab-jsonl",
        required=True,
        help="Path to candidate custom vocab JSONL",
    )
    p.add_argument(
        "--sp-model",
        default="/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
        help="Path to baseline sentencepiece model",
    )
    p.add_argument(
        "--num-docs",
        type=int,
        default=10000,
        help="Number of docs from docs_selected.jsonl to score",
    )
    p.add_argument(
        "--top-docs",
        type=int,
        default=50,
        help="How many worst docs to include in the report",
    )
    p.add_argument(
        "--top-spans",
        type=int,
        default=100,
        help="How many fallback spans to include in the report",
    )
    p.add_argument(
        "--top-chars",
        type=int,
        default=100,
        help="How many fallback chars to include in the report",
    )
    p.add_argument(
        "--out-prefix",
        default="",
        help="Optional output prefix. Defaults to vocab stem in /workspace/parameter-golf/analysis_scores/",
    )
    return p.parse_args()


def iter_docs(path: Path, num_docs: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_docs:
                break
            obj = json.loads(line)
            yield i, obj["text"]


def safe_preview(text: str, max_len: int = 220) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text[:max_len]


def percentile_sorted(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def analyze_custom_encoding(tokenizer: FastCustomDPTokenizer, text: str) -> dict[str, Any]:
    """
    Mirrors the DP path while tracking fallback usage.
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
            _piece, pid = payload  # type: ignore[misc]
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


def compute_scalar_score(
    *,
    token_ratio: float,
    fallback_token_share: float,
    mean_delta: float,
    p95_delta: float,
    p99_delta: float,
    worst100_mean_delta: float,
    improved_doc_share: float,
    worse_doc_share: float,
) -> float:
    """
    Higher is better.
    This is a ranking proxy, not a physically meaningful metric.
    """
    score = 0.0

    # Main objective: stay below baseline token count if possible.
    score -= 1000.0 * (token_ratio - 1.0)

    # Fallback is expensive and strongly predictive of bad allocations.
    score -= 3000.0 * fallback_token_share

    # Tail penalties matter a lot because ugly docs drive losses.
    score -= 50.0 * mean_delta
    score -= 100.0 * p95_delta
    score -= 150.0 * p99_delta
    score -= 200.0 * worst100_mean_delta

    # Mild reward for broadly improving more docs than you hurt.
    score += 50.0 * improved_doc_share
    score -= 50.0 * worse_doc_share

    return score


def main() -> None:
    args = parse_args()

    docs_path = Path(args.docs_jsonl)
    vocab_path = Path(args.vocab_jsonl)
    sp_model_path = Path(args.sp_model)

    if not docs_path.is_file():
        raise FileNotFoundError(f"Missing docs file: {docs_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not sp_model_path.is_file():
        raise FileNotFoundError(f"Missing sentencepiece model: {sp_model_path}")

    out_dir = Path("/workspace/parameter-golf/analysis_scores")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = args.out_prefix.strip()
    if not out_prefix:
        out_prefix = vocab_path.stem

    json_out = out_dir / f"{out_prefix}.score.json"
    txt_out = out_dir / f"{out_prefix}.score.txt"

    print("Loading baseline SentencePiece...")
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))

    print("Loading custom vocab...")
    vocab_rows = load_jsonl(vocab_path)
    custom = FastCustomDPTokenizer(vocab_rows)

    baseline_total = 0
    custom_total = 0
    fallback_token_total = 0
    fallback_char_total = 0

    doc_metrics: list[DocMetrics] = []

    fallback_span_counter: Counter[str] = Counter()
    fallback_char_counter: Counter[str] = Counter()
    fallback_span_docs: dict[str, list[int]] = defaultdict(list)

    for doc_id, text in iter_docs(docs_path, args.num_docs):
        baseline_ids = sp.encode(text)
        custom_info = analyze_custom_encoding(custom, text)

        b = len(baseline_ids)
        c = len(custom_info["ids"])
        delta = c - b
        ratio = (c / b) if b else float("inf")
        fallback_token_share = (custom_info["fallback_token_count"] / c) if c else 0.0

        baseline_total += b
        custom_total += c
        fallback_token_total += custom_info["fallback_token_count"]
        fallback_char_total += custom_info["fallback_char_count"]

        for span in custom_info["fallback_spans"]:
            fallback_span_counter[span] += 1
            if len(fallback_span_docs[span]) < 5:
                fallback_span_docs[span].append(doc_id)

        for ch in custom_info["fallback_chars"]:
            fallback_char_counter[ch] += 1

        doc_metrics.append(
            DocMetrics(
                doc_id=doc_id,
                baseline_len=b,
                custom_len=c,
                delta=delta,
                ratio=ratio,
                fallback_token_count=custom_info["fallback_token_count"],
                fallback_char_count=custom_info["fallback_char_count"],
                fallback_span_count=len(custom_info["fallback_spans"]),
                fallback_token_share=fallback_token_share,
                preview=safe_preview(text),
            )
        )

        if (doc_id + 1) % 1000 == 0:
            token_ratio = custom_total / baseline_total if baseline_total else float("nan")
            fb_share = fallback_token_total / custom_total if custom_total else 0.0
            print(
                f"docs={doc_id+1:,} "
                f"baseline={baseline_total:,} "
                f"custom={custom_total:,} "
                f"ratio={token_ratio:.6f} "
                f"fallback_tokens={fallback_token_total:,} "
                f"fallback_share={fb_share:.6f}",
                flush=True,
            )

    if not doc_metrics:
        raise ValueError("No docs were analyzed")

    deltas = [float(d.delta) for d in doc_metrics]
    deltas_sorted = sorted(deltas)

    improved_count = sum(1 for d in doc_metrics if d.delta < 0)
    worse_count = sum(1 for d in doc_metrics if d.delta > 0)
    same_count = len(doc_metrics) - improved_count - worse_count

    token_ratio = custom_total / baseline_total if baseline_total else float("inf")
    fallback_token_share = fallback_token_total / custom_total if custom_total else 0.0

    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)
    p95_delta = percentile_sorted(deltas_sorted, 0.95)
    p99_delta = percentile_sorted(deltas_sorted, 0.99)
    worst100 = sorted(deltas, reverse=True)[: min(100, len(deltas))]
    worst100_mean_delta = statistics.fmean(worst100) if worst100 else 0.0

    improved_doc_share = improved_count / len(doc_metrics)
    worse_doc_share = worse_count / len(doc_metrics)

    scalar_score = compute_scalar_score(
        token_ratio=token_ratio,
        fallback_token_share=fallback_token_share,
        mean_delta=mean_delta,
        p95_delta=p95_delta,
        p99_delta=p99_delta,
        worst100_mean_delta=worst100_mean_delta,
        improved_doc_share=improved_doc_share,
        worse_doc_share=worse_doc_share,
    )

    by_delta = sorted(doc_metrics, key=lambda x: (x.delta, x.fallback_token_count), reverse=True)
    by_fallback = sorted(doc_metrics, key=lambda x: (x.fallback_token_count, x.delta), reverse=True)

    result = {
        "config": {
            "docs_jsonl": str(docs_path),
            "vocab_jsonl": str(vocab_path),
            "sp_model": str(sp_model_path),
            "num_docs": args.num_docs,
            "top_docs": args.top_docs,
            "top_spans": args.top_spans,
            "top_chars": args.top_chars,
        },
        "summary": {
            "docs_analyzed": len(doc_metrics),
            "baseline_total_tokens": baseline_total,
            "custom_total_tokens": custom_total,
            "token_ratio_custom_over_baseline": token_ratio,
            "token_delta_total": custom_total - baseline_total,
            "fallback_token_total": fallback_token_total,
            "fallback_char_total": fallback_char_total,
            "fallback_token_share_of_custom": fallback_token_share,
            "improved_doc_count": improved_count,
            "worse_doc_count": worse_count,
            "same_doc_count": same_count,
            "improved_doc_share": improved_doc_share,
            "worse_doc_share": worse_doc_share,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "p95_delta": p95_delta,
            "p99_delta": p99_delta,
            "worst100_mean_delta": worst100_mean_delta,
            "scalar_score_higher_is_better": scalar_score,
        },
        "top_docs_by_delta": [asdict(x) for x in by_delta[: args.top_docs]],
        "top_docs_by_fallback": [asdict(x) for x in by_fallback[: args.top_docs]],
        "top_fallback_spans": [
            {
                "count": count,
                "span": span,
                "sample_doc_ids": fallback_span_docs[span],
            }
            for span, count in fallback_span_counter.most_common(args.top_spans)
        ],
        "top_fallback_chars": [
            {
                "count": count,
                "char": ch,
                "ord": ord(ch),
            }
            for ch, count in fallback_char_counter.most_common(args.top_chars)
        ],
    }

    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with txt_out.open("w", encoding="utf-8") as out:
        s = result["summary"]
        out.write("=== SUMMARY ===\n")
        for k, v in s.items():
            out.write(f"{k}: {v}\n")

        out.write("\n=== TOP DOCS BY DELTA ===\n")
        for row in result["top_docs_by_delta"]:
            out.write(
                f"\ndoc={row['doc_id']} delta={row['delta']} baseline={row['baseline_len']} "
                f"custom={row['custom_len']} ratio={row['ratio']:.6f} "
                f"fallback_tokens={row['fallback_token_count']} "
                f"fallback_chars={row['fallback_char_count']} "
                f"fallback_spans={row['fallback_span_count']} "
                f"fallback_token_share={row['fallback_token_share']:.6f}\n"
            )
            out.write(f"preview={row['preview']}\n")

        out.write("\n=== TOP DOCS BY FALLBACK TOKEN COUNT ===\n")
        for row in result["top_docs_by_fallback"]:
            out.write(
                f"\ndoc={row['doc_id']} fallback_tokens={row['fallback_token_count']} "
                f"delta={row['delta']} baseline={row['baseline_len']} custom={row['custom_len']} "
                f"fallback_chars={row['fallback_char_count']} "
                f"fallback_spans={row['fallback_span_count']} "
                f"fallback_token_share={row['fallback_token_share']:.6f}\n"
            )
            out.write(f"preview={row['preview']}\n")

        out.write("\n=== TOP FALLBACK SPANS ===\n")
        for row in result["top_fallback_spans"]:
            out.write(
                f"count={row['count']} span={repr(row['span'])} sample_doc_ids={row['sample_doc_ids']}\n"
            )

        out.write("\n=== TOP FALLBACK CHARS ===\n")
        for row in result["top_fallback_chars"]:
            out.write(
                f"count={row['count']} char={repr(row['char'])} ord={row['ord']}\n"
            )

    print("\n=== FINAL ===")
    print(f"docs_analyzed:           {len(doc_metrics)}")
    print(f"baseline_total_tokens:   {baseline_total}")
    print(f"custom_total_tokens:     {custom_total}")
    print(f"token_ratio:             {token_ratio:.9f}")
    print(f"fallback_token_share:    {fallback_token_share:.9f}")
    print(f"mean_delta:              {mean_delta:.6f}")
    print(f"p95_delta:               {p95_delta:.6f}")
    print(f"p99_delta:               {p99_delta:.6f}")
    print(f"worst100_mean_delta:     {worst100_mean_delta:.6f}")
    print(f"scalar_score:            {scalar_score:.6f}")
    print(f"json_report:             {json_out}")
    print(f"text_report:             {txt_out}")


if __name__ == "__main__":
    main()