from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from export_custom_dp_dataset_mp import FastCustomDPTokenizer, load_jsonl


BYTE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")
WORDISH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._:/-]{1,}")
SPACE_WORDISH_RE = re.compile(r" [A-Za-z0-9][A-Za-z0-9'._:/-]{1,}")
PUNCT_CLUSTER_RE = re.compile(r"[^\w\s]{2,}")
MULTISPACE_RE = re.compile(r" {2,}")


@dataclass
class TokenStat:
    piece: str
    token_id: int
    use_count: int
    doc_count: int
    char_len: int
    byte_len: int
    score_remove: float


@dataclass
class CandidateStat:
    piece: str
    source: str
    freq: int
    doc_count: int
    char_len: int
    byte_len: int
    fallback_hits: int
    score_add: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-generate vocab variants from a base vocab plus tokenizer gap analysis."
    )
    p.add_argument(
        "--base-vocab-jsonl",
        required=True,
        help="Path to current best vocab JSONL",
    )
    p.add_argument(
        "--score-report-json",
        required=True,
        help="Path to JSON from score_vocab_candidate.py",
    )
    p.add_argument(
        "--docs-jsonl",
        default="/workspace/parameter-golf/data/docs_selected.jsonl",
        help="Path to docs_selected.jsonl",
    )
    p.add_argument(
        "--num-docs",
        type=int,
        default=10000,
        help="How many docs to use for usage/candidate mining",
    )
    p.add_argument(
        "--variant-sizes",
        default="5,10,20",
        help="Comma-separated swap counts, e.g. 5,10,20",
    )
    p.add_argument(
        "--max-variants-per-family",
        type=int,
        default=3,
        help="How many ranked variants to emit per family/size",
    )
    p.add_argument(
        "--out-dir",
        default="/workspace/parameter-golf/vocab_variants",
        help="Directory to write generated vocab JSONLs and manifest",
    )
    p.add_argument(
        "--min-piece-len",
        type=int,
        default=2,
        help="Minimum character length for added candidates",
    )
    p.add_argument(
        "--max-piece-len",
        type=int,
        default=24,
        help="Maximum character length for added candidates",
    )
    p.add_argument(
        "--max-candidate-pool",
        type=int,
        default=1000,
        help="Max candidate additions kept after scoring",
    )
    p.add_argument(
        "--lock-top-n-ids",
        type=int,
        default=64,
        help="Do not remove ids below this threshold",
    )
    return p.parse_args()


def iter_docs(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            obj = json.loads(line)
            yield i, obj["text"]


def is_byte_piece(piece: str) -> bool:
    return bool(BYTE_RE.match(piece))


def piece_byte_len(piece: str) -> int:
    if is_byte_piece(piece):
        return 1
    return len(piece.encode("utf-8"))


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def collect_usage_stats(
    tokenizer: FastCustomDPTokenizer,
    docs_path: Path,
    num_docs: int,
) -> dict[int, TokenStat]:
    use_counter: Counter[int] = Counter()
    doc_counter: Counter[int] = Counter()

    for _, text in iter_docs(docs_path, num_docs):
        ids = tokenizer.encode_dp(text)
        use_counter.update(ids)
        doc_counter.update(set(ids))

    stats: dict[int, TokenStat] = {}
    for token_id, piece in tokenizer.id_to_piece_map.items():
        use_count = int(use_counter[token_id])
        doc_count = int(doc_counter[token_id])

        # Lower score_remove means "more removable"
        # We want to preserve tokens that are used often, appear in many docs, and are long/useful.
        char_len = len(piece)
        byte_len = piece_byte_len(piece)

        score_remove = (
            5.0 * use_count
            + 50.0 * doc_count
            + 10.0 * max(char_len - 1, 0)
            + 5.0 * max(byte_len - 1, 0)
        )

        stats[token_id] = TokenStat(
            piece=piece,
            token_id=token_id,
            use_count=use_count,
            doc_count=doc_count,
            char_len=char_len,
            byte_len=byte_len,
            score_remove=score_remove,
        )

    return stats


def load_score_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_candidate(piece: str) -> str:
    # Keep exact surface form except strip newlines/tabs.
    piece = piece.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return piece


def valid_candidate(
    piece: str,
    existing_pieces: set[str],
    min_len: int,
    max_len: int,
) -> bool:
    if not piece:
        return False
    if piece in existing_pieces:
        return False
    if is_byte_piece(piece):
        return False
    if len(piece) < min_len or len(piece) > max_len:
        return False
    if piece.strip() == "":
        return False
    if "\x00" in piece:
        return False
    return True


def extract_candidates_from_report(
    report: dict,
    existing_pieces: set[str],
    min_len: int,
    max_len: int,
) -> list[CandidateStat]:
    pool: dict[str, CandidateStat] = {}

    def add(piece: str, source: str, count_boost: int, doc_ids: Iterable[int] = ()) -> None:
        piece = normalize_candidate(piece)
        if not valid_candidate(piece, existing_pieces, min_len, max_len):
            return

        if piece not in pool:
            pool[piece] = CandidateStat(
                piece=piece,
                source=source,
                freq=0,
                doc_count=0,
                char_len=len(piece),
                byte_len=len(piece.encode("utf-8")),
                fallback_hits=0,
                score_add=0.0,
            )
        stat = pool[piece]
        stat.freq += count_boost
        stat.doc_count += len(set(doc_ids))
        if source.startswith("fallback"):
            stat.fallback_hits += count_boost

    for row in report.get("top_fallback_spans", []):
        add(
            row["span"],
            "fallback_span",
            int(row["count"]),
            row.get("sample_doc_ids", []),
        )

    for row in report.get("top_docs_by_delta", []):
        # Use word-ish spans from worst docs
        preview = row.get("preview", "")
        doc_id = row.get("doc_id", -1)

        for m in WORDISH_RE.finditer(preview):
            add(m.group(0), "worst_doc_wordish", max(1, row.get("delta", 1)), [doc_id])

        for m in SPACE_WORDISH_RE.finditer(preview):
            add(m.group(0), "worst_doc_leading_space", max(1, row.get("delta", 1)), [doc_id])

        for m in PUNCT_CLUSTER_RE.finditer(preview):
            add(m.group(0), "worst_doc_punct", max(1, row.get("delta", 1)), [doc_id])

        for m in MULTISPACE_RE.finditer(preview):
            add(m.group(0), "worst_doc_multispace", 1, [doc_id])

    for row in report.get("top_docs_by_fallback", []):
        preview = row.get("preview", "")
        doc_id = row.get("doc_id", -1)

        for m in WORDISH_RE.finditer(preview):
            add(m.group(0), "fallback_doc_wordish", max(1, row.get("fallback_tokens", 1)), [doc_id])

        for m in SPACE_WORDISH_RE.finditer(preview):
            add(m.group(0), "fallback_doc_leading_space", max(1, row.get("fallback_tokens", 1)), [doc_id])

        for m in PUNCT_CLUSTER_RE.finditer(preview):
            add(m.group(0), "fallback_doc_punct", max(1, row.get("fallback_tokens", 1)), [doc_id])

    # Final scoring
    out: list[CandidateStat] = []
    for stat in pool.values():
        stat.score_add = (
            8.0 * stat.freq
            + 40.0 * stat.doc_count
            + 20.0 * stat.fallback_hits
            + 4.0 * max(stat.char_len - 1, 0)
            + 2.0 * max(stat.byte_len - 1, 0)
        )
        out.append(stat)

    out.sort(key=lambda x: (x.score_add, x.fallback_hits, x.doc_count, x.freq), reverse=True)
    return out


def mine_candidates_from_docs(
    docs_path: Path,
    num_docs: int,
    existing_pieces: set[str],
    min_len: int,
    max_len: int,
) -> list[CandidateStat]:
    freq_counter: Counter[str] = Counter()
    doc_counter: Counter[str] = Counter()
    source_counter: dict[str, str] = {}

    regexes = [
        ("doc_wordish", WORDISH_RE),
        ("doc_leading_space", SPACE_WORDISH_RE),
        ("doc_punct", PUNCT_CLUSTER_RE),
    ]

    for _, text in iter_docs(docs_path, num_docs):
        seen: set[str] = set()
        for source, rx in regexes:
            for m in rx.finditer(text):
                piece = normalize_candidate(m.group(0))
                if not valid_candidate(piece, existing_pieces, min_len, max_len):
                    continue
                freq_counter[piece] += 1
                source_counter[piece] = source
                seen.add(piece)
        for piece in seen:
            doc_counter[piece] += 1

    out: list[CandidateStat] = []
    for piece, freq in freq_counter.items():
        stat = CandidateStat(
            piece=piece,
            source=source_counter[piece],
            freq=int(freq),
            doc_count=int(doc_counter[piece]),
            char_len=len(piece),
            byte_len=len(piece.encode("utf-8")),
            fallback_hits=0,
            score_add=0.0,
        )
        stat.score_add = (
            6.0 * stat.freq
            + 30.0 * stat.doc_count
            + 3.0 * max(stat.char_len - 1, 0)
            + 2.0 * max(stat.byte_len - 1, 0)
        )
        out.append(stat)

    out.sort(key=lambda x: (x.score_add, x.doc_count, x.freq), reverse=True)
    return out


def build_remove_lists(
    usage_stats: dict[int, TokenStat],
    lock_top_n_ids: int,
) -> dict[str, list[TokenStat]]:
    removable: list[TokenStat] = []
    for token_id, stat in usage_stats.items():
        piece = stat.piece
        if token_id < lock_top_n_ids:
            continue
        if is_byte_piece(piece):
            continue
        removable.append(stat)

    # weaker = smaller score_remove
    removable.sort(key=lambda x: (x.score_remove, x.use_count, x.doc_count, x.char_len))

    # Families
    low_usage = sorted(removable, key=lambda x: (x.use_count, x.doc_count, x.char_len))
    low_coverage = sorted(removable, key=lambda x: (x.doc_count, x.use_count, x.char_len))
    weak_overall = removable

    return {
        "low_usage": low_usage,
        "low_coverage": low_coverage,
        "weak_overall": weak_overall,
    }


def build_add_lists(
    report_candidates: list[CandidateStat],
    mined_candidates: list[CandidateStat],
    max_pool: int,
) -> dict[str, list[CandidateStat]]:
    report_candidates = report_candidates[:max_pool]
    mined_candidates = mined_candidates[:max_pool]

    fallback_heavy = [c for c in report_candidates if "fallback" in c.source]
    tail_fix = [c for c in report_candidates if "worst_doc" in c.source]
    general = mined_candidates

    # Dedup while preserving order
    def dedup(rows: list[CandidateStat]) -> list[CandidateStat]:
        seen = set()
        out = []
        for r in rows:
            if r.piece in seen:
                continue
            seen.add(r.piece)
            out.append(r)
        return out

    hybrid = dedup(fallback_heavy[: max_pool // 2] + tail_fix[: max_pool // 2] + general[: max_pool // 2])

    return {
        "fallback": dedup(fallback_heavy),
        "tail": dedup(tail_fix),
        "general": dedup(general),
        "hybrid": hybrid,
    }


def swap_vocab(
    base_rows: list[dict],
    remove_ids: list[int],
    add_pieces: list[str],
) -> list[dict]:
    remove_set = set(remove_ids)
    kept = [dict(r) for r in base_rows if int(r["id"]) not in remove_set]

    # preserve ids of removed slots, sorted
    free_ids = sorted(remove_ids)
    if len(free_ids) != len(add_pieces):
        raise ValueError("remove/add counts must match")

    for new_id, piece in zip(free_ids, add_pieces, strict=True):
        kept.append({"id": int(new_id), "piece": piece})

    kept.sort(key=lambda r: int(r["id"]))
    return kept


def main() -> None:
    args = parse_args()

    base_vocab_path = Path(args.base_vocab_jsonl)
    report_path = Path(args.score_report_json)
    docs_path = Path(args.docs_jsonl)
    out_dir = Path(args.out_dir)

    if not base_vocab_path.is_file():
        raise FileNotFoundError(base_vocab_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    if not docs_path.is_file():
        raise FileNotFoundError(docs_path)

    swap_sizes = [int(x) for x in args.variant_sizes.split(",") if x.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(base_vocab_path)
    base_rows = sorted(base_rows, key=lambda r: int(r["id"]))
    tokenizer = FastCustomDPTokenizer(base_rows)
    existing_pieces = set(tokenizer.piece_to_id_map.keys())

    print("Collecting usage stats from base vocab...")
    usage_stats = collect_usage_stats(tokenizer, docs_path, args.num_docs)
    remove_lists = build_remove_lists(usage_stats, args.lock_top_n_ids)

    print("Loading score report...")
    report = load_score_report(report_path)

    print("Extracting add candidates from gap report...")
    report_candidates = extract_candidates_from_report(
        report,
        existing_pieces,
        args.min_piece_len,
        args.max_piece_len,
    )

    print("Mining additional candidates from docs...")
    mined_candidates = mine_candidates_from_docs(
        docs_path,
        args.num_docs,
        existing_pieces,
        args.min_piece_len,
        args.max_piece_len,
    )

    add_lists = build_add_lists(
        report_candidates,
        mined_candidates,
        args.max_candidate_pool,
    )

    manifest: dict = {
        "base_vocab_jsonl": str(base_vocab_path),
        "score_report_json": str(report_path),
        "docs_jsonl": str(docs_path),
        "num_docs": args.num_docs,
        "variant_sizes": swap_sizes,
        "lock_top_n_ids": args.lock_top_n_ids,
        "generated_variants": [],
        "top_remove_lists": {
            name: [asdict(x) for x in rows[:100]]
            for name, rows in remove_lists.items()
        },
        "top_add_lists": {
            name: [asdict(x) for x in rows[:100]]
            for name, rows in add_lists.items()
        },
    }

    family_plans = [
        ("fallback", "weak_overall", "fallback"),
        ("tail", "weak_overall", "tail"),
        ("general", "low_usage", "general"),
        ("hybrid", "low_coverage", "hybrid"),
    ]

    written = 0
    base_stem = safe_filename(base_vocab_path.stem)

    for family_name, remove_key, add_key in family_plans:
        remove_pool = remove_lists[remove_key]
        add_pool = add_lists[add_key]

        if not remove_pool or not add_pool:
            continue

        for swap_size in swap_sizes:
            # emit several variants by offsetting into the ranked lists
            for variant_idx in range(args.max_variants_per_family):
                remove_start = variant_idx * swap_size
                add_start = variant_idx * swap_size

                remove_chunk = remove_pool[remove_start: remove_start + swap_size]
                add_chunk = add_pool[add_start: add_start + swap_size]

                if len(remove_chunk) < swap_size or len(add_chunk) < swap_size:
                    continue

                remove_ids = [x.token_id for x in remove_chunk]
                add_pieces = [x.piece for x in add_chunk]

                # Skip any accidental duplicate additions or collisions with kept vocab
                if len(set(add_pieces)) != len(add_pieces):
                    continue

                # Build and write variant
                variant_rows = swap_vocab(base_rows, remove_ids, add_pieces)

                name = f"{base_stem}_{family_name}_{swap_size:02d}_v{variant_idx+1}"
                out_path = out_dir / f"{name}.jsonl"

                with out_path.open("w", encoding="utf-8") as f:
                    for row in variant_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                manifest["generated_variants"].append(
                    {
                        "name": name,
                        "path": str(out_path),
                        "family": family_name,
                        "swap_size": swap_size,
                        "variant_idx": variant_idx + 1,
                        "remove_strategy": remove_key,
                        "add_strategy": add_key,
                        "removed": [asdict(x) for x in remove_chunk],
                        "added": [asdict(x) for x in add_chunk],
                    }
                )
                written += 1

    manifest_path = out_dir / f"{base_stem}_variant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Generated {written} variants")
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()