#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sentencepiece as spm
except ImportError as exc:
    raise SystemExit("Install sentencepiece: pip install sentencepiece") from exc


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class SampleMetrics:
    decoded_chars: int
    decoded_words: int
    unique_words: int
    unique_word_ratio: float
    ascii_ratio: float
    whitespace_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    non_printable_ratio: float
    url_hits: int
    html_hits: int
    repeated_char_hits: int
    line_count: int
    avg_line_length: float


@dataclass
class ShardSummary:
    shard_name: str
    split: str
    path: str
    file_size_bytes: int
    header_bytes: int
    token_dtype: str
    token_count: int
    min_token: int
    max_token: int
    mean_token: float
    std_token: float
    unique_token_count: int
    unique_token_ratio: float
    top_tokens: list[list[int | float]]
    sample_window_size: int
    sample_count: int
    aggregated_metrics: dict[str, Any]
    sample_previews: list[str]


class ShardAnalyzer:
    def __init__(
        self,
        dataset_dir: Path,
        tokenizer_path: Path,
        output_json: Path,
        sample_count: int = 32,
        sample_window_size: int = 256,
        preview_chars: int = 240,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.tokenizer_path = tokenizer_path
        self.output_json = output_json
        self.sample_count = sample_count
        self.sample_window_size = sample_window_size
        self.preview_chars = preview_chars

        self.sp = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))

    def analyze(self) -> dict[str, Any]:
        shard_paths = sorted(self.dataset_dir.glob("fineweb_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found under {self.dataset_dir}")

        summaries: list[ShardSummary] = []
        for shard_path in shard_paths:
            summaries.append(self._analyze_shard(shard_path))

        split_counts = Counter(summary.split for summary in summaries)
        report = {
            "dataset_dir": str(self.dataset_dir.resolve()),
            "tokenizer_path": str(self.tokenizer_path.resolve()),
            "shard_count": len(summaries),
            "split_counts": dict(split_counts),
            "sample_count": self.sample_count,
            "sample_window_size": self.sample_window_size,
            "summaries": [asdict(s) for s in summaries],
        }

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _analyze_shard(self, shard_path: Path) -> ShardSummary:
        token_array, dtype_name, header_bytes = self._load_token_array(shard_path)
        token_count = int(token_array.size)
        split = "train" if "train" in shard_path.name else "val"
        file_size_bytes = shard_path.stat().st_size

        bincount = np.bincount(token_array, minlength=int(token_array.max()) + 1)
        nonzero_ids = np.flatnonzero(bincount)
        counts = bincount[nonzero_ids]
        sort_idx = np.argsort(counts)[::-1][:20]
        top_tokens = [[int(nonzero_ids[i]), float(counts[i] / token_count)] for i in sort_idx]

        sample_texts = self._sample_and_decode(token_array)
        metric_objects = [self._measure_text(text) for text in sample_texts]
        aggregated_metrics = self._aggregate_metrics(metric_objects)
        previews = [self._preview_text(text) for text in sample_texts]

        return ShardSummary(
            shard_name=shard_path.name,
            split=split,
            path=str(shard_path.resolve()),
            file_size_bytes=file_size_bytes,
            header_bytes=header_bytes,
            token_dtype=dtype_name,
            token_count=token_count,
            min_token=int(token_array.min()),
            max_token=int(token_array.max()),
            mean_token=float(token_array.mean()),
            std_token=float(token_array.std()),
            unique_token_count=int(nonzero_ids.size),
            unique_token_ratio=float(nonzero_ids.size / max(token_count, 1)),
            top_tokens=top_tokens,
            sample_window_size=self.sample_window_size,
            sample_count=len(sample_texts),
            aggregated_metrics=aggregated_metrics,
            sample_previews=previews,
        )

    def _load_token_array(self, shard_path: Path) -> tuple[np.ndarray, str, int]:
        header_ints = np.fromfile(shard_path, dtype="<i4", count=256)
        if header_ints.size == 256 and int(header_ints[0]) == 20240520 and int(header_ints[1]) == 1:
            header_bytes = 256 * np.dtype("<i4").itemsize
            num_tokens = int(header_ints[2])
            arr = np.fromfile(shard_path, dtype="<u2", count=num_tokens, offset=header_bytes)
            return arr, "uint16", header_bytes

        # fallback
        file_size = shard_path.stat().st_size
        if file_size % 2 == 0:
            arr16 = np.memmap(shard_path, dtype=np.uint16, mode="r")
            return np.asarray(arr16), "uint16", 0
        raise ValueError(f"Could not parse shard: {shard_path}")

    def _sample_and_decode(self, token_array: np.ndarray) -> list[str]:
        token_count = int(token_array.size)
        window = min(self.sample_window_size, token_count)
        if token_count <= window:
            return [self._decode_ids(token_array.tolist())]

        starts = np.linspace(0, token_count - window, num=self.sample_count, dtype=int)
        decoded = []
        for start in starts:
            ids = token_array[start:start + window].tolist()
            decoded.append(self._decode_ids(ids))
        return decoded

    def _decode_ids(self, ids: list[int]) -> str:
        try:
            return self.sp.decode_ids([int(x) for x in ids])
        except Exception:
            return ""

    def _measure_text(self, text: str) -> SampleMetrics:
        if not text:
            return SampleMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0)

        chars = len(text)
        words = WORD_RE.findall(text)
        unique_words = len(set(word.lower() for word in words))
        whitespace_count = sum(ch.isspace() for ch in text)
        digit_count = sum(ch.isdigit() for ch in text)
        ascii_count = sum(ord(ch) < 128 for ch in text)
        non_printable = sum((not ch.isprintable()) and (not ch.isspace()) for ch in text)
        punctuation_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
        lines = [line for line in text.splitlines() if line.strip()]
        avg_line_length = float(sum(len(line) for line in lines) / len(lines)) if lines else 0.0

        return SampleMetrics(
            decoded_chars=chars,
            decoded_words=len(words),
            unique_words=unique_words,
            unique_word_ratio=(unique_words / len(words)) if words else 0.0,
            ascii_ratio=ascii_count / chars,
            whitespace_ratio=whitespace_count / chars,
            digit_ratio=digit_count / chars,
            punctuation_ratio=punctuation_count / chars,
            non_printable_ratio=non_printable / chars,
            url_hits=len(URL_RE.findall(text)),
            html_hits=len(HTML_RE.findall(text)),
            repeated_char_hits=len(REPEATED_CHAR_RE.findall(text)),
            line_count=len(lines),
            avg_line_length=avg_line_length,
        )

    def _aggregate_metrics(self, metrics: list[SampleMetrics]) -> dict[str, Any]:
        def mean_attr(name: str) -> float:
            vals = [float(getattr(m, name)) for m in metrics]
            return float(sum(vals) / len(vals)) if vals else 0.0

        fields = [
            "decoded_chars",
            "decoded_words",
            "unique_words",
            "unique_word_ratio",
            "ascii_ratio",
            "whitespace_ratio",
            "digit_ratio",
            "punctuation_ratio",
            "non_printable_ratio",
            "url_hits",
            "html_hits",
            "repeated_char_hits",
            "line_count",
            "avg_line_length",
        ]
        out = {f"mean_{field}": mean_attr(field) for field in fields}
        out["sample_count"] = len(metrics)
        return out

    def _preview_text(self, text: str) -> str:
        clean = WHITESPACE_RE.sub(" ", text).strip()
        return clean[:self.preview_chars]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--output-json", type=Path, default=Path("data/analysis/fineweb10B_sp1024/shard_report.json"))
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-window-size", type=int, default=256)
    parser.add_argument("--preview-chars", type=int, default=240)
    args = parser.parse_args()

    analyzer = ShardAnalyzer(
        dataset_dir=args.dataset_dir,
        tokenizer_path=args.tokenizer_path,
        output_json=args.output_json,
        sample_count=args.sample_count,
        sample_window_size=args.sample_window_size,
        preview_chars=args.preview_chars,
    )
    report = analyzer.analyze()
    print(json.dumps({
        "status": "ok",
        "dataset_dir": report["dataset_dir"],
        "tokenizer_path": report["tokenizer_path"],
        "shard_count": report["shard_count"],
        "split_counts": report["split_counts"],
        "output_json": str(args.output_json.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()