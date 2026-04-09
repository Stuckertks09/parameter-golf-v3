from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
STOP_RE = re.compile(
    r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iters>\d+)"
)
ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
ROUNDTRIP_FAST_RE = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
STEP_AVG_RE = re.compile(r"step_avg:(?P<step_avg_ms>[0-9.]+)ms")


@dataclass
class RunSummary:
    name: str
    path: str
    final_step: int | None = None
    final_step_val_loss: float | None = None
    final_step_val_bpb: float | None = None
    stop_step: int | None = None
    train_time_ms: int | None = None
    roundtrip_val_loss: float | None = None
    roundtrip_val_bpb: float | None = None
    last_seen_step_avg_ms: float | None = None


def parse_log(path: Path) -> RunSummary:
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = RunSummary(name=path.stem, path=str(path))

    for line in text.splitlines():
        m = STEP_VAL_RE.search(line)
        if m:
            summary.final_step = int(m.group("step"))
            summary.final_step_val_loss = float(m.group("val_loss"))
            summary.final_step_val_bpb = float(m.group("val_bpb"))

        m = STOP_RE.search(line)
        if m:
            summary.stop_step = int(m.group("step"))
            summary.train_time_ms = int(m.group("train_time_ms"))

        m = ROUNDTRIP_RE.search(line)
        if m:
            summary.roundtrip_val_loss = float(m.group("val_loss"))
            summary.roundtrip_val_bpb = float(m.group("val_bpb"))

        if summary.roundtrip_val_bpb is None:
            m = ROUNDTRIP_FAST_RE.search(line)
            if m:
                summary.roundtrip_val_loss = float(m.group("val_loss"))
                summary.roundtrip_val_bpb = float(m.group("val_bpb"))

        m = STEP_AVG_RE.search(line)
        if m:
            summary.last_seen_step_avg_ms = float(m.group("step_avg_ms"))

    return summary


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, int):
        return f"{x}"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare training log summaries.")
    ap.add_argument("logs", nargs="+", help="Paths to log files")
    args = ap.parse_args()

    runs = [parse_log(Path(p)) for p in args.logs]

    # sort by final step val_bpb ascending if present, else inf
    runs.sort(key=lambda r: (float("inf") if r.final_step_val_bpb is None else r.final_step_val_bpb))

    print(
        f"{'rank':>4}  {'name':<32}  {'step_val_bpb':>12}  {'rt_bpb':>12}  "
        f"{'stop_step':>10}  {'train_ms':>10}  {'step_avg_ms':>12}"
    )
    print("-" * 106)

    for i, r in enumerate(runs, 1):
        print(
            f"{i:>4}  "
            f"{r.name[:32]:<32}  "
            f"{fmt(r.final_step_val_bpb):>12}  "
            f"{fmt(r.roundtrip_val_bpb):>12}  "
            f"{fmt(r.stop_step):>10}  "
            f"{fmt(r.train_time_ms):>10}  "
            f"{fmt(r.last_seen_step_avg_ms, 2):>12}"
        )

    print("\nDetailed:\n")
    for r in runs:
        print(f"{r.name}")
        print(f"  path:                 {r.path}")
        print(f"  final_step:           {fmt(r.final_step)}")
        print(f"  final_step_val_loss:  {fmt(r.final_step_val_loss)}")
        print(f"  final_step_val_bpb:   {fmt(r.final_step_val_bpb)}")
        print(f"  stop_step:            {fmt(r.stop_step)}")
        print(f"  train_time_ms:        {fmt(r.train_time_ms)}")
        print(f"  roundtrip_val_loss:   {fmt(r.roundtrip_val_loss)}")
        print(f"  roundtrip_val_bpb:    {fmt(r.roundtrip_val_bpb)}")
        print(f"  last_step_avg_ms:     {fmt(r.last_seen_step_avg_ms, 2)}")
        print()

    if len(runs) >= 2 and runs[0].final_step_val_bpb is not None:
        best = runs[0]
        print("Deltas vs best final_step_val_bpb:\n")
        for r in runs[1:]:
            if r.final_step_val_bpb is None:
                continue
            delta = r.final_step_val_bpb - best.final_step_val_bpb
            print(f"{r.name}: +{delta:.6f} bpb vs {best.name}")


if __name__ == "__main__":
    main()