"""
summarize_logs.py
-----------------
CLI tool that reads daily JSONL log files (from logs/) over a date range
and prints aggregate observability stats.

Usage
-----
    # Today only (default)
    python src/observability/summarize_logs.py

    # Specific date
    python src/observability/summarize_logs.py --date 2026-09-11

    # Date range
    python src/observability/summarize_logs.py --from 2026-09-01 --to 2026-09-11

    # All available log files
    python src/observability/summarize_logs.py --all

Output
------
    ══════════════════════════════════════════════════
      k8s-RAG Query Observability Summary
      Period : 2026-09-11  (1 file)
    ══════════════════════════════════════════════════

    OVERVIEW
      Total queries      : 8
      Pipelines          : baseline=5, contextual=3
      Date range         : 2026-09-11 → 2026-09-11
      Log files read     : 1

    LATENCY  (ms)
    ┌─────────────────────────────┬────────┬────────┬────────┐
    │ Metric                      │   avg  │   p50  │   p95  │
    ├─────────────────────────────┼────────┼────────┼────────┤
    │ Retrieval (hybrid+RRF)      │   145  │   138  │   201  │
    │ Reranking (cross-encoder)   │    88  │    82  │   117  │
    │ Total retrieval             │   233  │   224  │   312  │
    │ Generation (LLM)            │  3012  │  2950  │  4200  │
    │ End-to-end                  │  3245  │  3180  │  4510  │
    └─────────────────────────────┴────────┴────────┴────────┘

    COST  (USD)
      Total cost         : $0.008640
      Avg cost / query   : $0.001080
      Total input tok    : 4,896
      Total output tok   : 1,184
      Avg input  / query : 612
      Avg output / query : 148

    PIPELINE BREAKDOWN
      baseline   : 5 queries | avg latency 3,190ms | avg cost $0.001050
      contextual : 3 queries | avg latency 3,320ms | avg cost $0.001130
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR     = PROJECT_ROOT / "logs"

# ── terminal width ────────────────────────────────────────────────────────────
W = 56   # table inner width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: date, end: date) -> list[date]:
    """Return every calendar date from start to end (inclusive)."""
    days = []
    cur  = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _load_events(dates: list[date]) -> list[dict[str, Any]]:
    """Read and parse all JSONL events for the given dates."""
    events: list[dict[str, Any]] = []
    for d in dates:
        path = LOGS_DIR / f"queries_{d.isoformat()}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(f"  [WARN] Skipping malformed line in {path.name}: {exc}",
                              file=sys.stderr)
    return events


def _p(values: list[float], pct: float) -> float:
    """Single-percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # quantiles() needs at least 2 data points and n >= 2
    n = max(100, len(sorted_vals))
    qs = quantiles(sorted_vals, n=n, method="inclusive")
    idx = max(0, min(int(pct / 100 * n) - 1, len(qs) - 1))
    return qs[idx]


def _fmt_ms(val: float) -> str:
    return f"{val:>7,.0f}"


def _fmt_usd(val: float) -> str:
    return f"${val:.6f}"


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(events: list[dict[str, Any]], dates: list[date], files_read: int) -> None:
    if not events:
        print("\n  [NO DATA] No log events found for the requested period.")
        return

    # ── latency fields ───────────────────────────────────────────────────────
    retr_ms  = [e["retrieval_latency_ms"]  for e in events if "retrieval_latency_ms"  in e]
    rrnk_ms  = [e["rerank_latency_ms"]     for e in events if "rerank_latency_ms"     in e]
    tretr_ms = [e["total_retrieval_ms"]    for e in events if "total_retrieval_ms"    in e]
    gen_ms   = [e["generation_latency_ms"] for e in events if "generation_latency_ms" in e]
    e2e_ms   = [e["total_latency_ms"]      for e in events if "total_latency_ms"      in e]

    # ── cost / token fields ───────────────────────────────────────────────────
    costs     = [e["cost_usd"]      for e in events if "cost_usd"      in e]
    in_toks   = [e["input_tokens"]  for e in events if "input_tokens"  in e]
    out_toks  = [e["output_tokens"] for e in events if "output_tokens" in e]

    n = len(events)

    # ── pipeline breakdown ────────────────────────────────────────────────────
    pipelines: dict[str, list[dict]] = {}
    for e in events:
        p = e.get("pipeline", "unknown")
        pipelines.setdefault(p, []).append(e)

    period_str = (
        f"{dates[0].isoformat()} → {dates[-1].isoformat()}"
        if len(dates) > 1 else dates[0].isoformat()
    )

    border = "═" * W

    print(f"\n{border}")
    print(f"  k8s-RAG Query Observability Summary")
    print(f"  Period  : {period_str}  ({files_read} file{'s' if files_read != 1 else ''})")
    print(f"{border}")

    # ── Overview ──────────────────────────────────────────────────────────────
    print(f"\n  OVERVIEW")
    print(f"    Total queries      : {n}")
    pip_str = ", ".join(f"{k}={len(v)}" for k, v in sorted(pipelines.items()))
    print(f"    Pipelines          : {pip_str}")
    print(f"    Date range         : {period_str}")
    print(f"    Log files read     : {files_read}")

    # ── Latency table ─────────────────────────────────────────────────────────
    print(f"\n  LATENCY  (ms)")
    print(f"  {'Metric':<30} {'avg':>7} {'p50':>7} {'p95':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}")

    rows = [
        ("Retrieval (hybrid+RRF)",    retr_ms),
        ("Reranking (cross-encoder)", rrnk_ms),
        ("Total retrieval",           tretr_ms),
        ("Generation (LLM)",          gen_ms),
        ("End-to-end",                e2e_ms),
    ]
    for label, vals in rows:
        if vals:
            avg = mean(vals)
            p50 = median(vals)
            p95 = _p(vals, 95)
            print(f"  {label:<30} {_fmt_ms(avg)} {_fmt_ms(p50)} {_fmt_ms(p95)}")
        else:
            print(f"  {label:<30} {'—':>7} {'—':>7} {'—':>7}")

    # ── Cost ──────────────────────────────────────────────────────────────────
    print(f"\n  COST  (USD)")
    total_cost   = sum(costs)
    avg_cost     = total_cost / n if n else 0.0
    total_in     = sum(in_toks)
    total_out    = sum(out_toks)

    print(f"    Total cost         : {_fmt_usd(total_cost)}")
    print(f"    Avg cost / query   : {_fmt_usd(avg_cost)}")
    print(f"    Total input tok    : {total_in:,}")
    print(f"    Total output tok   : {total_out:,}")
    print(f"    Avg input  / query : {total_in // n if n else 0:,}")
    print(f"    Avg output / query : {total_out // n if n else 0:,}")

    # ── Pipeline breakdown ────────────────────────────────────────────────────
    print(f"\n  PIPELINE BREAKDOWN")
    for pipe, evts in sorted(pipelines.items()):
        pipe_costs = [e["cost_usd"] for e in evts if "cost_usd" in e]
        pipe_e2e   = [e["total_latency_ms"] for e in evts if "total_latency_ms" in e]
        avg_lat    = f"{mean(pipe_e2e):,.0f}ms"   if pipe_e2e   else "—"
        avg_c      = f"${mean(pipe_costs):.6f}"   if pipe_costs else "—"
        print(f"    {pipe:<12} : {len(evts):>3} queries | avg e2e {avg_lat:>9} | avg cost {avg_c}")

    print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize k8s-RAG query observability logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Summarize a single day (default: today).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Summarize all available log files.",
    )
    parser.add_argument(
        "--from", dest="date_from", metavar="YYYY-MM-DD",
        help="Start date for a date range (inclusive).",
    )
    parser.add_argument(
        "--to", dest="date_to", metavar="YYYY-MM-DD",
        help="End date for a date range (inclusive). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.now(timezone.utc).date()

    if args.all:
        # Discover all log files
        log_files = sorted(LOGS_DIR.glob("queries_*.jsonl"))
        if not log_files:
            print(f"\n  [NO LOGS] No log files found in {LOGS_DIR}")
            sys.exit(0)
        dates = []
        for lf in log_files:
            try:
                d = date.fromisoformat(lf.stem.replace("queries_", ""))
                dates.append(d)
            except ValueError:
                pass
        files_read = len(dates)
        events     = _load_events(dates)

    elif args.date_from or args.date_to:
        start  = date.fromisoformat(args.date_from) if args.date_from else today
        end    = date.fromisoformat(args.date_to)   if args.date_to   else today
        if start > end:
            print(f"  [ERROR] --from ({start}) must be <= --to ({end})", file=sys.stderr)
            sys.exit(1)
        dates      = _date_range(start, end)
        files_read = sum(1 for d in dates if (LOGS_DIR / f"queries_{d}.jsonl").exists())
        events     = _load_events(dates)

    elif args.date:
        d          = date.fromisoformat(args.date)
        dates      = [d]
        files_read = 1 if (LOGS_DIR / f"queries_{d}.jsonl").exists() else 0
        events     = _load_events(dates)

    else:
        # Default: today
        dates      = [today]
        files_read = 1 if (LOGS_DIR / f"queries_{today}.jsonl").exists() else 0
        events     = _load_events(dates)

    print_report(events, dates, files_read)


if __name__ == "__main__":
    main()
