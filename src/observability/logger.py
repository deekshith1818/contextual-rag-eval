"""
logger.py
---------
Structured per-query observability logger for the k8s-rag pipeline.

Records one JSONL event per query to:
    logs/queries_YYYY-MM-DD.jsonl   (appended, never overwritten)

Event schema
------------
{
    "timestamp":              "2026-09-11T17:30:00.123456+00:00",  # ISO-8601 UTC
    "query":                  "how do I mount a ConfigMap as a file?",
    "pipeline":               "baseline" | "contextual",
    "retrieval_latency_ms":   142.3,   # hybrid-search + RRF fusion
    "rerank_latency_ms":      89.1,    # cross-encoder reranking only
    "total_retrieval_ms":     231.4,   # retrieval_latency_ms + rerank_latency_ms
    "generation_latency_ms":  2984.7,  # LLM call wall time
    "total_latency_ms":       3216.1,  # end-to-end
    "generation_model":       "claude-haiku-4-5-20251001",
    "input_tokens":           612,
    "output_tokens":          148,
    "cost_usd":               0.001083,
    "num_chunks_retrieved":   5,
    "retrieved_chunk_ids":    ["docs__...#0003", ...]
}

Pricing (claude-haiku-4-5-20251001, as of Sep 2026 — source: anthropic.com/pricing)
    Input  : $0.80  / 1M tokens
    Output : $4.00  / 1M tokens
    Cache write : $1.00  / 1M tokens
    Cache read  : $0.08  / 1M tokens

Usage
-----
    from src.observability.logger import log_query_event

    log_query_event(
        query="...",
        pipeline="baseline",
        retrieval_latency_ms=142.3,
        rerank_latency_ms=89.1,
        generation_latency_ms=2984.7,
        generation_model="claude-haiku-4-5-20251001",
        input_tokens=612,
        output_tokens=148,
        retrieved_chunk_ids=["docs__...#0003"],
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR      = _PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Haiku 4.5 pricing (USD per million tokens)
# Source: https://www.anthropic.com/pricing  (verified Sep 2026)
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    # claude-haiku-4-5-20251001
    "claude-haiku-4-5-20251001": {
        "input_per_m":       0.80,
        "output_per_m":      4.00,
        "cache_write_per_m": 1.00,
        "cache_read_per_m":  0.08,
    },
    # claude-haiku-3 (legacy, kept for reference)
    "claude-3-haiku-20240307": {
        "input_per_m":       0.25,
        "output_per_m":      1.25,
        "cache_write_per_m": 0.30,
        "cache_read_per_m":  0.03,
    },
}

# Fallback for unknown models: assume same as Haiku 4.5
_DEFAULT_PRICING = _PRICING["claude-haiku-4-5-20251001"]


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Compute the USD cost for a single API call.

    Non-cached input tokens = input_tokens - cache_write_tokens - cache_read_tokens.
    """
    pricing  = _PRICING.get(model, _DEFAULT_PRICING)
    plain_in = max(input_tokens - cache_write_tokens - cache_read_tokens, 0)

    cost = (
        plain_in             * pricing["input_per_m"]       / 1_000_000
        + cache_write_tokens * pricing["cache_write_per_m"] / 1_000_000
        + cache_read_tokens  * pricing["cache_read_per_m"]  / 1_000_000
        + output_tokens      * pricing["output_per_m"]       / 1_000_000
    )
    return round(cost, 8)


# ---------------------------------------------------------------------------
# Log file path (one file per UTC day)
# ---------------------------------------------------------------------------

def _log_file_for_date(date: datetime | None = None) -> Path:
    """Return the JSONL path for the given UTC date (default: today)."""
    if date is None:
        date = datetime.now(timezone.utc)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"queries_{date.strftime('%Y-%m-%d')}.jsonl"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query_event(
    *,
    query:                str,
    pipeline:             str,                        # "baseline" | "contextual"
    retrieval_latency_ms: float,                      # dense + BM25 + RRF
    rerank_latency_ms:    float,                      # cross-encoder reranking
    generation_latency_ms: float,                     # LLM wall time
    generation_model:     str,
    input_tokens:         int,
    output_tokens:        int,
    retrieved_chunk_ids:  Sequence[str],
    cache_write_tokens:   int = 0,
    cache_read_tokens:    int = 0,
    extra:                dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a structured query event and append it to today's JSONL log file.

    Parameters
    ----------
    query                 : The raw user query string.
    pipeline              : Pipeline name ("baseline" or "contextual").
    retrieval_latency_ms  : Time for hybrid retrieval (dense + BM25 + RRF), in ms.
    rerank_latency_ms     : Time for cross-encoder reranking, in ms.
    generation_latency_ms : LLM API wall-clock time, in ms.
    generation_model      : Anthropic model identifier.
    input_tokens          : Prompt token count from the API response.
    output_tokens         : Completion token count from the API response.
    retrieved_chunk_ids   : Ordered list of final chunk IDs returned to the user.
    cache_write_tokens    : Prompt-cache write tokens (default 0).
    cache_read_tokens     : Prompt-cache read tokens (default 0).
    extra                 : Optional dict of arbitrary extra fields to merge in.

    Returns
    -------
    dict  The event dict that was written (useful for tests / chaining).
    """
    now = datetime.now(timezone.utc)

    total_retrieval_ms = retrieval_latency_ms + rerank_latency_ms
    total_latency_ms   = total_retrieval_ms + generation_latency_ms

    cost = compute_cost(
        model              = generation_model,
        input_tokens       = input_tokens,
        output_tokens      = output_tokens,
        cache_write_tokens = cache_write_tokens,
        cache_read_tokens  = cache_read_tokens,
    )

    event: dict[str, Any] = {
        "timestamp":              now.isoformat(),
        "query":                  query,
        "pipeline":               pipeline,
        "retrieval_latency_ms":   round(retrieval_latency_ms, 2),
        "rerank_latency_ms":      round(rerank_latency_ms, 2),
        "total_retrieval_ms":     round(total_retrieval_ms, 2),
        "generation_latency_ms":  round(generation_latency_ms, 2),
        "total_latency_ms":       round(total_latency_ms, 2),
        "generation_model":       generation_model,
        "input_tokens":           input_tokens,
        "output_tokens":          output_tokens,
        "cache_write_tokens":     cache_write_tokens,
        "cache_read_tokens":      cache_read_tokens,
        "cost_usd":               cost,
        "num_chunks_retrieved":   len(retrieved_chunk_ids),
        "retrieved_chunk_ids":    list(retrieved_chunk_ids),
    }

    if extra:
        event.update(extra)

    log_path = _log_file_for_date(now)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    return event
