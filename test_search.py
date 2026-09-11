"""
test_search.py
--------------
Phase 4 smoke-test: runs 3 ConfigMaps-focused queries through BOTH the
baseline and contextual pipelines (hybrid retrieval + cross-encoder reranking)
and prints a side-by-side comparison of the top-5 results.

Run from the project root:
    python test_search.py

Because the contextual collection currently only contains the 20 chunks from
the ConfigMaps page, these queries are deliberately scoped to ConfigMaps so the
comparison is meaningful.
"""

from __future__ import annotations

import io
import sys
import textwrap
from typing import Any

# ── Windows UTF-8 stdout fix ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# Import after stdout fix so tqdm / other libs don't choke on Unicode
from src.retrieval.pipeline import make_baseline_pipeline, make_contextual_pipeline

# ---------------------------------------------------------------------------
# Test queries (all ConfigMaps-specific)
# ---------------------------------------------------------------------------
QUERIES = [
    "how do I mount specific keys from a ConfigMap as files in a pod",
    "how do I expose ConfigMap values as environment variables",
    "what's the difference between using a ConfigMap as an env var vs a volume",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
COL_WIDTH    = 72   # characters per pipeline column
DIVIDER_CHAR = "─"
H1_CHAR      = "═"


def _fmt_result(rank: int, result: dict[str, Any]) -> str:
    """Format one reranked result into a printable block."""
    payload = result.get("payload", {})
    chunk_id    = result.get("chunk_id", payload.get("chunk_id", "???"))
    rrf_score   = result.get("rrf_score", 0.0)
    rerank_score= result.get("rerank_score", 0.0)
    page_title  = payload.get("page_title", "")
    source_url  = payload.get("source_url", "")
    chunk_idx   = payload.get("chunk_index", "?")

    # Prefer contextualized text for display in contextual results
    text = (
        payload.get("contextualized_text")
        or payload.get("text")
        or ""
    )
    # Trim to a readable snippet
    snippet = textwrap.fill(text[:350].replace("\n", " "), width=COL_WIDTH - 4)
    if len(text) > 350:
        snippet += " [...]"

    lines = [
        f"  [{rank}] {chunk_id}",
        f"      page   : {page_title}  (chunk #{chunk_idx})",
        f"      rrf    : {rrf_score:.5f}  |  rerank : {rerank_score:.4f}",
        f"      url    : {source_url}",
        f"      snippet:",
    ]
    for line in snippet.split("\n"):
        lines.append(f"        {line}")
    return "\n".join(lines)


def _print_comparison(
    query:    str,
    baseline: list[dict[str, Any]],
    contextual: list[dict[str, Any]],
    query_num: int,
) -> None:
    """Print a side-by-side (stacked) comparison for one query."""
    border = H1_CHAR * (COL_WIDTH + 4)
    divider = DIVIDER_CHAR * (COL_WIDTH + 4)

    print(border)
    print(f"  QUERY {query_num}: {query}")
    print(border)

    # ── Baseline ──────────────────────────────────────────────────────
    print(f"\n  {'BASELINE PIPELINE':^{COL_WIDTH}}  (k8s_baseline | all chunks)")
    print(f"  {DIVIDER_CHAR * COL_WIDTH}")
    if baseline:
        for i, r in enumerate(baseline, 1):
            print(_fmt_result(i, r))
            if i < len(baseline):
                print(f"  {'·' * (COL_WIDTH - 2)}")
    else:
        print("  (no results)")

    # ── Contextual ────────────────────────────────────────────────────
    print(f"\n  {'CONTEXTUAL PIPELINE':^{COL_WIDTH}}  (k8s_contextual | 20 ConfigMaps chunks)")
    print(f"  {DIVIDER_CHAR * COL_WIDTH}")
    if contextual:
        for i, r in enumerate(contextual, 1):
            print(_fmt_result(i, r))
            if i < len(contextual):
                print(f"  {'·' * (COL_WIDTH - 2)}")
    else:
        print("  (no results)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nLoading pipelines (embedding model + BM25 index) ...")
    print("  Building baseline pipeline  ...")
    baseline_pipe    = make_baseline_pipeline(hybrid_top_k=20, rerank_top_k=5)
    print(f"    {baseline_pipe}  — {baseline_pipe.retriever.num_chunks:,} chunks in BM25 index")
    print("  Building contextual pipeline ...")
    contextual_pipe  = make_contextual_pipeline(hybrid_top_k=20, rerank_top_k=5)
    print(f"    {contextual_pipe}  — {contextual_pipe.retriever.num_chunks:,} chunks in BM25 index")
    print("\nAll pipelines ready. Running queries ...\n")

    for q_num, query in enumerate(QUERIES, 1):
        print(f"[Query {q_num}/{len(QUERIES)}] Retrieving: {query!r}")
        baseline_results   = baseline_pipe.run(query)
        contextual_results = contextual_pipe.run(query)
        _print_comparison(query, baseline_results, contextual_results, q_num)

    print("Done.")


if __name__ == "__main__":
    main()
