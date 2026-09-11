"""
diagnose_retrieval.py
---------------------
Phase 4 verification script — NO existing source files are modified.

For each of 3 ConfigMaps queries this script prints all 4 retrieval
stages with scores and provenance labels, then a reranking-vs-fused
diagnostic.

Stages per query
----------------
  1. Dense-only top 5   (cosine similarity from Qdrant)
  2. BM25-only  top 5   (BM25Okapi score)
  3. RRF-fused  top 20  (marked: dense-only / bm25-only / both)
  4. Reranked   top 5   (cross-encoder score)
  + Diagnostic: did reranking change the order vs first-5 of fused?

Run:
    python diagnose_retrieval.py
"""

from __future__ import annotations
import io, sys, os, textwrap
from pathlib import Path
from typing import Any

# ── Windows UTF-8 stdout fix ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Set env vars to suppress noisy warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add project root to path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.hybrid_retriever import HybridRetriever, _get_embed_model, _tokenize
from src.retrieval.reranker import Reranker

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
BASELINE_JSONL = PROCESSED_DIR / "baseline_chunks.jsonl"
COLLECTION     = "k8s_baseline"

QUERIES = [
    "how do I mount specific keys from a ConfigMap as files in a pod",
    "how do I expose ConfigMap values as environment variables",
    "what happens if I update a ConfigMap that a running pod is using",
]

W = 80  # line width


# ── Helpers ──────────────────────────────────────────────────────────────────

def _text(item: dict[str, Any], max_chars: int = 300) -> str:
    p   = item.get("payload", item)
    raw = p.get("text") or ""
    snip = raw.replace("\n", " ").strip()[:max_chars]
    if len(raw) > max_chars:
        snip += " [...]"
    return textwrap.fill(snip, width=W - 6)


def _header(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print("═" * W)


def _subhdr(title: str) -> None:
    print(f"\n  {'─' * (W-2)}")
    print(f"  {title}")
    print(f"  {'─' * (W-2)}")


def _sep() -> None:
    print(f"  {'·' * (W-4)}")


# ── Stage 1: Dense only ──────────────────────────────────────────────────────

def show_dense(retriever: HybridRetriever, query: str) -> list[dict]:
    hits = retriever._dense_retrieve(query)
    _subhdr("STAGE 1 — Dense-only top 5  (Qdrant cosine similarity)")
    for i, h in enumerate(hits[:5], 1):
        p = h["payload"]
        print(f"\n  [{i}] chunk_id : {h['chunk_id']}")
        print(f"      score    : {h['score']:.5f}")
        print(f"      page     : {p.get('page_title','')}  chunk#{p.get('chunk_index','?')}")
        for line in _text(h).split("\n"):
            print(f"       {line}")
        if i < 5:
            _sep()
    return hits


# ── Stage 2: BM25 only ───────────────────────────────────────────────────────

def show_bm25(retriever: HybridRetriever, query: str) -> list[dict]:
    hits = retriever._bm25_retrieve(query)
    _subhdr("STAGE 2 — BM25-only top 5  (BM25Okapi score)")
    for i, h in enumerate(hits[:5], 1):
        p = h["payload"]
        print(f"\n  [{i}] chunk_id : {h['chunk_id']}")
        print(f"      score    : {h['score']:.5f}")
        print(f"      page     : {p.get('page_title','')}  chunk#{p.get('chunk_index','?')}")
        for line in _text(h).split("\n"):
            print(f"       {line}")
        if i < 5:
            _sep()
    return hits


# ── Stage 3: RRF Fused top 20 ───────────────────────────────────────────────

def show_fused(
    retriever: HybridRetriever,
    dense_hits: list[dict],
    bm25_hits:  list[dict],
) -> list[dict]:
    dense_ids = {h["chunk_id"] for h in dense_hits}
    bm25_ids  = {h["chunk_id"] for h in bm25_hits}

    fused = retriever._rrf_fuse([dense_hits, bm25_hits], k=retriever.rrf_k)

    _subhdr("STAGE 3 — RRF-fused top 20  (RRF score | provenance)")
    for i, h in enumerate(fused[:20], 1):
        cid = h["chunk_id"]
        in_d = cid in dense_ids
        in_b = cid in bm25_ids
        if in_d and in_b:
            prov = "BOTH  ★"
        elif in_d:
            prov = "dense-only"
        else:
            prov = "bm25-only"

        p = h["payload"]
        print(f"\n  [{i:2d}] chunk_id : {cid}")
        print(f"       rrf      : {h['rrf_score']:.6f}  | provenance: {prov}")
        print(f"       page     : {p.get('page_title','')}  chunk#{p.get('chunk_index','?')}")
        snip = (p.get("text") or "").replace("\n", " ").strip()[:200]
        print(f"       text     : {snip} [...]")

    return fused


# ── Stage 4: Reranked top 5 + diagnostic ────────────────────────────────────

def show_reranked(
    reranker:   Reranker,
    query:      str,
    fused:      list[dict],
) -> None:
    fused_top5_ids = [h["chunk_id"] for h in fused[:5]]

    reranked = reranker.rerank(query, fused[:20], top_k=5)
    reranked_ids = [h["chunk_id"] for h in reranked]

    _subhdr("STAGE 4 — Reranked top 5  (cross-encoder score)")
    for i, h in enumerate(reranked, 1):
        p = h["payload"]
        print(f"\n  [{i}] chunk_id     : {h['chunk_id']}")
        print(f"      rerank_score : {h['rerank_score']:.4f}")
        print(f"      rrf_score    : {h.get('rrf_score', 0.0):.6f}")
        print(f"      page         : {p.get('page_title','')}  chunk#{p.get('chunk_index','?')}")
        for line in _text(h).split("\n"):
            print(f"       {line}")
        if i < 5:
            _sep()

    # ── Diagnostic ────────────────────────────────────────────────────────
    _subhdr("DIAGNOSTIC — Did reranking change the order vs fused[:5]?")
    changed = (reranked_ids != fused_top5_ids)
    print(f"\n  Changed order: {'YES' if changed else 'NO'}")
    print(f"\n  Fused[:5]   order: {fused_top5_ids}")
    print(f"  Reranked[:5] order: {reranked_ids}")

    if changed:
        print("\n  Differences:")
        for pos, (f_id, r_id) in enumerate(zip(fused_top5_ids, reranked_ids), 1):
            if f_id != r_id:
                # find where r_id was in fused_top5
                fused_pos = fused_top5_ids.index(r_id) + 1 if r_id in fused_top5_ids else "outside top-5"
                print(f"    position {pos}: fused had '{f_id}' → reranker promoted '{r_id}'"
                      f" (was at fused pos {fused_pos})")
        # check for chunks that entered top-5 from outside fused[:5]
        newcomers = [r for r in reranked_ids if r not in fused_top5_ids]
        for nc in newcomers:
            fused_rank = next((i+1 for i, h in enumerate(fused) if h["chunk_id"] == nc), "?")
            print(f"    newcomer from fused position {fused_rank}: '{nc}'")
    else:
        print("\n  → The cross-encoder confirmed the RRF ranking for all top-5.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*W}")
    print("  Phase 4 Verification — Hybrid Retrieval + Reranking Diagnostic")
    print(f"  Collection : {COLLECTION}")
    print(f"  JSONL      : {BASELINE_JSONL.name}  ({BASELINE_JSONL.stat().st_size/1e6:.1f} MB)")
    print(f"{'═'*W}\n")

    print("Loading HybridRetriever (builds BM25 index, lazy-loads embed model)...")
    retriever = HybridRetriever(
        collection_name=COLLECTION,
        jsonl_path=BASELINE_JSONL,
        dense_top_k=20,
        bm25_top_k=20,
        rrf_k=60,
    )
    print(f"  BM25 index: {retriever.num_chunks:,} chunks\n")

    print("Loading Reranker (BAAI/bge-reranker-base)...")
    reranker = Reranker(top_k=5)
    print("  Reranker ready.\n")

    for q_num, query in enumerate(QUERIES, 1):
        _header(f"QUERY {q_num}/{len(QUERIES)}: {query!r}")

        dense_hits = show_dense(retriever, query)
        bm25_hits  = show_bm25(retriever, query)
        fused      = show_fused(retriever, dense_hits, bm25_hits)
        show_reranked(reranker, query, fused)

    print(f"\n{'═'*W}")
    print("  Verification complete.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
