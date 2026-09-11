"""
pipeline.py
-----------
End-to-end retrieval pipeline: HybridRetriever  →  Reranker.

Provides a thin Pipeline class that wires the two stages together and
a pair of pre-built factory functions for the two supported configurations:

    make_baseline_pipeline()    -> Pipeline("k8s_baseline",    baseline_chunks.jsonl)
    make_contextual_pipeline()  -> Pipeline("k8s_contextual",  contextual_chunks.jsonl)

Usage
-----
    from src.retrieval.pipeline import make_baseline_pipeline, make_contextual_pipeline

    pipe = make_baseline_pipeline()
    results = pipe.run("how do I mount ConfigMap keys as files?", top_k=5)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# ---------------------------------------------------------------------------
# Default paths (relative to the project root; callers may override)
# ---------------------------------------------------------------------------
_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
_PROCESSED_DIR   = _PROJECT_ROOT / "data" / "processed"

BASELINE_JSONL    = _PROCESSED_DIR / "baseline_chunks.jsonl"
CONTEXTUAL_JSONL  = _PROCESSED_DIR / "contextual_chunks.jsonl"

COLLECTION_BASELINE   = "k8s_baseline"
COLLECTION_CONTEXTUAL = "k8s_contextual"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Chains HybridRetriever and Reranker into a single search pipeline.

    Parameters
    ----------
    collection_name : str
        Qdrant collection to run dense search against.
    jsonl_path : str | Path
        JSONL corpus used for BM25.
    hybrid_top_k : int
        Candidates returned by the hybrid retriever before reranking (default: 20).
    rerank_top_k : int
        Final results returned after reranking (default: 5).
    """

    def __init__(
        self,
        collection_name: str,
        jsonl_path:      str | Path,
        hybrid_top_k:    int = 20,
        rerank_top_k:    int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.jsonl_path      = Path(jsonl_path)
        self.hybrid_top_k    = hybrid_top_k
        self.rerank_top_k    = rerank_top_k

        self.retriever = HybridRetriever(
            collection_name=collection_name,
            jsonl_path=jsonl_path,
        )
        self.reranker = Reranker(top_k=rerank_top_k)

    def run(
        self,
        query:        str,
        hybrid_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute the full retrieval pipeline for a single query.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        hybrid_top_k : int | None
            Override for the hybrid retrieval candidate count.
        rerank_top_k : int | None
            Override for the reranker output size.

        Returns
        -------
        list[dict]
            Top-k reranked results, each containing:
            - chunk_id     : str
            - rrf_score    : float
            - rerank_score : float
            - payload      : dict
        """
        h_k = hybrid_top_k if hybrid_top_k is not None else self.hybrid_top_k
        r_k = rerank_top_k if rerank_top_k is not None else self.rerank_top_k

        candidates = self.retriever.retrieve(query, top_k=h_k)
        return self.reranker.rerank(query, candidates, top_k=r_k)

    def run_with_generation(
        self,
        query:            str,
        anthropic_client: Any,
        model:            str  = "claude-haiku-4-5-20251001",
        max_tokens:       int  = 400,
        system_prompt:    str  = (
            "You are an expert Kubernetes assistant. Answer the user's question "
            "based strictly on the provided context documentation chunks below. "
            "If the context does not contain enough information, state that clearly. "
            "Keep your answer concise, technical, and accurate."
        ),
        hybrid_top_k:     int | None = None,
        rerank_top_k:     int | None = None,
        log:              bool = True,
    ) -> dict[str, Any]:
        """
        Run the complete query → retrieve → rerank → generate pipeline with
        automatic per-stage timing and observability logging.

        Parameters
        ----------
        query            : Natural-language question from the user.
        anthropic_client : An initialised ``anthropic.Anthropic`` client.
        model            : Anthropic model ID (default: claude-haiku-4-5-20251001).
        max_tokens       : Max tokens for the generation response (default: 400).
        system_prompt    : System prompt for the LLM.
        hybrid_top_k     : Override hybrid retrieval candidate count.
        rerank_top_k     : Override reranker output size.
        log              : If True (default), write an event to the JSONL log.

        Returns
        -------
        dict with keys:
            answer         – str   : The generated answer text.
            chunks         – list  : Reranked chunk dicts passed to the LLM.
            retrieval_ms   – float : Hybrid search + RRF time in ms.
            rerank_ms      – float : Cross-encoder time in ms.
            generation_ms  – float : LLM API wall time in ms.
            total_ms       – float : End-to-end time in ms.
            input_tokens   – int
            output_tokens  – int
            cost_usd       – float
            log_event      – dict | None : The event written to disk (None if log=False).
        """
        from src.observability.logger import log_query_event, compute_cost

        h_k = hybrid_top_k if hybrid_top_k is not None else self.hybrid_top_k
        r_k = rerank_top_k if rerank_top_k is not None else self.rerank_top_k

        # ── Stage 1 & 2: Hybrid retrieval (dense + BM25 + RRF) ───────────────
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query, top_k=h_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 3: Cross-encoder reranking ─────────────────────────────────
        t1 = time.perf_counter()
        chunks = self.reranker.rerank(query, candidates, top_k=r_k)
        rerank_ms = (time.perf_counter() - t1) * 1000

        # ── Build context string for the LLM ──────────────────────────────────
        context_blocks: list[str] = []
        for i, c in enumerate(chunks, start=1):
            payload = c.get("payload", c)
            text    = (
                payload.get("contextualized_text")
                or payload.get("text")
                or ""
            )
            cid     = c.get("chunk_id", f"chunk_{i}")
            context_blocks.append(f"[Chunk {i}: {cid}]\n{text}")
        context_str  = "\n\n".join(context_blocks)
        user_content = (
            f"Context:\n---\n{context_str}\n---\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        # ── Stage 4: LLM generation ───────────────────────────────────────────
        t2 = time.perf_counter()
        response = anthropic_client.messages.create(
            model      = model,
            max_tokens = max_tokens,
            temperature= 0.0,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_content}],
        )
        generation_ms = (time.perf_counter() - t2) * 1000

        answer        = response.content[0].text.strip()
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_ms      = retrieval_ms + rerank_ms + generation_ms

        cost      = compute_cost(
            model         = model,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
        )
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        # ── Observability logging ─────────────────────────────────────────────
        log_event = None
        if log:
            pipeline_name = (
                "contextual" if "contextual" in self.collection_name else "baseline"
            )
            log_event = log_query_event(
                query                 = query,
                pipeline              = pipeline_name,
                retrieval_latency_ms  = retrieval_ms,
                rerank_latency_ms     = rerank_ms,
                generation_latency_ms = generation_ms,
                generation_model      = model,
                input_tokens          = input_tokens,
                output_tokens         = output_tokens,
                retrieved_chunk_ids   = chunk_ids,
            )

        return {
            "answer":        answer,
            "chunks":        chunks,
            "retrieval_ms":  round(retrieval_ms, 2),
            "rerank_ms":     round(rerank_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms":      round(total_ms, 2),
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost,
            "log_event":     log_event,
        }

    def __repr__(self) -> str:
        return (
            f"Pipeline(collection={self.collection_name!r}, "
            f"jsonl={self.jsonl_path.name!r}, "
            f"hybrid_top_k={self.hybrid_top_k}, "
            f"rerank_top_k={self.rerank_top_k})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_baseline_pipeline(
    hybrid_top_k: int = 20,
    rerank_top_k: int = 5,
) -> Pipeline:
    """Return a pipeline backed by the full baseline collection."""
    return Pipeline(
        collection_name=COLLECTION_BASELINE,
        jsonl_path=BASELINE_JSONL,
        hybrid_top_k=hybrid_top_k,
        rerank_top_k=rerank_top_k,
    )


def make_contextual_pipeline(
    hybrid_top_k: int = 20,
    rerank_top_k: int = 5,
) -> Pipeline:
    """Return a pipeline backed by the (partial) contextual collection."""
    return Pipeline(
        collection_name=COLLECTION_CONTEXTUAL,
        jsonl_path=CONTEXTUAL_JSONL,
        hybrid_top_k=hybrid_top_k,
        rerank_top_k=rerank_top_k,
    )
