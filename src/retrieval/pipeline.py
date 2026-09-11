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
