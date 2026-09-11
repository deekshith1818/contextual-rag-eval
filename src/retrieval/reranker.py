"""
reranker.py
-----------
Cross-encoder reranker that takes a list of candidate chunks (from hybrid
retrieval) and re-scores them with BAAI/bge-reranker-base, returning the
top-k most relevant ones.

Usage
-----
    from src.retrieval.reranker import Reranker

    reranker = Reranker()
    top5 = reranker.rerank(query="...", candidates=fused_results, top_k=5)
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_RERANK_TOP_K = 5

# Singleton cross-encoder (shared across all Reranker instances)
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Cross-encoder reranker backed by BAAI/bge-reranker-base.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the cross-encoder (default: BAAI/bge-reranker-base).
    top_k : int
        Number of results to return after reranking (default: 5).
    """

    def __init__(
        self,
        model_name: str  = RERANKER_MODEL_NAME,
        top_k:      int  = DEFAULT_RERANK_TOP_K,
    ) -> None:
        self.model_name = model_name
        self.top_k      = top_k
        # Force-load the model now so start-up cost is paid once.
        _get_cross_encoder()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        """
        Pull the best available text from a retrieval result dict.

        For baseline payloads: 'text'
        For contextual payloads: 'contextualized_text' (preferred) or 'text'
        """
        payload = item.get("payload", item)  # support both wrapped & raw dicts
        return (
            payload.get("contextualized_text")
            or payload.get("text")
            or ""
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        query:      str,
        candidates: list[dict[str, Any]],
        top_k:      int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank a list of retrieval candidates using the cross-encoder.

        Parameters
        ----------
        query : str
            The original user query.
        candidates : list[dict]
            Retrieval results, each with a 'payload' dict containing at least
            a 'text' (and optionally 'contextualized_text') field.
        top_k : int | None
            How many results to return. Defaults to self.top_k.

        Returns
        -------
        list[dict]
            Re-ranked subset, each item augmented with a 'rerank_score' key.
        """
        if not candidates:
            return []

        k = top_k if top_k is not None else self.top_k

        # Build (query, passage) pairs for the cross-encoder
        texts  = [self._extract_text(c) for c in candidates]
        pairs  = [[query, t] for t in texts]

        model  = _get_cross_encoder()
        scores = model.predict(pairs)   # numpy array of floats

        # Attach scores and sort descending
        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:k]
