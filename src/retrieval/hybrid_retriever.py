"""
hybrid_retriever.py
--------------------
Implements a two-leg hybrid retrieval pipeline:

  1. Dense retrieval  – query → embed (BAAI/bge-base-en-v1.5) → Qdrant ANN search
  2. Sparse retrieval – BM25 over the raw chunks loaded from a JSONL file
  3. Fusion           – Reciprocal Rank Fusion (RRF, k=60) over both ranked lists

The retriever is fully parameterized: pass the Qdrant collection name and the
path to the JSONL corpus, so the same class works for either:

  * k8s_baseline    + data/processed/baseline_chunks.jsonl
  * k8s_contextual  + data/processed/contextual_chunks.jsonl

Usage
-----
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        collection_name="k8s_baseline",
        jsonl_path="data/processed/baseline_chunks.jsonl",
    )
    results = retriever.retrieve("how do I use ConfigMap env vars?", top_k=20)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_TOP_K      = 20   # candidates from each leg before fusion
BM25_TOP_K       = 20
RRF_K            = 60   # RRF smoothing constant

# Lazily-loaded singleton embedding model (shared across instances)
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


# ---------------------------------------------------------------------------
# Qdrant client factory (mirrors setup_qdrant.py logic)
# ---------------------------------------------------------------------------

def _make_qdrant_client() -> QdrantClient:
    load_dotenv()
    url = os.getenv("QDRANT_URL", "").strip()
    key = os.getenv("QDRANT_API_KEY", "").strip()

    if url and not url.startswith("http://localhost"):
        return QdrantClient(url=url, api_key=key or None, timeout=60.0)
    return QdrantClient(url=url or "http://localhost:6333", timeout=60.0)


# ---------------------------------------------------------------------------
# Helper: simple whitespace tokenizer for BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Dense + BM25 hybrid retriever with Reciprocal Rank Fusion.

    Parameters
    ----------
    collection_name : str
        Qdrant collection to search (e.g. "k8s_baseline" or "k8s_contextual").
    jsonl_path : str | Path
        Path to the JSONL file that backs the collection.
        * For k8s_baseline   -> use the "text" field
        * For k8s_contextual -> if "contextualized_text" is present it is used
          for BM25; otherwise falls back to "text"
    dense_top_k : int
        Number of candidates to retrieve from each leg (default: 20).
    bm25_top_k : int
        Number of BM25 candidates (default: 20).
    rrf_k : int
        RRF smoothing constant (default: 60).
    """

    def __init__(
        self,
        collection_name: str,
        jsonl_path: str | Path,
        dense_top_k: int = DENSE_TOP_K,
        bm25_top_k:  int = BM25_TOP_K,
        rrf_k:       int = RRF_K,
    ) -> None:
        self.collection_name = collection_name
        self.jsonl_path      = Path(jsonl_path)
        self.dense_top_k     = dense_top_k
        self.bm25_top_k      = bm25_top_k
        self.rrf_k           = rrf_k

        # Load chunks + build BM25 index eagerly
        self._chunks: list[dict[str, Any]] = self._load_chunks()
        self._bm25_index, self._bm25_texts = self._build_bm25_index()

        # Lazy clients (first retrieve call)
        self._qdrant: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_chunks(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"[HybridRetriever] JSONL not found: {self.jsonl_path}"
            )
        with self.jsonl_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _build_bm25_index(self) -> tuple[BM25Okapi, list[str]]:
        """
        Build a BM25 index over chunks.
        Prefer 'contextualized_text' if available (contextual collection),
        otherwise fall back to 'text'.
        """
        texts: list[str] = []
        for chunk in self._chunks:
            txt = chunk.get("contextualized_text") or chunk.get("text", "")
            texts.append(txt)

        tokenized = [_tokenize(t) for t in texts]
        return BM25Okapi(tokenized), texts

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = _make_qdrant_client()
        return self._qdrant

    # ------------------------------------------------------------------
    # Retrieval legs
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Embed the query and perform an ANN search in Qdrant.
        Returns a list of dicts with keys: chunk_id, score, payload.

        Uses query_points() (qdrant-client >= 1.9). For older clients,
        fall back to search() automatically.
        """
        model = _get_embed_model()
        # BGE instruction-prefixed query embedding
        q_vec = model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()

        client = self._get_qdrant()

        # qdrant-client >= 1.9 uses query_points(); < 1.9 used search()
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=self.dense_top_k,
                with_payload=True,
            )
            scored_points = response.points
        else:
            scored_points = client.search(
                collection_name=self.collection_name,
                query_vector=q_vec,
                limit=self.dense_top_k,
                with_payload=True,
            )

        hits = []
        for hit in scored_points:
            payload = hit.payload or {}
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "score":    hit.score,
                    "payload":  payload,
                    "source":   "dense",
                }
            )
        return hits

    def _bm25_retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Score all chunks with BM25 and return top-k.
        Returns a list of dicts with keys: chunk_id, score, payload (chunk dict).
        """
        q_tokens = _tokenize(query)
        scores   = self._bm25_index.get_scores(q_tokens)

        # Sort indices by descending score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.bm25_top_k]

        hits = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            hits.append(
                {
                    "chunk_id": chunk.get("chunk_id", str(idx)),
                    "score":    float(scores[idx]),
                    "payload":  chunk,
                    "source":   "bm25",
                }
            )
        return hits

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[dict[str, Any]]],
        k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.

        Each item is keyed by 'chunk_id'. The fused score is:
            sum_over_lists(1 / (k + rank))   where rank is 1-indexed.

        Returns a merged, re-ranked list of dicts with an added 'rrf_score'.
        """
        rrf_scores: dict[str, float] = {}
        # Keep one payload reference per chunk_id (from whichever list first sees it)
        payloads:   dict[str, dict]  = {}

        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                cid = item["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in payloads:
                    payloads[cid] = item["payload"]

        fused = []
        for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            fused.append(
                {
                    "chunk_id":  cid,
                    "rrf_score": rrf_score,
                    "payload":   payloads[cid],
                }
            )
        return fused

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        Run hybrid retrieval for a query.

        Parameters
        ----------
        query : str
            Natural-language search query.
        top_k : int
            Maximum number of results to return after RRF fusion (default: 20).

        Returns
        -------
        list[dict]
            Sorted list of fused results, each containing:
            - chunk_id   : str
            - rrf_score  : float
            - payload    : dict  (the Qdrant/chunk payload)
        """
        dense_hits = self._dense_retrieve(query)
        bm25_hits  = self._bm25_retrieve(query)

        fused = self._rrf_fuse([dense_hits, bm25_hits], k=self.rrf_k)
        return fused[:top_k]

    # Convenience property
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
