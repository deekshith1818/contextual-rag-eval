"""
setup_qdrant.py
---------------
Creates two Qdrant collections and ingests chunks:

  - "k8s_baseline"   <-- embeds raw chunk text from baseline_chunks.jsonl
  - "k8s_contextual" <-- embeds contextualized_text from contextual_chunks.jsonl

Embeddings are computed locally with sentence-transformers (BAAI/bge-base-en-v1.5),
so there is no API cost for embedding.

Connection:
  - If QDRANT_URL env var is set to a cloud URL, uses cloud with QDRANT_API_KEY.
  - Otherwise falls back to http://localhost:6333 (local Docker / binary).

Usage:
    python src/retrieval/setup_qdrant.py
    python src/retrieval/setup_qdrant.py --collection baseline    # only baseline
    python src/retrieval/setup_qdrant.py --collection contextual  # only contextual
    python src/retrieval/setup_qdrant.py --recreate               # drop & recreate
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# ── Windows UTF-8 stdout fix ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
BASELINE_JSONL  = PROCESSED_DIR / "baseline_chunks.jsonl"
CONTEXTUAL_JSONL= PROCESSED_DIR / "contextual_chunks.jsonl"

EMBED_MODEL_NAME   = "BAAI/bge-base-en-v1.5"
EMBED_DIMENSION    = 768          # BAAI/bge-base-en-v1.5 output dim
COLLECTION_BASELINE    = "k8s_baseline"
COLLECTION_CONTEXTUAL  = "k8s_contextual"
BATCH_SIZE = 64                   # points per upsert batch

# ---------------------------------------------------------------------------
# Qdrant connection
# ---------------------------------------------------------------------------

def make_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()

    if qdrant_url and not qdrant_url.startswith("http://localhost"):
        print(f"[Qdrant] Connecting to cloud: {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_key or None, timeout=60.0)
    else:
        local_url = qdrant_url or "http://localhost:6333"
        print(f"[Qdrant] Connecting to local: {local_url}")
        return QdrantClient(url=local_url, timeout=60.0)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool) -> None:
    """Create the collection if it doesn't exist (or recreate if requested)."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if recreate:
            print(f"  [DROP] Deleting existing collection '{name}'")
            client.delete_collection(name)
        else:
            print(f"  [SKIP] Collection '{name}' already exists. Use --recreate to overwrite.")
            return

    print(f"  [CREATE] Creating collection '{name}' (dim={dim}, cosine distance)")
    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(
            size=dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def print_collection_stats(client: QdrantClient, name: str) -> None:
    info = client.get_collection(name)
    count = info.points_count
    dim   = info.config.params.vectors.size
    print(f"\n  [STATS] '{name}'")
    print(f"          Points   : {count:,}")
    print(f"          Vector dim: {dim}")
    print(f"          Status   : {info.status}")


# ---------------------------------------------------------------------------
# Ingest one collection
# ---------------------------------------------------------------------------

def ingest(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    jsonl_path: Path,
    text_field: str,     # "text" for baseline, "contextualized_text" for contextual
    recreate: bool,
) -> None:
    if not jsonl_path.exists():
        print(f"\n[SKIP] {jsonl_path.name} not found — run the prior ingestion step first.")
        return

    print(f"\n{'='*60}")
    print(f"Ingesting into '{collection_name}'")
    print(f"  Source : {jsonl_path}")
    print(f"  Field  : '{text_field}'")
    print("=" * 60)

    ensure_collection(client, collection_name, EMBED_DIMENSION, recreate)

    # Check if collection was just created or already existed without recreate
    existing_count = client.count(collection_name).count
    if existing_count > 0 and not recreate:
        print(f"  [INFO] Collection already has {existing_count:,} points. Skipping upsert.")
        print_collection_stats(client, collection_name)
        return

    # Load chunks
    with jsonl_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"  Loaded {len(chunks):,} chunks from disk.")

    # Embed in batches
    texts = [c.get(text_field, c.get("text", "")) for c in chunks]
    print(f"  Embedding {len(texts):,} texts with '{EMBED_MODEL_NAME}' ...")

    all_vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Embedding", unit="batch"):
        batch_texts = texts[i : i + BATCH_SIZE]
        # BGE models benefit from the query prefix for embedding
        vecs = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_vectors.extend(vecs.tolist())

    # Upsert in batches
    print(f"  Upserting {len(all_vectors):,} vectors ...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Upserting", unit="batch"):
        batch_chunks  = chunks[i : i + BATCH_SIZE]
        batch_vectors = all_vectors[i : i + BATCH_SIZE]

        points = []
        for j, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors)):
            # Use a stable integer ID derived from the global chunk index
            point_id = i + j
            payload  = {
                "chunk_id":        chunk.get("chunk_id", ""),
                "source_file":     chunk.get("source_file", ""),
                "source_url":      chunk.get("source_url", ""),
                "page_title":      chunk.get("page_title", ""),
                "section_category":chunk.get("section_category", ""),
                "chunk_index":     chunk.get("chunk_index", 0),
                "char_start":      chunk.get("char_start", 0),
                "char_end":        chunk.get("char_end", 0),
                "chunk_size_chars":chunk.get("chunk_size_chars", 0),
                "text":            chunk.get("text", ""),
                # contextual fields (empty string in baseline collection)
                "context_blurb":       chunk.get("context_blurb", ""),
                "contextualized_text": chunk.get("contextualized_text", ""),
            }
            points.append(
                qdrant_models.PointStruct(id=point_id, vector=vec, payload=payload)
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)

    print_collection_stats(client, collection_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest k8s chunks into Qdrant.")
    parser.add_argument(
        "--collection", choices=["baseline", "contextual", "both"], default="both",
        help="Which collection to populate (default: both)."
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate existing collections before ingesting."
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    client = make_client()

    print(f"\nLoading embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"  Model loaded. Vector dim: {EMBED_DIMENSION}\n")

    if args.collection in {"baseline", "both"}:
        ingest(
            client, model,
            collection_name=COLLECTION_BASELINE,
            jsonl_path=BASELINE_JSONL,
            text_field="text",
            recreate=args.recreate,
        )

    if args.collection in {"contextual", "both"}:
        ingest(
            client, model,
            collection_name=COLLECTION_CONTEXTUAL,
            jsonl_path=CONTEXTUAL_JSONL,
            text_field="contextualized_text",
            recreate=args.recreate,
        )

    print("\n[DONE] All requested collections are ready.")


if __name__ == "__main__":
    main()
