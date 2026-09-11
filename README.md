# k8s-rag-eval

**Production-grade RAG system for Kubernetes documentation Q&A**, implementing hybrid retrieval (dense + BM25), Anthropic's Contextual Retrieval technique, cross-encoder reranking, a RAGAS-based evaluation harness, and cost/latency observability.

---

## Overview

This project builds and evaluates a Retrieval-Augmented Generation (RAG) pipeline over the official Kubernetes documentation. It is designed as a rigorous benchmark for comparing retrieval strategies, chunking methods, and generation quality.

### Key Features

| Feature | Detail |
|---|---|
| **Corpus** | ~35 scoped pages from kubernetes.io/docs |
| **Chunking** | Baseline (RecursiveCharacterTextSplitter) + Contextual (Anthropic-augmented) |
| **Retrieval** | Dense (sentence-transformers) + Sparse (BM25) hybrid with RRF fusion |
| **Reranking** | Cross-encoder reranking (ms-marco-MiniLM) |
| **Generation** | Anthropic Claude via langchain-anthropic |
| **Evaluation** | RAGAS metrics: faithfulness, answer relevancy, context precision/recall |
| **Observability** | Cost tracking, latency logging, Streamlit dashboard |

---

## Project Structure

```
k8s-rag-eval/
├── data/
│   ├── raw_docs/          # Scraped K8s doc pages as .txt
│   └── processed/         # Chunked output (JSONL)
├── src/
│   ├── ingestion/
│   │   ├── scrape_k8s_docs.py   # Scrapes kubernetes.io/docs
│   │   └── chunk_documents.py   # Chunking pipeline
│   ├── retrieval/               # Dense + BM25 + reranking
│   ├── generation/              # LLM answer generation
│   ├── evaluation/              # RAGAS harness
│   └── observability/           # Cost + latency tracking
├── dashboard/                   # Streamlit UI
├── eval_set/                    # Golden Q&A pairs
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Scrape Kubernetes docs

```bash
python src/ingestion/scrape_k8s_docs.py
```

### 4. Chunk documents

```bash
python src/ingestion/chunk_documents.py
```

### 5. Contextualize chunks (Anthropic Contextual Retrieval)

```bash
# Test on 20 chunks first to verify quality & estimate cost
python src/ingestion/contextualize_chunks.py --limit 20

# Run on all chunks (requires ANTHROPIC_API_KEY in .env)
python src/ingestion/contextualize_chunks.py
```

### 6. Ingest into Qdrant

```bash
# Start Qdrant locally first: docker run -p 6333:6333 qdrant/qdrant
# Or point QDRANT_URL/.env to a Qdrant Cloud free-tier cluster
python src/retrieval/setup_qdrant.py
```

---

## Contextual Retrieval (Phase 2)

This project implements **Anthropic's Contextual Retrieval** technique as described in their
[November 2024 blog post](https://www.anthropic.com/news/contextual-retrieval).

### The Problem with Standard Chunking

When documents are split into chunks, each chunk loses surrounding context. A chunk
like *"The default value is 30 seconds"* is meaningless without knowing it describes a
liveness probe timeout. This causes retrieval failures — the chunk won't match queries
about liveness probes even though it contains relevant information.

### The Solution: Prepend LLM-Generated Context

For every chunk, we call Claude with the **full document** and the **chunk**, asking:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

The returned blurb (1-2 sentences) is prepended to the chunk before embedding:

```
<context_blurb>\n\n<original_chunk_text>
```

### Prompt Caching for Cost Efficiency

Sending a 60KB document for every chunk of that document would be extremely expensive.
Instead, we use **Anthropic's prompt caching** (`cache_control: ephemeral`) on the
document block. The first chunk of a document incurs a cache *write* cost (~1.25x normal),
but all subsequent chunks for that document are cache *reads* — roughly **10x cheaper**
and significantly faster.

| Token type       | Price per 1M tokens |
|-----------------|---------------------|
| Standard input   | $0.80               |
| Cache write      | $1.00               |
| Cache read       | $0.08               |
| Output           | $1.25               |

### Empirical Impact

Anthropic reports that Contextual Retrieval reduces retrieval failures by **49%** for
standard RAG and **67%** when combined with BM25 hybrid retrieval. Our evaluation harness
(RAGAS) measures this improvement directly on the Kubernetes documentation corpus.


---

## Evaluation

Evaluation uses **RAGAS** with a golden Q&A eval set built from the scraped corpus:

- `faithfulness` – is the answer grounded in the retrieved context?
- `answer_relevancy` – does the answer actually address the question?
- `context_precision` – how much of the retrieved context is relevant?
- `context_recall` – how much of the ground truth is covered by context?

---

## Observability

Every RAG call is logged with:
- Retrieval latency (ms)
- Generation latency (ms)
- Input / output token counts
- Estimated cost (USD)

Results are visualised in a Streamlit dashboard (`dashboard/`).

---

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RAGAS](https://docs.ragas.io/)
- [Qdrant](https://qdrant.tech/documentation/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
