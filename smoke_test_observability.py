"""
smoke_test_observability.py
---------------------------
Phase 6 smoke test: runs 8 diverse queries through BOTH the baseline and
contextual pipelines using run_with_generation(), prints results, then
invokes the summarize_logs CLI to validate captured log data.

Run from project root:
    python smoke_test_observability.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# ── Windows UTF-8 stdout ──────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Silence HF symlink warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from src.retrieval.pipeline import make_baseline_pipeline, make_contextual_pipeline

# ---------------------------------------------------------------------------
# Test queries (spread across topics in the corpus)
# ---------------------------------------------------------------------------
QUERIES = [
    # ConfigMaps
    "How do I mount specific keys from a ConfigMap as files in a pod?",
    "What is the difference between a ConfigMap env var and a volume mount?",
    # Workloads
    "What happens to pods when a Deployment rollout fails?",
    "How does a DaemonSet ensure one pod per node?",
    # Networking
    "What is the difference between ClusterIP and NodePort services?",
    "How do Ingress controllers route traffic to backend services?",
    # Storage
    "What is the lifecycle of a PersistentVolumeClaim?",
    # Security
    "What are the three Pod Security Standards and what do they restrict?",
]

MODEL = "claude-haiku-4-5-20251001"
W    = 72


def _sep(char: str = "─") -> None:
    print(f"  {char * W}")


def main() -> None:
    print(f"\n{'═' * (W + 4)}")
    print("  Phase 6 Smoke Test — Observability Logging")
    print(f"  Model      : {MODEL}")
    print(f"  Queries    : {len(QUERIES)} × 2 pipelines = {len(QUERIES) * 2} LLM calls")
    print(f"{'═' * (W + 4)}\n")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env

    print("  Loading pipelines (BM25 index + embedding model warm-up)...")
    baseline_pipe   = make_baseline_pipeline()
    contextual_pipe = make_contextual_pipeline()
    print(f"  Baseline   : {baseline_pipe.retriever.num_chunks:,} chunks in BM25 index")
    print(f"  Contextual : {contextual_pipe.retriever.num_chunks:,} chunks in BM25 index")
    print()

    total_cost = 0.0

    for q_num, query in enumerate(QUERIES, start=1):
        print(f"  [{q_num}/{len(QUERIES)}] {query}")
        _sep()

        for pipe, label in [(baseline_pipe, "baseline"), (contextual_pipe, "contextual")]:
            result = pipe.run_with_generation(
                query            = query,
                anthropic_client = client,
                model            = MODEL,
                log              = True,
            )
            total_cost += result["cost_usd"]
            print(f"  [{label:>10}]  "
                  f"ret={result['retrieval_ms']:>6.0f}ms  "
                  f"rnk={result['rerank_ms']:>5.0f}ms  "
                  f"gen={result['generation_ms']:>6.0f}ms  "
                  f"tok={result['input_tokens']}+{result['output_tokens']}  "
                  f"${result['cost_usd']:.5f}")
            answer_preview = result["answer"][:120].replace("\n", " ")
            print(f"              → {answer_preview}…")

        print()

    print(f"{'═' * (W + 4)}")
    print(f"  DONE  |  {len(QUERIES) * 2} queries  |  estimated total cost: ${total_cost:.4f} USD")
    print(f"{'═' * (W + 4)}\n")

    # ── Run summarize_logs to verify captured data ────────────────────────────
    print("  Running summarize_logs.py to verify log capture...\n")
    from src.observability.summarize_logs import main as summarize_main
    summarize_main()


if __name__ == "__main__":
    main()
