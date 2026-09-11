"""
run_eval.py
-----------
Runs evaluation harness across 18 ConfigMaps questions comparing:
  1. Baseline Pipeline   (k8s_baseline + baseline_chunks.jsonl)
  2. Contextual Pipeline (k8s_contextual + contextual_chunks.jsonl)

Uses Claude (claude-haiku-4-5-20251001) for answer generation.
Saves raw output records to:
  - eval_set/results_baseline.jsonl
  - eval_set/results_contextual.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anthropic
from dotenv import load_dotenv

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SET_PATH       = Path("eval_set/configmap_eval.jsonl")
RESULTS_BASELINE   = Path("eval_set/results_baseline.jsonl")
RESULTS_CONTEXTUAL = Path("eval_set/results_contextual.jsonl")

MODEL_NAME = "claude-haiku-4-5-20251001"
TOP_K_RETRIEVAL = 20
TOP_K_RERANK    = 5

SYSTEM_PROMPT = """You are an expert Kubernetes assistant. Answer the user's question based strictly on the provided context documentation chunks below. If the context does not contain enough information, state that clearly. Keep your answer concise, technical, and accurate."""


def load_eval_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def generate_answer(client: anthropic.Anthropic, query: str, context_chunks: list[dict[str, Any]]) -> tuple[str, float]:
    """Build context prompt and call Claude API. Returns (generated_text, latency_sec)."""
    context_blocks = []
    for i, c in enumerate(context_chunks, start=1):
        payload = c.get("payload", c)
        text_content = payload.get("contextualized_text") or payload.get("text") or c.get("text", "")
        chunk_id = c.get("chunk_id", f"chunk_{i}")
        context_blocks.append(f"[Chunk {i}: {chunk_id}]\n{text_content}")

    context_str = "\n\n".join(context_blocks)
    user_content = f"Context:\n---\n{context_str}\n---\n\nQuestion: {query}\n\nAnswer:"

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    latency = time.time() - start_time
    answer_text = response.content[0].text.strip()
    return answer_text, latency


def print_cost_estimate(num_questions: int) -> None:
    num_calls = num_questions * 2
    avg_prompt_tokens = 650
    avg_completion_tokens = 150
    
    total_prompt_tokens = num_calls * avg_prompt_tokens
    total_completion_tokens = num_calls * avg_completion_tokens
    
    # Approx pricing for Haiku: $0.25 / 1M prompt, $1.25 / 1M completion
    est_cost = (total_prompt_tokens / 1_000_000 * 0.25) + (total_completion_tokens / 1_000_000 * 1.25)

    print("=" * 70)
    print("  EVALUATION COST ESTIMATE")
    print("=" * 70)
    print(f"  Questions       : {num_questions}")
    print(f"  Pipelines       : 2 (baseline, contextual)")
    print(f"  Total API Calls : {num_calls}")
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Est. Tokens     : ~{total_prompt_tokens:,} input, ~{total_completion_tokens:,} output")
    print(f"  Estimated Cost  : ~${est_cost:.4f} USD")
    print("=" * 70)
    if est_cost > 0.20:
        print("[COST ALERT] Estimated cost exceeds $0.20 threshold!")
    else:
        print("[COST OK] Estimated cost is well below the $0.20 limit.")
    print("=" * 70 + "\n")


def run_pipeline_for_eval(
    questions: list[dict[str, Any]],
    pipeline_name: str,
    collection_name: str,
    jsonl_path: str,
    output_path: Path,
    anthropic_client: anthropic.Anthropic,
    reranker: Reranker,
) -> None:
    print(f"\n[RUNNING PIPELINE: {pipeline_name.upper()}] ({collection_name} | {jsonl_path})")
    retriever = HybridRetriever(collection_name=collection_name, jsonl_path=jsonl_path)

    results = []
    for idx, item in enumerate(questions, start=1):
        q_id = item["id"]
        q_text = item["question"]
        gt = item["ground_truth"]
        diff = item["difficulty"]

        print(f"  [{idx}/{len(questions)}] ({diff.upper()}) {q_id}: '{q_text[:50]}...'")

        # 1. Retrieve top 20 candidates
        candidates = retriever.retrieve(q_text, top_k=TOP_K_RETRIEVAL)

        # 2. Rerank down to top 5
        top_5_chunks = reranker.rerank(q_text, candidates=candidates, top_k=TOP_K_RERANK)

        # 3. Generate answer
        answer, latency = generate_answer(anthropic_client, q_text, top_5_chunks)

        record = {
            "id": q_id,
            "question": q_text,
            "ground_truth": gt,
            "difficulty": diff,
            "target_topic": item.get("target_topic", ""),
            "pipeline": pipeline_name,
            "retrieved_chunks": top_5_chunks,
            "generated_answer": answer,
            "latency_seconds": round(latency, 3),
        }
        results.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  [SAVED] {len(results)} records written to {output_path}")


def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file.")

    client = anthropic.Anthropic(api_key=api_key)
    questions = load_eval_questions(EVAL_SET_PATH)

    print_cost_estimate(len(questions))

    reranker = Reranker()

    # Pipeline 1: Baseline
    run_pipeline_for_eval(
        questions=questions,
        pipeline_name="baseline",
        collection_name="k8s_baseline",
        jsonl_path="data/processed/baseline_chunks.jsonl",
        output_path=RESULTS_BASELINE,
        anthropic_client=client,
        reranker=reranker,
    )

    # Pipeline 2: Contextual
    run_pipeline_for_eval(
        questions=questions,
        pipeline_name="contextual",
        collection_name="k8s_contextual",
        jsonl_path="data/processed/contextual_chunks.jsonl",
        output_path=RESULTS_CONTEXTUAL,
        anthropic_client=client,
        reranker=reranker,
    )

    print("\n[COMPLETE] Evaluation execution finished for both baseline and contextual pipelines.")


if __name__ == "__main__":
    main()
