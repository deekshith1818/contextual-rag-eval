"""
contextualize_chunks.py
-----------------------
Implements Anthropic's Contextual Retrieval technique.

For each chunk in baseline_chunks.jsonl:
  1. Load the full source document text.
  2. Call Claude (Haiku) with the full doc CACHED + the chunk, asking it to
     produce a short context blurb that situates the chunk in the document.
  3. Prepend the blurb to the original chunk text -> "contextualized_text".
  4. Write to data/processed/contextual_chunks.jsonl.

Prompt caching:
  The whole document is sent as a user-turn block with cache_control so
  Anthropic only charges full-document tokens once per document across all
  its chunks (subsequent cache hits are ~10x cheaper and faster).

Output schema (superset of baseline schema):
{
    "chunk_id":           "...",
    "text":               "<original chunk>",
    "contextualized_text":"<blurb>\\n<original chunk>",
    "context_blurb":      "<blurb>",
    "source_file":        "...",
    "source_url":         "...",
    "page_title":         "...",
    "section_category":   "...",
    "chunk_index":        0,
    "char_start":         0,
    "char_end":           800,
    "chunk_size_chars":   800
}

Usage:
    python src/ingestion/contextualize_chunks.py              # all 1562 chunks
    python src/ingestion/contextualize_chunks.py --limit 20   # quick test
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# ── Windows UTF-8 stdout fix ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RAW_DOCS_DIR   = PROJECT_ROOT / "data" / "raw_docs"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
BASELINE_JSONL = PROCESSED_DIR / "baseline_chunks.jsonl"
OUTPUT_JSONL   = PROCESSED_DIR / "contextual_chunks.jsonl"

# ---------------------------------------------------------------------------
# Model & pricing
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"

# Pricing per million tokens (approximate, as of late 2025)
# Cache write: $0.25 / MTok input  | Cache read: $0.03 / MTok input
# Output:      $1.25 / MTok output
PRICE_INPUT_PER_M       = 0.80   # non-cached input
PRICE_CACHE_WRITE_PER_M = 1.00   # cache write (slightly higher than base)
PRICE_CACHE_READ_PER_M  = 0.08   # cache read (10x cheaper)
PRICE_OUTPUT_PER_M      = 1.25

CONTEXT_PROMPT_TEMPLATE = """\
<document>
{whole_document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""

# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
BASE_BACKOFF = 2.0   # seconds


def call_with_retry(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    """Call client.messages.create with prompt-caching header and exponential backoff.

    In anthropic SDK >= 0.28, prompt caching graduated from beta to GA.
    cache_control blocks still work with the standard messages.create endpoint;
    the beta header is passed via extra_headers to activate cache accounting.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            wait = BASE_BACKOFF ** (attempt + 1)
            tqdm.write(f"  [RATE LIMIT] Waiting {wait:.0f}s before retry {attempt+1}/{MAX_RETRIES}... ({exc})")
            time.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code in {500, 529}:   # server error / overloaded
                wait = BASE_BACKOFF ** (attempt + 1)
                tqdm.write(f"  [SERVER ERR {exc.status_code}] Waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
        except anthropic.APIConnectionError:
            wait = BASE_BACKOFF ** (attempt + 1)
            tqdm.write(f"  [CONNECTION ERR] Waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError(f"Exhausted {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def tokens_to_cost(usage) -> float:
    """Compute USD cost from an Anthropic usage object."""
    inp   = getattr(usage, "input_tokens", 0)
    out   = getattr(usage, "output_tokens", 0)
    cw    = getattr(usage, "cache_creation_input_tokens", 0)
    cr    = getattr(usage, "cache_read_input_tokens", 0)
    # Non-cached input = total input - cache_write - cache_read
    plain = max(inp - cw - cr, 0)
    cost  = (
        plain * PRICE_INPUT_PER_M / 1_000_000
        + cw  * PRICE_CACHE_WRITE_PER_M / 1_000_000
        + cr  * PRICE_CACHE_READ_PER_M  / 1_000_000
        + out * PRICE_OUTPUT_PER_M      / 1_000_000
    )
    return cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Contextualize chunks via Claude Haiku.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N chunks (for testing). Default: all."
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    # ── Load baseline chunks ─────────────────────────────────────────────────
    with BASELINE_JSONL.open(encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    if args.limit:
        all_chunks = all_chunks[: args.limit]
        print(f"[INFO] Running in test mode: processing {len(all_chunks)} chunks.\n")
    else:
        print(f"[INFO] Processing all {len(all_chunks)} chunks.\n")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Per-document document cache (avoid re-reading disk repeatedly) ────────
    doc_text_cache: dict[str, str] = {}

    # ── Accumulators ─────────────────────────────────────────────────────────
    total_cost      = 0.0
    total_input_tok = 0
    total_output_tok= 0
    total_cache_write = 0
    total_cache_read  = 0
    results: list[dict] = []
    t_start = time.perf_counter()

    out_file = OUTPUT_JSONL.open("w", encoding="utf-8")

    try:
        pbar = tqdm(all_chunks, desc="Contextualizing", unit="chunk", dynamic_ncols=True)
        for chunk in pbar:
            source_file = chunk["source_file"]

            # Load document text (cached in memory)
            if source_file not in doc_text_cache:
                doc_path = RAW_DOCS_DIR / source_file
                if not doc_path.exists():
                    tqdm.write(f"  [WARN] Source file not found: {doc_path} — skipping chunk.")
                    continue
                doc_text_cache[source_file] = doc_path.read_text(encoding="utf-8")
            whole_doc = doc_text_cache[source_file]

            chunk_text = chunk["text"]

            # ── Build the prompt with cache_control on the document block ──────
            # The document block is marked for caching; chunk varies per call.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{whole_doc}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Here is the chunk we want to situate within the whole document:\n"
                                f"<chunk>\n{chunk_text}\n</chunk>\n"
                                "Please give a short succinct context to situate this chunk within "
                                "the overall document for the purposes of improving search retrieval "
                                "of the chunk. Answer only with the succinct context and nothing else."
                            ),
                        },
                    ],
                }
            ]

            response = call_with_retry(
                client,
                model=MODEL,
                max_tokens=200,
                messages=messages,
            )

            context_blurb = response.content[0].text.strip()
            contextualized_text = f"{context_blurb}\n\n{chunk_text}"

            # ── Accumulate token usage ─────────────────────────────────────────
            usage = response.usage
            inp   = getattr(usage, "input_tokens", 0)
            out   = getattr(usage, "output_tokens", 0)
            cw    = getattr(usage, "cache_creation_input_tokens", 0)
            cr    = getattr(usage, "cache_read_input_tokens", 0)
            cost  = tokens_to_cost(usage)

            total_input_tok   += inp
            total_output_tok  += out
            total_cache_write += cw
            total_cache_read  += cr
            total_cost        += cost

            # ── Write output record ───────────────────────────────────────────
            record = {
                **chunk,
                "context_blurb":      context_blurb,
                "contextualized_text": contextualized_text,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            # Update progress bar suffix
            pbar.set_postfix(
                cost=f"${total_cost:.4f}",
                cache_hits=total_cache_read,
                refresh=False,
            )

    finally:
        out_file.close()

    elapsed = time.perf_counter() - t_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("CONTEXTUALIZATION COMPLETE")
    print("=" * 64)
    print(f"  Chunks processed    : {len(results)}")
    print(f"  Time taken          : {elapsed:.1f}s  ({elapsed/max(len(results),1):.2f}s/chunk)")
    print(f"  Input tokens        : {total_input_tok:,}")
    print(f"  Cache writes (tok)  : {total_cache_write:,}")
    print(f"  Cache reads  (tok)  : {total_cache_read:,}")
    print(f"  Output tokens       : {total_output_tok:,}")
    print(f"  Total cost (est.)   : ${total_cost:.4f} USD")
    if len(all_chunks) > len(results):
        extrapolated = total_cost / max(len(results), 1) * len(all_chunks)
        print(f"  Extrapolated full cost ({len(all_chunks)} chunks): ${extrapolated:.4f} USD")
    print(f"  Output JSONL        : {OUTPUT_JSONL}")
    print("=" * 64)

    # ── Print 3 before/after examples ────────────────────────────────────────
    sample_indices = [0, len(results) // 2, len(results) - 1] if len(results) >= 3 else list(range(len(results)))
    for i, idx in enumerate(sample_indices):
        r = results[idx]
        sep = "-" * 64
        print(f"\n{sep}")
        print(f"EXAMPLE {i+1}  |  chunk_id: {r['chunk_id']}")
        print(f"Page: {r['page_title']}  [{r['section_category']}]")
        print(sep)
        print("ORIGINAL TEXT (first 300 chars):")
        print(f"  {r['text'][:300].strip()}")
        print()
        print("CONTEXT BLURB (generated by Claude):")
        print(f"  {r['context_blurb']}")
        print()
        print("CONTEXTUALIZED TEXT (first 400 chars):")
        print(f"  {r['contextualized_text'][:400].strip()}")
        print(sep)


if __name__ == "__main__":
    main()
