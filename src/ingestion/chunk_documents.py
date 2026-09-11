"""
chunk_documents.py
------------------
Loads every .txt file from data/raw_docs/, chunks it with LangChain's
RecursiveCharacterTextSplitter, and writes all chunks (with metadata) to
data/processed/baseline_chunks.jsonl.

This is our BASELINE (non-contextual) chunking strategy.
We will compare against Anthropic's Contextual Retrieval chunking later.

Output schema (one JSON object per line):
{
    "chunk_id":       "docs__concepts__workloads__pods__#003",
    "text":           "...",
    "source_file":    "docs__concepts__workloads__pods__.txt",
    "source_url":     "https://kubernetes.io/docs/concepts/workloads/pods/",
    "page_title":     "Pods",
    "section_category": "Workloads",
    "chunk_index":    3,
    "char_start":     2400,
    "char_end":       3198,
    "chunk_size_chars": 798
}

Usage:
    python src/ingestion/chunk_documents.py
"""

import csv
import io
import json
import sys
import textwrap
from pathlib import Path

# Reconfigure stdout to UTF-8 so Unicode characters print correctly on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_JSONL = PROCESSED_DIR / "baseline_chunks.jsonl"
METADATA_CSV = RAW_DOCS_DIR / "metadata.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_metadata() -> dict[str, dict]:
    """
    Load metadata.csv into a dict keyed by filename.
    Returns an empty mapping if the CSV does not exist.
    """
    meta: dict[str, dict] = {}
    if not METADATA_CSV.exists():
        print(
            f"  ⚠️  metadata.csv not found at {METADATA_CSV}. "
            "source_url / page_title will be empty."
        )
        return meta
    with METADATA_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = row
    return meta


def build_chunk_id(slug: str, index: int) -> str:
    """Create a deterministic chunk identifier."""
    return f"{slug}#{index:04d}"


def chunk_file(
    txt_path: Path,
    splitter: RecursiveCharacterTextSplitter,
    file_meta: dict,
) -> list[dict]:
    """Split a single .txt file into chunks and return a list of records."""
    text = txt_path.read_text(encoding="utf-8")
    slug = txt_path.stem  # filename without extension

    # LangChain splitter returns a list of strings
    raw_chunks = splitter.split_text(text)

    records = []
    cursor = 0
    for idx, chunk_text in enumerate(raw_chunks):
        # Find where this chunk starts in the original text (approximate).
        # We search from the current cursor to handle overlapping chunks.
        start = text.find(chunk_text[:50], max(0, cursor - CHUNK_OVERLAP))
        if start == -1:
            start = cursor  # fallback
        end = start + len(chunk_text)
        cursor = end

        records.append(
            {
                "chunk_id": build_chunk_id(slug, idx),
                "text": chunk_text,
                "source_file": txt_path.name,
                "source_url": file_meta.get("source_url", ""),
                "page_title": file_meta.get("page_title", ""),
                "section_category": file_meta.get("section_category", ""),
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "chunk_size_chars": len(chunk_text),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    txt_files = sorted(RAW_DOCS_DIR.glob("*.txt"))
    if not txt_files:
        print(
            f"❌  No .txt files found in {RAW_DOCS_DIR}.\n"
            "    Run scrape_k8s_docs.py first."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[dict] = []
    print(f"Chunking {len(txt_files)} files  (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    for txt_path in txt_files:
        file_meta = metadata.get(txt_path.name, {})
        chunks = chunk_file(txt_path, splitter, file_meta)
        all_chunks.extend(chunks)
        category = file_meta.get("section_category", "?")
        print(f"  [OK] {txt_path.name:<55} -> {len(chunks):>3} chunks  [{category}]")

    # ── Write JSONL ────────────────────────────────────────────────────────
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for record in all_chunks:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print("-" * 60)
    print("Chunking complete!")
    print(f"   Files processed : {len(txt_files)}")
    print(f"   Total chunks    : {len(all_chunks)}")
    print(f"   Output          : {OUTPUT_JSONL}")
    print()

    # ── Sample: first 2 chunks ─────────────────────────────────────────────
    if len(all_chunks) >= 1:
        print("=" * 60)
        print("SAMPLE -- Chunk 0")
        print("=" * 60)
        c0 = all_chunks[0]
        print(f"  chunk_id         : {c0['chunk_id']}")
        print(f"  source_file      : {c0['source_file']}")
        print(f"  source_url       : {c0['source_url']}")
        print(f"  page_title       : {c0['page_title']}")
        print(f"  section_category : {c0['section_category']}")
        print(f"  chunk_index      : {c0['chunk_index']}")
        print(f"  char_start       : {c0['char_start']}")
        print(f"  char_end         : {c0['char_end']}")
        print(f"  chunk_size_chars : {c0['chunk_size_chars']}")
        print(f"  text preview     :")
        preview = textwrap.fill(c0["text"][:400], width=72, initial_indent="    ", subsequent_indent="    ")
        print(preview)
        print()

    if len(all_chunks) >= 2:
        print("=" * 60)
        print("SAMPLE -- Chunk 1")
        print("=" * 60)
        c1 = all_chunks[1]
        print(f"  chunk_id         : {c1['chunk_id']}")
        print(f"  source_file      : {c1['source_file']}")
        print(f"  source_url       : {c1['source_url']}")
        print(f"  page_title       : {c1['page_title']}")
        print(f"  section_category : {c1['section_category']}")
        print(f"  chunk_index      : {c1['chunk_index']}")
        print(f"  char_start       : {c1['char_start']}")
        print(f"  char_end         : {c1['char_end']}")
        print(f"  chunk_size_chars : {c1['chunk_size_chars']}")
        print(f"  text preview     :")
        preview = textwrap.fill(c1["text"][:400], width=72, initial_indent="    ", subsequent_indent="    ")
        print(preview)
        print()


if __name__ == "__main__":
    main()
