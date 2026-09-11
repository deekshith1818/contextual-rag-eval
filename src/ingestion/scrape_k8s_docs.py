"""
scrape_k8s_docs.py
------------------
Scrapes a scoped subset of the official Kubernetes documentation
(https://kubernetes.io/docs/) and saves each page as a .txt file
in data/raw_docs/.  Also writes a metadata CSV.

Scope (~35 pages):
  - Concepts > Workloads
  - Concepts > Services, Load Balancing, and Networking
  - Concepts > Configuration
  - Concepts > Storage
  - Concepts > Security
  - Tasks > a handful of common task pages

Usage:
    python src/ingestion/scrape_k8s_docs.py
"""

import csv
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://kubernetes.io"

# (url_path, section_category)
PAGES = [
    # ── Workloads ──────────────────────────────────────────────────────────
    ("/docs/concepts/workloads/", "Workloads"),
    ("/docs/concepts/workloads/pods/", "Workloads"),
    ("/docs/concepts/workloads/pods/pod-lifecycle/", "Workloads"),
    ("/docs/concepts/workloads/pods/init-containers/", "Workloads"),
    ("/docs/concepts/workloads/controllers/deployment/", "Workloads"),
    ("/docs/concepts/workloads/controllers/replicaset/", "Workloads"),
    ("/docs/concepts/workloads/controllers/statefulset/", "Workloads"),
    ("/docs/concepts/workloads/controllers/daemonset/", "Workloads"),
    ("/docs/concepts/workloads/controllers/job/", "Workloads"),
    ("/docs/concepts/workloads/controllers/cron-jobs/", "Workloads"),
    # ── Services, Load Balancing, and Networking ───────────────────────────
    ("/docs/concepts/services-networking/service/", "Networking"),
    ("/docs/concepts/services-networking/ingress/", "Networking"),
    ("/docs/concepts/services-networking/ingress-controllers/", "Networking"),
    ("/docs/concepts/services-networking/dns-pod-service/", "Networking"),
    ("/docs/concepts/services-networking/network-policies/", "Networking"),
    ("/docs/concepts/services-networking/endpoint-slices/", "Networking"),
    # ── Configuration ──────────────────────────────────────────────────────
    ("/docs/concepts/configuration/configmap/", "Configuration"),
    ("/docs/concepts/configuration/secret/", "Configuration"),
    ("/docs/concepts/configuration/manage-resources-containers/", "Configuration"),
    ("/docs/concepts/storage/volumes/", "Storage"),
    ("/docs/concepts/storage/persistent-volumes/", "Storage"),
    ("/docs/concepts/storage/storage-classes/", "Storage"),
    ("/docs/concepts/storage/dynamic-provisioning/", "Storage"),
    # -- Security ---------------------------------------------------------------
    ("/docs/concepts/security/rbac-good-practices/", "Security"),
    ("/docs/reference/access-authn-authz/rbac/", "Security"),
    ("/docs/concepts/security/service-accounts/", "Security"),
    ("/docs/concepts/security/pod-security-standards/", "Security"),
    ("/docs/concepts/security/pod-security-admission/", "Security"),
    # -- Tasks ------------------------------------------------------------------
    ("/docs/tasks/configure-pod-container/configure-pod-configmap/", "Tasks"),
    ("/docs/tasks/configure-pod-container/assign-pods-nodes/", "Tasks"),
    ("/docs/tasks/configure-pod-container/assign-memory-resource/", "Tasks"),
    ("/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/", "Tasks"),
    ("/docs/tasks/configure-pod-container/configure-service-account/", "Tasks"),
    ("/docs/tasks/run-application/run-stateless-application-deployment/", "Tasks"),
]

RAW_DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_docs"
METADATA_CSV = RAW_DOCS_DIR / "metadata.csv"
REQUEST_DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; k8s-rag-eval-scraper/1.0; "
        "+https://github.com/your-org/k8s-rag-eval)"
    )
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def url_to_slug(url_path: str) -> str:
    """Convert a URL path to a safe filename slug."""
    slug = url_path.strip("/").replace("/", "__")
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    return slug or "index"


def extract_main_content(html: str) -> tuple[str, str]:
    """
    Parse the HTML and return (title, body_text).

    The k8s docs site wraps the main article in <div id="page-content-wrapper">
    or a <main> / <article> element.  We try several selectors in order.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Extract title ──────────────────────────────────────────────────────
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(separator=" ", strip=True)
    if not title:
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "").strip()
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    # ── Remove boilerplate elements ────────────────────────────────────────
    for selector in [
        "nav", "header", "footer",
        "[class*='sidebar']", "[class*='toc']",
        "[id*='sidebar']", "[id*='toc']",
        "[class*='feedback']", "[class*='edit-page']",
        "script", "style", "noscript",
    ]:
        for el in soup.select(selector):
            el.decompose()

    # ── Find the main content container ───────────────────────────────────
    content_el = (
        soup.find("div", id="page-content-wrapper")
        or soup.find("div", {"class": re.compile(r"td-content|content__body")})
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )

    if content_el is None:
        return title, ""

    # Extract text preserving paragraph breaks
    lines = []
    for element in content_el.descendants:
        if element.name in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "code", "td", "th"}:
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)

    body_text = "\n\n".join(lines)
    # Collapse excessive blank lines
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)
    return title, body_text


def fetch_page(url: str) -> str | None:
    """Fetch a URL and return the response HTML, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    pages_scraped = 0

    print(f"[START] Scraping {len(PAGES)} pages planned")
    print(f"   Output directory: {RAW_DOCS_DIR}\n")

    for url_path, category in PAGES:
        full_url = BASE_URL + url_path
        slug = url_to_slug(url_path)
        out_file = RAW_DOCS_DIR / f"{slug}.txt"

        print(f"[{pages_scraped + 1:02d}/{len(PAGES)}] Fetching: {full_url}")

        html = fetch_page(full_url)
        if html is None:
            print(f"         -> Skipped (fetch error)\n")
            continue

        title, body_text = extract_main_content(html)

        if not body_text.strip():
            print(f"         -> Skipped (no content extracted)\n")
            continue

        # Write raw text
        out_file.write_text(body_text, encoding="utf-8")

        # Record metadata
        metadata_rows.append(
            {
                "filename": out_file.name,
                "source_url": full_url,
                "page_title": title,
                "section_category": category,
            }
        )

        pages_scraped += 1
        chars = len(body_text)
        print(f"         -> Saved '{title}' ({chars:,} chars) -> {out_file.name}\n")

        time.sleep(REQUEST_DELAY)

    # ── Write metadata CSV ─────────────────────────────────────────────────
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "source_url", "page_title", "section_category"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("-" * 60)
    print("Scraping complete!")
    print(f"   Pages scraped : {pages_scraped} / {len(PAGES)} attempted")
    print(f"   Output dir    : {RAW_DOCS_DIR}")
    print(f"   Metadata CSV  : {METADATA_CSV}")


if __name__ == "__main__":
    main()
