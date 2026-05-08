"""
Enrich the dataset with problem descriptions from LeetCode's GraphQL API.

Reads datasets/dataset.csv, extracts the titleSlug from each problem's URL,
fetches the problem body via LeetCode's GraphQL endpoint, and writes
datasets/dataset_enriched.jsonl (one JSON object per line).

Usage:
    pip install requests beautifulsoup4
    python scripts/enrich_dataset.py

The script saves progress incrementally so it can be resumed if interrupted.
"""

import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "datasets" / "dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "dataset_enriched.jsonl"

GRAPHQL_URL = "https://leetcode.com/graphql"
DELAY_BETWEEN_REQUESTS = 2.0

QUESTION_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    content
    hints
  }
}
"""


def extract_title_slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def html_to_text(html: str) -> str:
    """Convert HTML problem body to clean plain text, preserving superscripts and subscripts."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Handle superscripts and subscripts
    for sup in soup.find_all("sup"):
        sup.replace_with(f"^{sup.get_text().strip()}")
    for sub in soup.find_all("sub"):
        sub.replace_with(f"_{sub.get_text().strip()}")
        
    text = soup.get_text(separator="\n")
    
    # Clean up formatting:
    # 1. Remove newlines immediately before or after ^ or _ (common in 10\n^5)
    text = re.sub(r'\n+(\^|_)', r'\1', text)
    text = re.sub(r'(\^|_)\n+', r'\1', text)
    
    # 2. Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_session() -> requests.Session:
    """Create a requests session with a CSRF token from LeetCode."""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Referer": "https://leetcode.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    })
    resp = session.get("https://leetcode.com/problemset/")
    csrf = resp.cookies.get("csrftoken", "")
    if csrf:
        session.headers["x-csrftoken"] = csrf
    return session


def scrape_problem_body(session: requests.Session, title_slug: str) -> str:
    """Fetch problem description from LeetCode GraphQL API."""
    payload = {
        "operationName": "questionData",
        "variables": {"titleSlug": title_slug},
        "query": QUESTION_QUERY,
    }
    try:
        resp = session.post(GRAPHQL_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("data", {}).get("question", {}).get("content")
        if content:
            return html_to_text(content)
        return ""
    except Exception as e:
        print(f"  Failed to scrape '{title_slug}': {e}")
        return ""


def load_already_scraped() -> dict[str, str]:
    """Load slug->body map from the output file if it exists (for resuming)."""
    scraped = {}
    if not OUTPUT_PATH.exists():
        return scraped
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            slug = extract_title_slug(row["link"])
            body = row.get("body", "")
            if body:
                scraped[slug] = body
    return scraped


def _write_output(rows, slug_to_body):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            slug = extract_title_slug(row["link"])
            enriched_row = dict(row)
            enriched_row["body"] = slug_to_body.get(slug, "")
            f.write(json.dumps(enriched_row, ensure_ascii=False) + "\n")


def enrich():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    unique_slugs = {extract_title_slug(row["link"]) for row in rows}
    # Check if a body needs re-scraping (missing exponents or empty)
    def needs_re_scrape(body: str) -> bool:
        if not body: return True
        # Common pattern where exponents were lost: "10 4", "10 5", "10 9"
        if re.search(r'10\s+[459]', body): return True
        # If it has newlines before ^, it's also corrupted
        if '\n^' in body or '\n_' in body: return True
        return False

    already_scraped = load_already_scraped()
    
    # Identify unique slugs that either haven't been scraped or were corrupted
    slug_to_body = {}
    slugs_to_scrape = set()
    
    unique_slugs = {extract_title_slug(row["link"]) for row in rows}
    
    for slug in unique_slugs:
        body = already_scraped.get(slug, "")
        if needs_re_scrape(body):
            slugs_to_scrape.add(slug)
        else:
            slug_to_body[slug] = body

    slugs_to_scrape = sorted(list(slugs_to_scrape))

    print(f"Total rows: {len(rows)}")
    print(f"Unique problems: {len(unique_slugs)}")
    print(f"Already scraped (clean): {len(slug_to_body)}")
    print(f"Remaining (or corrupted) to scrape: {len(slugs_to_scrape)}")

    if not slugs_to_scrape:
        print("Nothing to scrape or fix. Writing final output...")
        _write_output(rows, slug_to_body)
        return

    session = create_session()
    for i, slug in enumerate(slugs_to_scrape, 1):
        print(f"  [{i}/{len(slugs_to_scrape)}] Scraping: {slug}")
        body = scrape_problem_body(session, slug)
        slug_to_body[slug] = body

        if i % 1 == 0:
            _write_output(rows, slug_to_body)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    _write_output(rows, slug_to_body)
    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    enrich()
