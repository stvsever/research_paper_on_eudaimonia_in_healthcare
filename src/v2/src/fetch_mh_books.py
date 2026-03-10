"""
Fetch word frequencies from Internet Archive mental-health books.

Strategy
--------
1. Search Internet Archive for English-language psychology / psychiatry /
   mental-health books, filtered to non-restricted (downloadable) items.
2. Stratify by decade (1900s–2010s) to ensure temporal coverage.
3. Download OCR text (_djvu.txt) in parallel.
4. Tokenize and count occurrences of the 9,288 words already embedded
   (from the Google Books Ngram pipeline).
5. Aggregate by (word, year) and save as ``mh_books_word_freq.csv``.

This produces a frequency dataset analogous to ``ngram_filtered.csv``
but drawn exclusively from mental-health literature, enabling a
book-level (rather than word-level) domain filter.

Output
------
- outputs/mh_books_word_freq.csv   (word, year, count)
- outputs/mh_books_metadata.json   (search & processing statistics)
"""

import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Search configuration ──────────────────────────────────────────────

SUBJECT_QUERY = (
    '(subject:"psychology" OR subject:"psychiatry" '
    'OR subject:"mental health" OR subject:"psychotherapy" '
    'OR subject:"clinical psychology" OR subject:"psychopathology" '
    'OR subject:"mental illness" OR subject:"abnormal psychology" '
    'OR subject:"behavioral sciences" OR subject:"psychometrics" '
    'OR subject:"counseling psychology" OR subject:"cognitive therapy" '
    'OR subject:"psychological assessment")'
)

BASE_QUERY = (
    f'{SUBJECT_QUERY} AND mediatype:(texts) AND language:(eng) '
    'AND NOT lending___status:is_lendable'
)

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"

DECADES = list(range(1900, 2020, 10))
MAX_PER_DECADE = 500          # books to fetch per decade
MIN_BOOK_WORDS = 500          # skip very short texts (pamphlets)
DOWNLOAD_WORKERS = 8          # parallel download threads
DOWNLOAD_TIMEOUT = 45         # seconds per request
TOKENIZE_PATTERN = re.compile(r'\b[a-zA-Z]{3,}\b')


# ── Helpers ────────────────────────────────────────────────────────────

def _make_session():
    """Requests session with retry."""
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def search_decade(decade_start, session):
    """Return list of {identifier, year, title} for one decade."""
    decade_end = decade_start + 9
    query = f"{BASE_QUERY} AND year:[{decade_start} TO {decade_end}]"
    books = []
    page = 1
    while len(books) < MAX_PER_DECADE:
        params = {
            "q": query,
            "fl[]": ["identifier", "title", "year"],
            "rows": min(500, MAX_PER_DECADE - len(books)),
            "page": page,
            "output": "json",
            "sort[]": "downloads desc",
        }
        try:
            resp = session.get(IA_SEARCH_URL, params=params, timeout=30)
            data = resp.json()
            docs = data.get("response", {}).get("docs", [])
            if not docs:
                break
            books.extend(docs)
            page += 1
            if len(docs) < 500:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"    Search error decade {decade_start}: {e}")
            break
    return books[:MAX_PER_DECADE]


def download_and_count(identifier, year, vocab, session):
    """Download book text, tokenize, count vocab words. Returns dict."""
    urls = [
        f"https://archive.org/download/{identifier}/{identifier}_djvu.txt",
        f"https://archive.org/download/{identifier}/{identifier}.txt",
    ]
    text = None
    for url in urls:
        try:
            resp = session.get(url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code == 200 and len(resp.text) > 200:
                text = resp.text
                break
        except Exception:
            continue

    if text is None:
        return None

    # Tokenize: extract alphabetic words ≥3 chars, lowercase
    tokens = TOKENIZE_PATTERN.findall(text)
    total_tokens = len(tokens)
    if total_tokens < MIN_BOOK_WORDS:
        return None

    # Count only words in our vocabulary
    counts = defaultdict(int)
    for token in tokens:
        w = token.lower()
        if w in vocab:
            counts[w] += 1

    return dict(counts) if counts else None


# ── Main ───────────────────────────────────────────────────────────────

def run():
    # Load vocabulary (words we have embeddings for)
    print("Loading vocabulary from ngram_embeddings.json ...")
    with open(OUTPUT_DIR / "ngram_embeddings.json") as f:
        ngram_data = json.load(f)
    vocab = set(ngram_data["words"])
    print(f"  Vocabulary size: {len(vocab)} words")

    session = _make_session()

    # ── Phase 1: Search ────────────────────────────────────────────────
    print("\nSearching Internet Archive for mental-health books ...")
    all_books = []
    for decade in DECADES:
        books = search_decade(decade, session)
        all_books.extend(
            {"identifier": b["identifier"],
             "year": int(b["year"]),
             "title": b.get("title", "")}
            for b in books if b.get("year")
        )
        print(f"  {decade}s: {len(books)} books found")

    print(f"  Total: {len(all_books)} books to process")

    # ── Phase 2: Download & count ──────────────────────────────────────
    print(f"\nDownloading and tokenizing ({DOWNLOAD_WORKERS} workers) ...")
    word_year_counts = defaultdict(lambda: defaultdict(int))
    processed = 0
    success = 0
    fail = 0
    books_per_year = defaultdict(int)

    def _process(book):
        return book, download_and_count(
            book["identifier"], book["year"], vocab, session
        )

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_process, b): b for b in all_books}
        for future in as_completed(futures):
            processed += 1
            book, counts = future.result()
            if counts:
                year = book["year"]
                books_per_year[year] += 1
                for word, count in counts.items():
                    word_year_counts[word][year] += count
                success += 1
            else:
                fail += 1
            if processed % 200 == 0:
                print(f"    {processed}/{len(all_books)} processed "
                      f"({success} OK, {fail} failed)")

    print(f"\n  Finished: {success} books processed, {fail} failed")
    print(f"  Unique words with counts: {len(word_year_counts)}")
    print(f"  Year coverage: {min(books_per_year)} - {max(books_per_year)}")

    # ── Phase 3: Save ──────────────────────────────────────────────────
    rows = []
    for word in sorted(word_year_counts):
        for year in sorted(word_year_counts[word]):
            rows.append({
                "word": word,
                "year": year,
                "count": word_year_counts[word][year],
            })

    output_csv = OUTPUT_DIR / "mh_books_word_freq.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "year", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {output_csv}")

    # Save metadata
    metadata = {
        "total_books_searched": len(all_books),
        "books_processed": success,
        "books_failed": fail,
        "unique_words": len(word_year_counts),
        "total_rows": len(rows),
        "year_range": f"{min(books_per_year)}-{max(books_per_year)}",
        "books_per_decade": {
            f"{d}s": sum(1 for b in all_books
                         if d <= b["year"] < d + 10)
            for d in DECADES
        },
        "processed_per_decade": {
            f"{d}s": sum(v for y, v in books_per_year.items()
                         if d <= y < d + 10)
            for d in DECADES
        },
    }
    with open(OUTPUT_DIR / "mh_books_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {OUTPUT_DIR / 'mh_books_metadata.json'}")


if __name__ == "__main__":
    run()
