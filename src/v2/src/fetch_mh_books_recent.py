"""
Augment the mental-health book corpus with more recent books (1980-2020).

The initial fetch yielded only 169 / 62 / 45 / 12 books for the
1980s / 1990s / 2000s / 2010s respectively, creating noise at the tail
of the H2 temporal analysis.  This script:
  1. Uses broader subject terms + title-based search to find more items.
  2. Searches up to 3000 results per decade for recent decades.
  3. Merges new data with the existing mh_books_word_freq.csv.
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

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"

# Broader subject/title queries for recent books
QUERIES = [
    # Original subject-based
    (
        '(subject:"psychology" OR subject:"psychiatry" '
        'OR subject:"mental health" OR subject:"psychotherapy" '
        'OR subject:"clinical psychology" OR subject:"psychopathology" '
        'OR subject:"mental illness" OR subject:"abnormal psychology" '
        'OR subject:"behavioral sciences" OR subject:"psychometrics" '
        'OR subject:"counseling psychology" OR subject:"cognitive therapy" '
        'OR subject:"psychological assessment") '
        'AND mediatype:(texts) AND language:(eng)'
    ),
    # Title-based (catches books without proper subject tags)
    (
        '(title:"psychology" OR title:"psychiatry" '
        'OR title:"mental health" OR title:"psychotherapy" '
        'OR title:"clinical psychology" OR title:"psychopathology" '
        'OR title:"mental illness" OR title:"abnormal psychology") '
        'AND mediatype:(texts) AND language:(eng)'
    ),
    # Additional clinical subjects
    (
        '(subject:"depression" OR subject:"anxiety" '
        'OR subject:"stress" OR subject:"trauma" '
        'OR subject:"personality disorders" OR subject:"schizophrenia" '
        'OR subject:"mood disorders" OR subject:"eating disorders" '
        'OR subject:"addiction" OR subject:"substance abuse" '
        'OR subject:"PTSD" OR subject:"bipolar disorder" '
        'OR subject:"obsessive-compulsive" OR subject:"phobia") '
        'AND mediatype:(texts) AND language:(eng)'
    ),
    # Broader mental health / well-being subjects
    (
        '(subject:"well-being" OR subject:"wellbeing" '
        'OR subject:"positive psychology" OR subject:"resilience" '
        'OR subject:"self-help" OR subject:"cognitive behavioral" '
        'OR subject:"mindfulness" OR subject:"neuroscience" '
        'OR subject:"neuropsychology" OR subject:"child psychology" '
        'OR subject:"developmental psychology" OR subject:"social psychology") '
        'AND mediatype:(texts) AND language:(eng)'
    ),
    # Creator/publisher based
    (
        '(publisher:"American Psychological Association" '
        'OR publisher:"Guilford Press" '
        'OR publisher:"Wiley" '
        'OR publisher:"Springer" '
        'OR publisher:"Cambridge University Press" '
        'OR publisher:"Oxford University Press") '
        'AND (subject:"psychology" OR subject:"psychiatry" OR subject:"mental") '
        'AND mediatype:(texts) AND language:(eng)'
    ),
]

DECADES_TO_AUGMENT = [1980, 1990, 2000, 2010]
MAX_PER_QUERY_DECADE = 500
MIN_BOOK_WORDS = 500
DOWNLOAD_WORKERS = 8
DOWNLOAD_TIMEOUT = 45
TOKENIZE_PATTERN = re.compile(r'\b[a-zA-Z]{3,}\b')


def _make_session():
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def search_decade_query(decade_start, query, max_results, session):
    """Return list of {identifier, year, title} for one decade+query."""
    decade_end = decade_start + 9
    full_query = f"{query} AND year:[{decade_start} TO {decade_end}]"
    books = []
    page = 1

    # Try multiple sort orders to get diverse results
    for sort_order in ["downloads desc", "addeddate desc", "date desc"]:
        if len(books) >= max_results:
            break
        page = 1
        while len(books) < max_results:
            params = {
                "q": full_query,
                "fl[]": ["identifier", "title", "year"],
                "rows": 500,
                "page": page,
                "output": "json",
                "sort[]": sort_order,
            }
            try:
                resp = session.get(IA_SEARCH_URL, params=params, timeout=30)
                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                if not docs:
                    break
                for d in docs:
                    if d.get("year") and d["identifier"] not in seen_ids:
                        books.append(d)
                        seen_ids.add(d["identifier"])
                page += 1
                if len(docs) < 500:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"    Search error {decade_start} sort={sort_order}: {e}")
                break

    return books[:max_results]


def download_and_count(identifier, year, vocab, session):
    """Download book text, tokenize, count vocab words."""
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

    tokens = TOKENIZE_PATTERN.findall(text)
    if len(tokens) < MIN_BOOK_WORDS:
        return None

    counts = defaultdict(int)
    for token in tokens:
        w = token.lower()
        if w in vocab:
            counts[w] += 1

    return dict(counts) if counts else None


def run():
    global seen_ids

    # Load vocabulary
    print("Loading vocabulary from ngram_embeddings.json ...")
    with open(OUTPUT_DIR / "ngram_embeddings.json") as f:
        ngram_data = json.load(f)
    vocab = set(ngram_data["words"])
    print(f"  Vocabulary size: {len(vocab)} words")

    # Load existing metadata to get already-processed identifiers
    existing_meta = {}
    meta_path = OUTPUT_DIR / "mh_books_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)

    # Load existing word-year counts to merge with
    existing_csv = OUTPUT_DIR / "mh_books_word_freq.csv"
    existing_counts = defaultdict(lambda: defaultdict(int))
    if existing_csv.exists():
        with open(existing_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_counts[row["word"]][int(row["year"])] += int(row["count"])
    print(f"  Loaded existing data: {len(existing_counts)} words")

    session = _make_session()
    seen_ids = set()  # track all IDs to avoid duplicate downloads

    # ── Phase 1: Search for more books per decade ──────────────────────
    print("\nSearching for additional mental-health books (recent decades) ...")
    all_new_books = []

    for decade in DECADES_TO_AUGMENT:
        decade_books = []
        for qi, query in enumerate(QUERIES):
            books = search_decade_query(
                decade, query, MAX_PER_QUERY_DECADE, session
            )
            decade_books.extend(books)
            print(f"  {decade}s query {qi+1}/{len(QUERIES)}: "
                  f"found {len(books)} new unique")
            time.sleep(0.5)

        # Deduplicate
        unique = {}
        for b in decade_books:
            if b["identifier"] not in unique:
                unique[b["identifier"]] = b
        all_new_books.extend(unique.values())
        print(f"  {decade}s total unique: {len(unique)}")

    print(f"\n  Total new candidates: {len(all_new_books)}")

    # ── Phase 2: Download & count ──────────────────────────────────────
    print(f"\nDownloading and tokenizing ({DOWNLOAD_WORKERS} workers) ...")
    new_word_year_counts = defaultdict(lambda: defaultdict(int))
    processed = 0
    success = 0
    fail = 0
    books_per_year = defaultdict(int)

    def _process(book):
        return book, download_and_count(
            book["identifier"], book["year"], vocab, session
        )

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_process, b): b for b in all_new_books}
        for future in as_completed(futures):
            processed += 1
            book, counts = future.result()
            if counts:
                year = int(book["year"])
                books_per_year[year] += 1
                for word, count in counts.items():
                    new_word_year_counts[word][year] += count
                success += 1
            else:
                fail += 1
            if processed % 200 == 0:
                print(f"    {processed}/{len(all_new_books)} processed "
                      f"({success} OK, {fail} failed)")

    print(f"\n  Finished: {success} new books processed, {fail} failed")
    for d in DECADES_TO_AUGMENT:
        n = sum(v for y, v in books_per_year.items() if d <= y < d + 10)
        print(f"  {d}s: {n} new books")

    # ── Phase 3: Merge with existing data ──────────────────────────────
    print("\nMerging with existing corpus ...")
    merged = defaultdict(lambda: defaultdict(int))

    # Start with existing
    for word, years_dict in existing_counts.items():
        for year, count in years_dict.items():
            merged[word][year] = count

    # Add new
    for word, years_dict in new_word_year_counts.items():
        for year, count in years_dict.items():
            merged[word][year] += count

    # Save merged
    rows = []
    for word in sorted(merged):
        for year in sorted(merged[word]):
            rows.append({
                "word": word,
                "year": year,
                "count": merged[word][year],
            })

    with open(existing_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "year", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {existing_csv}")

    # Update metadata
    old_processed = existing_meta.get("books_processed", 0)
    old_failed = existing_meta.get("books_failed", 0)

    # Count processed per decade in the MERGED data
    merged_per_decade = {}
    old_per_decade = existing_meta.get("processed_per_decade", {})
    for d in range(1900, 2020, 10):
        old_n = old_per_decade.get(f"{d}s", 0)
        new_n = sum(v for y, v in books_per_year.items() if d <= y < d + 10)
        merged_per_decade[f"{d}s"] = old_n + new_n

    metadata = {
        "total_books_searched": existing_meta.get("total_books_searched", 0) + len(all_new_books),
        "books_processed": old_processed + success,
        "books_failed": old_failed + fail,
        "unique_words": len(merged),
        "total_rows": len(rows),
        "year_range": f"{min(y for yd in merged.values() for y in yd)}-{max(y for yd in merged.values() for y in yd)}",
        "processed_per_decade": merged_per_decade,
        "augmented_recent_decades": {
            f"{d}s": sum(v for y, v in books_per_year.items() if d <= y < d + 10)
            for d in DECADES_TO_AUGMENT
        },
    }
    with open(OUTPUT_DIR / "mh_books_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved updated metadata")


if __name__ == "__main__":
    run()
