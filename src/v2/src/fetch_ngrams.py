"""
Google Books Ngram v3 fetcher.

Downloads, streams, and filters English 1-gram data from Google Books Ngram
Viewer (2020 export).  Only adjectives (_ADJ) and verbs (_VERB) are retained,
as these carry evaluative and experiential content most relevant to the
hedonic-eudaimonic distinction (e.g. 'happy', 'anxious', 'meaningful',
'suffer', 'flourish').  Nouns are excluded because they are predominantly
concrete / domain-specific; function words carry no evaluative content.

The v3 data uses Universal POS tags appended to each 1-gram, meaning POS
filtering can be done on the raw text without any NLP dependency.

Efficiency: each of the 24 compressed shards (~1 GB each) is streamed via
urllib + gzip and processed line-by-line, so no full download is ever stored.
Workers operate in parallel via ThreadPoolExecutor.

Output: outputs/ngram_filtered.csv  (columns: word, pos, year, count)
        Only the top-N words by cumulative frequency are kept.
"""

import csv
import gzip
import io
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BASE_URL = (
    "https://storage.googleapis.com/books/ngrams/books/"
    "20200217/eng/1-{shard:05d}-of-00024.gz"
)
N_SHARDS = 24
YEAR_MIN = 1900
YEAR_MAX = 2019
TOP_N = 10_000
MAX_WORKERS = 6
ALLOWED_POS = {"_ADJ", "_VERB"}


def _is_valid_word(word: str) -> bool:
    """Keep only pure-alphabetic words of length >= 3."""
    return len(word) >= 3 and word.isalpha()


def _process_shard(shard_idx: int) -> dict:
    """Stream one .gz shard, filter to ADJ/VERB, return {(word,pos): {year: count}}."""
    url = BASE_URL.format(shard=shard_idx)
    result: dict[tuple, dict] = {}

    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600, context=ctx) as resp:
            # Stream decompression: read in chunks to avoid loading
            # the entire ~1 GB compressed shard into memory at once.
            buf = io.BytesIO()
            while True:
                chunk = resp.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                buf.write(chunk)
            buf.seek(0)
            with gzip.GzipFile(fileobj=buf) as gz:
                for raw_line in gz:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue

                    ngram = parts[0]
                    # v3 format: word_POS\tyear,match_count,vol\tyear,match_count,vol\t...
                    underscore_idx = ngram.rfind("_")
                    if underscore_idx < 0:
                        continue
                    pos_tag = ngram[underscore_idx:]
                    if pos_tag not in ALLOWED_POS:
                        continue

                    word = ngram[:underscore_idx].lower()
                    if not _is_valid_word(word):
                        continue

                    key = (word, pos_tag.lstrip("_"))
                    if key not in result:
                        result[key] = {}

                    # Each subsequent field is "year,match_count,volume_count"
                    for entry in parts[1:]:
                        triplet = entry.split(",")
                        if len(triplet) < 2:
                            continue
                        try:
                            year = int(triplet[0])
                            count = int(triplet[1])
                        except (ValueError, IndexError):
                            continue
                        if year < YEAR_MIN or year > YEAR_MAX:
                            continue
                        result[key][year] = result[key].get(year, 0) + count

    except Exception as e:
        print(f"  [WARN] Shard {shard_idx:05d} failed: {e}")

    return result


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "ngram_filtered.csv"

    # Cache check
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and header == ["word", "pos", "year", "count"]:
                row_count = sum(1 for _ in reader)
                if row_count > 10_000:
                    print(f"Ngram cache valid ({row_count:,} rows). Skipping.")
                    return
        print("Ngram cache exists but seems incomplete. Re-fetching.")

    print(f"Fetching Google Books Ngram v3 data ({N_SHARDS} shards)...")
    print(f"  Filtering: POS ∈ {{ADJ, VERB}}, years {YEAR_MIN}–{YEAR_MAX}, "
          f"alphabetic words ≥ 3 chars")

    # Merge all shards
    merged: dict[tuple, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_shard, i): i for i in range(N_SHARDS)}
        for future in as_completed(futures):
            shard_idx = futures[future]
            try:
                shard_data = future.result()
                for key, year_counts in shard_data.items():
                    if key not in merged:
                        merged[key] = {}
                    for yr, cnt in year_counts.items():
                        merged[key][yr] = merged[key].get(yr, 0) + cnt
                print(f"  Shard {shard_idx:05d} done "
                      f"({len(shard_data):,} word-POS combos)")
            except Exception as e:
                print(f"  [ERR] Shard {shard_idx:05d}: {e}")

    print(f"  Total unique (word, POS) pairs: {len(merged):,}")

    # Rank by cumulative frequency, keep top N
    cumulative = {key: sum(yc.values()) for key, yc in merged.items()}
    top_keys = sorted(cumulative, key=cumulative.get, reverse=True)[:TOP_N]
    top_set = set(top_keys)

    print(f"  Keeping top {len(top_keys):,} by cumulative frequency")
    print(f"  Frequency range: {cumulative[top_keys[0]]:,} – "
          f"{cumulative[top_keys[-1]]:,}")

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "pos", "year", "count"])
        rows_written = 0
        for key in sorted(top_set):
            word, pos = key
            for yr in sorted(merged[key].keys()):
                writer.writerow([word, pos, yr, merged[key][yr]])
                rows_written += 1

    print(f"  Saved {rows_written:,} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    run()
