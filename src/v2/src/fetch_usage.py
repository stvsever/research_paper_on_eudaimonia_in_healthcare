"""
PubMed usage-data fetcher.
Queries NCBI E-utilities for annual publication counts per clinical scale (2000-2025).
Rate-limited to 3 requests/second (no API key).
Resume-aware: skips already-fetched (scale, year) pairs.
Reads from clinical_scales_200.txt (full 200+ scale set).
"""

import csv
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
YEARS = list(range(2000, 2026))
REQUEST_INTERVAL = 0.34  # ~3 req/sec


def parse_scales_file(filepath: Path) -> list[dict]:
    scales = []
    for line in filepath.read_text().strip().split("\n"):
        parts = line.split(" | ")
        if len(parts) >= 3:
            scales.append({
                "abbreviation": parts[0].strip(),
                "name": parts[1].strip(),
                "description": " | ".join(parts[2:]).strip(),
            })
    return scales


# Common English words / short ambiguous strings that double as scale
# abbreviations — these produce inflated PubMed counts when searched [tiab].
_NON_SPECIFIC_ABBREVS = {
    'ACT', 'AIMS', 'ASSIST', 'AUDIT', 'BRIEF', 'CAGE', 'CORE', 'DES',
    'DOCS', 'FAST', 'MAST', 'BARS', 'SCARED', 'SPIN', 'SRS', 'RAS',
    'MRS', 'SDI', 'PAI', 'PIL', 'WAI', 'AQ', 'ILS', 'ORS', 'SAS',
    'ERQ', 'FIT', 'ACE', 'CSS', 'NPI', 'AIS', 'ASI', 'PCS', 'SDS',
    'SPS', 'DAS', 'FSS', 'MDQ', 'IRI',
}


def _abbrev_is_specific(abbrev: str) -> bool:
    """Return True if the abbreviation is specific enough to search PubMed.

    Abbreviations containing digits or hyphens (e.g. PHQ-9, DASS-21, K10,
    SCL-90-R, HAM-D, CES-D) are always specific.  Long medical acronyms
    (PANSS, STAI, PSQI, HADS, EPDS) are specific too.  Only common English
    words and very short strings (≤2 chars) are excluded.
    """
    import re
    if re.search(r'[\d-]', abbrev):
        return True
    if abbrev.upper() in _NON_SPECIFIC_ABBREVS:
        return False
    if len(abbrev) <= 2:
        return False
    return True


def build_query(scale: dict) -> str:
    name = scale["name"]
    abbrev = scale["abbreviation"]
    if _abbrev_is_specific(abbrev):
        return f'("{name}"[tiab] OR "{abbrev}"[tiab])'
    return f'"{name}"[tiab]'


def fetch_count(query: str, year: int) -> int:
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "mindate": f"{year}/01/01",
        "maxdate": f"{year}/12/31",
        "datetype": "pdat",
        "rettype": "count",
    })
    url = f"{ESEARCH_URL}?{params}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            count_el = root.find("Count")
            return int(count_el.text) if count_el is not None else 0
        except Exception:
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                return 0


def run():
    scales = parse_scales_file(DATA_DIR / "scales" / "clinical_scales_200.txt")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "usage_counts.csv"

    # Resume support: load already-fetched (scale, year) pairs
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["scale"], int(row["year"])))

    total = len(scales) * len(YEARS)
    remaining = total - len(done)
    print(f"Total queries: {total}  |  Already done: {len(done)}  |  Remaining: {remaining}")

    if remaining == 0:
        print("All usage counts already fetched. Skipping.")
        return

    mode = "a" if done else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not done:
            writer.writerow(["scale", "year", "count"])

        completed = 0
        for scale in scales:
            abbrev = scale["abbreviation"]
            query = build_query(scale)
            for year in YEARS:
                if (abbrev, year) in done:
                    continue
                count = fetch_count(query, year)
                writer.writerow([abbrev, year, count])
                f.flush()
                completed += 1
                if completed % 50 == 0 or completed == remaining:
                    print(f"  [{completed}/{remaining}] {abbrev} {year}: {count}")
                time.sleep(0.15)  # Speed up — PubMed allows ~10 req/s with API key

    print(f"Usage counts saved to {out_path}")


if __name__ == "__main__":
    run()
