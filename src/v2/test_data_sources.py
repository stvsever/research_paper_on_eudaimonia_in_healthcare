"""Quick test: check Internet Archive availability for mental health books."""
import requests
import json

# Test 1: Search for mental health books
print("=== TEST 1: Search for mental health books ===")
query = (
    '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health" '
    'OR subject:"psychotherapy" OR subject:"clinical psychology") '
    'AND mediatype:(texts) AND language:(eng OR English)'
)
url = "https://archive.org/advancedsearch.php"
params = {
    "q": query,
    "fl[]": ["identifier", "title", "year", "subject"],
    "rows": 5,
    "page": 1,
    "output": "json",
}
try:
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    total = data.get("response", {}).get("numFound", 0)
    docs = data.get("response", {}).get("docs", [])
    print(f"Total results: {total}")
    for d in docs[:5]:
        print(f"  {d.get('year', '?')} | {d.get('identifier', '?')} | {d.get('title', '?')[:80]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Check year distribution
print("\n=== TEST 2: Year distribution ===")
for decade_start in [1900, 1920, 1940, 1960, 1980, 2000, 2010]:
    decade_end = decade_start + 9
    q2 = query + f" AND year:[{decade_start} TO {decade_end}]"
    try:
        resp2 = requests.get(url, params={"q": q2, "rows": 0, "output": "json"}, timeout=15)
        n = resp2.json().get("response", {}).get("numFound", 0)
        print(f"  {decade_start}-{decade_end}: {n} books")
    except Exception as e:
        print(f"  {decade_start}-{decade_end}: Error - {e}")

# Test 3: Try downloading text from one book
print("\n=== TEST 3: Download test ===")
if docs:
    test_id = docs[0].get("identifier", "")
    text_url = f"https://archive.org/download/{test_id}/{test_id}_djvu.txt"
    try:
        resp3 = requests.get(text_url, timeout=30, stream=True)
        if resp3.status_code == 200:
            # Read first 500 chars
            text = resp3.text[:500]
            print(f"  Downloaded text for '{test_id}' ({len(resp3.text)} chars total)")
            print(f"  Preview: {text[:200]}...")
        else:
            print(f"  HTTP {resp3.status_code} for {test_id}")
            # Try alternative
            alt_url = f"https://archive.org/download/{test_id}/{test_id}.txt"
            resp4 = requests.get(alt_url, timeout=30)
            print(f"  Alt URL status: {resp4.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

# Test 4: Check HathiTrust EF API accessibility
print("\n=== TEST 4: HathiTrust Extracted Features API ===")
try:
    # Try the HTRC EF API
    htrc_url = "https://data.analytics.hathitrust.org/extracted-features/"
    resp5 = requests.head(htrc_url, timeout=10)
    print(f"  HTRC EF base: HTTP {resp5.status_code}")
except Exception as e:
    print(f"  HTRC EF: {e}")

try:
    # Try the HTRC catalog search
    htrc_cat = "https://catalog.hathitrust.org/api/volumes/brief/oclc/12345678.json"
    resp6 = requests.get(htrc_cat, timeout=10)
    print(f"  HTRC catalog API: HTTP {resp6.status_code}")
except Exception as e:
    print(f"  HTRC catalog: {e}")
