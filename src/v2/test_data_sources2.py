"""Test: find downloadable (public domain) mental health books on Internet Archive."""
import requests
import re
from collections import Counter

# Search specifically for downloadable books
# Try different filters for public domain / downloadable content
print("=== Search for downloadable mental health books ===")
url = "https://archive.org/advancedsearch.php"

# Strategy: search with rights field, or older books that are more likely public domain
queries = [
    # Public domain filter attempts
    ('rights:public + subject', 
     '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health") '
     'AND mediatype:(texts) AND language:(eng) AND rights:(public*)'),
    
    # Pre-1929 (definitely public domain in US)
    ('pre-1929 psychology books',
     '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health" '
     'OR subject:"psychotherapy") AND mediatype:(texts) AND language:(eng) '
     'AND year:[1900 TO 1928]'),
    
    # All years, include "open" in collection
    ('all years accessible',
     '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health" '
     'OR subject:"psychotherapy") AND mediatype:(texts) AND language:(eng) '
     'AND NOT lending___status:is_lendable'),
]

for label, q in queries:
    try:
        params = {"q": q, "rows": 0, "output": "json"}
        resp = requests.get(url, params=params, timeout=15)
        n = resp.json().get("response", {}).get("numFound", 0)
        print(f"  {label}: {n} results")
    except Exception as e:
        print(f"  {label}: Error - {e}")

# Get some pre-1929 books and try downloading text
print("\n=== Try downloading pre-1929 books ===")
q_old = (
    '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health" '
    'OR subject:"psychotherapy" OR subject:"abnormal psychology") '
    'AND mediatype:(texts) AND language:(eng) AND year:[1900 TO 1928]'
)
params = {
    "q": q_old,
    "fl[]": ["identifier", "title", "year"],
    "rows": 20,
    "output": "json",
    "sort[]": "downloads desc",
}
resp = requests.get(url, params=params, timeout=15)
docs = resp.json().get("response", {}).get("docs", [])

success = 0
fail = 0
for d in docs[:20]:
    ident = d.get("identifier", "")
    year = d.get("year", "?")
    title = d.get("title", "?")[:60]
    
    # Try _djvu.txt
    text_url = f"https://archive.org/download/{ident}/{ident}_djvu.txt"
    try:
        r = requests.head(text_url, timeout=10, allow_redirects=True)
        if r.status_code == 200:
            # Actually download a small piece
            r2 = requests.get(text_url, timeout=15, headers={"Range": "bytes=0-500"})
            text_preview = r2.text[:100].replace("\n", " ")
            print(f"  OK  {year} | {ident} | {title}")
            print(f"       Preview: {text_preview}")
            success += 1
        else:
            print(f"  {r.status_code} {year} | {ident} | {title}")
            fail += 1
    except Exception as e:
        print(f"  ERR {year} | {ident} | {title} | {e}")
        fail += 1

print(f"\nDownloadable: {success}/{success+fail}")

# Check year coverage for downloadable books
print("\n=== Year coverage for NOT-restricted books ===")
q_base = (
    '(subject:"psychology" OR subject:"psychiatry" OR subject:"mental health" '
    'OR subject:"psychotherapy" OR subject:"abnormal psychology" OR '
    'subject:"clinical psychology" OR subject:"mental illness") '
    'AND mediatype:(texts) AND language:(eng) '
    'AND NOT lending___status:is_lendable'
)
for decade_start in [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]:
    decade_end = decade_start + 9
    q2 = q_base + f" AND year:[{decade_start} TO {decade_end}]"
    try:
        resp2 = requests.get(url, params={"q": q2, "rows": 0, "output": "json"}, timeout=15)
        n = resp2.json().get("response", {}).get("numFound", 0)
        print(f"  {decade_start}-{decade_end}: {n} books")
    except:
        print(f"  {decade_start}-{decade_end}: error")
