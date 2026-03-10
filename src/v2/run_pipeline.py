"""
Main pipeline runner for v2.
Executes: fetch_usage → embed → analyze_h1 → analyze_h2 → analyze_posthoc
         → visualize_h1 → visualize_h2 → visualize_domains
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch_usage import run as fetch_usage
from src.embed import run as embed
from src.analyze_h1 import run as analyze_h1
from src.analyze_h2 import run as analyze_h2
from src.analyze_posthoc import run as analyze_posthoc
from src.visualize_h1 import run as visualize_h1
from src.visualize_h2 import run as visualize_h2
from src.visualize_domains import run as visualize_domains
from src.fetch_ngrams import run as fetch_ngrams
from src.embed_ngrams import run as embed_ngrams
from src.analyze_ngrams_h1 import run as analyze_ngrams_h1
from src.analyze_ngrams_h2 import run as analyze_ngrams_h2
from src.visualize_ngrams import run as visualize_ngrams


STEPS = [
    ("1/13", "Fetching PubMed usage counts for 200+ scales (2000-2025)", fetch_usage),
    ("2/13", "Generating embeddings (OpenAI text-embedding-3-large, parallel)", embed),
    ("3/13", "H1 analysis: 4 sub-analyses (top-N raw/weighted, all raw/weighted)", analyze_h1),
    ("4/13", "H2 analysis: per-dimension temporal trend", analyze_h2),
    ("5/13", "Post-hoc: domains, permutation, sensitivity, scatter", analyze_posthoc),
    ("6/13", "H1 visualisation (2×2 panel + effect sizes)", visualize_h1),
    ("7/13", "H2 visualisation (6-line dimension plot)", visualize_h2),
    ("8/13", "Post-hoc visualisation (domains, permutation, sensitivity, scatter)", visualize_domains),
    ("9/13", "Fetching Google Books Ngram v3 data (ADJ/VERB, 1900-2019)", fetch_ngrams),
    ("10/13", "Embedding top-10K ngram words (text-embedding-3-large)", embed_ngrams),
    ("11/13", "Ngram H1 replication (usage-weighted)", analyze_ngrams_h1),
    ("12/13", "Ngram H2 replication (temporal trend, 1900-2019)", analyze_ngrams_h2),
    ("13/13", "Ngram visualisation (supplementary figures)", visualize_ngrams),
]


def main():
    for label, desc, fn in STEPS:
        print("\n" + "=" * 72)
        print(f"STEP {label} — {desc}")
        print("=" * 72)
        fn()

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
