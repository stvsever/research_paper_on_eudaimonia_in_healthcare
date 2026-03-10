"""
Embedding module for Google Books Ngram words.

Embeds the top-N adjectives/verbs from the filtered ngram data using
OpenAI text-embedding-3-large (same model as the main clinical-scale
pipeline for direct comparability).

Reuses dimension embeddings from embeddings.json (already computed by
the main pipeline) and only embeds the word list.

Output: outputs/ngram_embeddings.json
  {
    "words": [...],
    "word_embeddings": {"word": [float, ...], ...},
  }
"""

import csv
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BATCH_SIZE = 2048  # Max OpenAI batch
MAX_WORKERS = 10


def _load_env():
    # Try multiple possible .env locations
    candidates = [
        PROJECT_ROOT / ".env",                          # src/v2/.env
        PROJECT_ROOT.parent / ".env",                   # src/.env
        PROJECT_ROOT.parent.parent / ".env",            # research_paper/.env
        PROJECT_ROOT.parent.parent.parent / ".env",     # repo root/.env
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            return
    load_dotenv()  # fallback: search current dir + parents


def _embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [item.embedding for item in response.data]


def run():
    _load_env()

    out_path = OUTPUT_DIR / "ngram_embeddings.json"
    ngram_csv = OUTPUT_DIR / "ngram_filtered.csv"

    # Load unique words from filtered ngram data
    words_set: set[str] = set()
    with open(ngram_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            words_set.add(row["word"])
    words = sorted(words_set)

    # Cache check
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if len(existing.get("word_embeddings", {})) == len(words):
            print(f"Ngram embeddings cache valid ({len(words)} words). Skipping.")
            return existing

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it via .env file or "
            "export OPENAI_API_KEY=sk-... in your shell."
        )
    client = OpenAI(api_key=api_key)

    print(f"Embedding {len(words)} ngram words...")
    batches = [words[i:i + BATCH_SIZE] for i in range(0, len(words), BATCH_SIZE)]
    all_embeddings = []

    for i, batch in enumerate(batches):
        embs = _embed_batch(batch, client)
        all_embeddings.extend(embs)
        print(f"  Batch {i + 1}/{len(batches)} done ({len(batch)} words)")

    word_embeddings = {w: emb for w, emb in zip(words, all_embeddings)}

    result = {
        "words": words,
        "word_embeddings": word_embeddings,
    }

    out_path.write_text(json.dumps(result))
    print(f"Saved ngram embeddings to {out_path} ({len(words)} words)")
    return result


if __name__ == "__main__":
    run()
