"""
Embedding generation module.
Generates embeddings for hedonic/eudaimonic dimension texts and 200+ clinical scale
descriptions using OpenAI text-embedding-3-large.
Uses ThreadPoolExecutor (max 50 workers) for faster batched embedding.
Caches results — skips if embeddings.json already exists and is current.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BATCH_SIZE = 50  # OpenAI handles up to 2048 inputs, but we chunk for progress
MAX_WORKERS = 50


def load_env():
    repo_root = PROJECT_ROOT.parent.parent.parent.parent
    load_dotenv(repo_root / ".env")


def parse_dimension_file(filepath: Path) -> dict[str, str]:
    dimensions = {}
    current_dim = None
    current_text = []

    for line in filepath.read_text().strip().split("\n"):
        if line.startswith("DIMENSION: "):
            if current_dim is not None:
                dimensions[current_dim] = " ".join(current_text).strip()
            current_dim = line.replace("DIMENSION: ", "").strip()
            current_text = []
        else:
            if line.strip():
                current_text.append(line.strip())

    if current_dim is not None:
        dimensions[current_dim] = " ".join(current_text).strip()

    return dimensions


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


def _embed_batch(batch: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=batch,
    )
    return [item.embedding for item in response.data]


def get_embeddings_parallel(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed texts using ThreadPoolExecutor for concurrent API calls."""
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results = [None] * len(batches)

    def process_batch(idx, batch):
        return idx, _embed_batch(batch, client)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            idx, embeddings = future.result()
            results[idx] = embeddings
            print(f"  Batch {idx + 1}/{len(batches)} done ({len(embeddings)} texts)")

    # Flatten
    all_embeddings = []
    for batch_embs in results:
        all_embeddings.extend(batch_embs)
    return all_embeddings


def run():
    load_env()

    output_path = OUTPUT_DIR / "embeddings.json"
    scales_file = DATA_DIR / "scales" / "clinical_scales_200.txt"

    # Cache check: skip if embeddings exist and scale file hasn't changed
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        existing_count = len(existing.get("scale_embeddings", {}))
        current_scales = parse_scales_file(scales_file)
        if existing_count == len(current_scales):
            print(f"Embeddings cache valid ({existing_count} scales). Skipping.")
            return existing

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    hedonic = parse_dimension_file(DATA_DIR / "dimensions" / "hedonic.txt")
    eudaimonic = parse_dimension_file(DATA_DIR / "dimensions" / "eudaimonic.txt")
    scales = parse_scales_file(scales_file)

    dimension_names = list(hedonic.keys())
    assert dimension_names == list(eudaimonic.keys()), "Dimension mismatch"

    hedonic_texts = [hedonic[d] for d in dimension_names]
    eudaimonic_texts = [eudaimonic[d] for d in dimension_names]
    scale_texts = [s["description"] for s in scales]

    all_texts = hedonic_texts + eudaimonic_texts + scale_texts
    print(f"Embedding {len(all_texts)} texts ({len(hedonic_texts)} hedonic dims, "
          f"{len(eudaimonic_texts)} eudaimonic dims, {len(scale_texts)} scales)...")

    all_embeddings = get_embeddings_parallel(all_texts, client)

    n_h = len(hedonic_texts)
    n_e = len(eudaimonic_texts)

    hedonic_embeddings = {dim: emb for dim, emb in zip(dimension_names, all_embeddings[:n_h])}
    eudaimonic_embeddings = {dim: emb for dim, emb in zip(dimension_names, all_embeddings[n_h:n_h + n_e])}
    scale_embeddings = {s["abbreviation"]: emb for s, emb in zip(scales, all_embeddings[n_h + n_e:])}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "dimension_names": dimension_names,
        "hedonic_embeddings": hedonic_embeddings,
        "eudaimonic_embeddings": eudaimonic_embeddings,
        "scale_embeddings": scale_embeddings,
        "scales_metadata": scales,
    }

    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved embeddings to {output_path}")

    return result


if __name__ == "__main__":
    run()
