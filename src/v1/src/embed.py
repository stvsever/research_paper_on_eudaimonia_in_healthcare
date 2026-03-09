"""
Embedding generation module.
Generates embeddings for hedonic/eudaimonic dimension texts and clinical scale descriptions
using OpenAI text-embedding-3-large.
"""

import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_env():
    """Load environment variables from .env at repo root."""
    repo_root = PROJECT_ROOT.parent.parent.parent
    load_dotenv(repo_root / ".env")


def parse_dimension_file(filepath: Path) -> dict[str, str]:
    """Parse a dimension file into {dimension_name: text} pairs."""
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
    """Parse clinical scales file into list of {abbrev, name, description}."""
    scales = []
    for line in filepath.read_text().strip().split("\n"):
        parts = line.split(" | ")
        if len(parts) == 3:
            scales.append({
                "abbreviation": parts[0].strip(),
                "name": parts[1].strip(),
                "description": parts[2].strip(),
            })
    return scales


def get_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Call OpenAI text-embedding-3-large for a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [item.embedding for item in response.data]


def run():
    load_env()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # --- Parse inputs ---
    hedonic = parse_dimension_file(DATA_DIR / "dimensions" / "hedonic.txt")
    eudaimonic = parse_dimension_file(DATA_DIR / "dimensions" / "eudaimonic.txt")
    scales = parse_scales_file(DATA_DIR / "scales" / "clinical_scales.txt")

    dimension_names = list(hedonic.keys())
    assert dimension_names == list(eudaimonic.keys()), "Dimension mismatch"

    # --- Collect all texts for embedding ---
    hedonic_texts = [hedonic[d] for d in dimension_names]
    eudaimonic_texts = [eudaimonic[d] for d in dimension_names]
    scale_texts = [s["description"] for s in scales]

    all_texts = hedonic_texts + eudaimonic_texts + scale_texts
    print(f"Embedding {len(all_texts)} texts ({len(hedonic_texts)} hedonic dims, "
          f"{len(eudaimonic_texts)} eudaimonic dims, {len(scale_texts)} scales)...")

    all_embeddings = get_embeddings(all_texts, client)

    n_h = len(hedonic_texts)
    n_e = len(eudaimonic_texts)

    hedonic_embeddings = {dim: emb for dim, emb in zip(dimension_names, all_embeddings[:n_h])}
    eudaimonic_embeddings = {dim: emb for dim, emb in zip(dimension_names, all_embeddings[n_h:n_h + n_e])}
    scale_embeddings = {s["abbreviation"]: emb for s, emb in zip(scales, all_embeddings[n_h + n_e:])}

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "dimension_names": dimension_names,
        "hedonic_embeddings": hedonic_embeddings,
        "eudaimonic_embeddings": eudaimonic_embeddings,
        "scale_embeddings": scale_embeddings,
        "scales_metadata": scales,
    }

    output_path = OUTPUT_DIR / "embeddings.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved embeddings to {output_path}")

    return result


if __name__ == "__main__":
    run()
