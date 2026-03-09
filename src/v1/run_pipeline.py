"""
Main pipeline runner.
Executes the full research pipeline: embed → analyze → visualize.
"""

from src.embed import run as embed
from src.analyze import run as analyze
from src.visualize import run as visualize


def main():
    print("=" * 72)
    print("STEP 1 / 3 — Generating embeddings (OpenAI text-embedding-3-large)")
    print("=" * 72)
    embed()

    print("\n" + "=" * 72)
    print("STEP 2 / 3 — Computing cosine similarities & statistical tests")
    print("=" * 72)
    analyze()

    print("\n" + "=" * 72)
    print("STEP 3 / 3 — Generating research figure")
    print("=" * 72)
    visualize()

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
