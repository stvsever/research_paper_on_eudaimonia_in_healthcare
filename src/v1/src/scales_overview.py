"""
Scales overview figure - simple circular arrangement.
Creates a clean circle figure with all 30 clinical scale names for reference.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"


def run():
    with open(OUTPUT_DIR / "embeddings.json") as f:
        data = json.load(f)

    scales_meta = data["scales_metadata"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Extract abbreviations
    scale_abbrevs = [s["abbreviation"] for s in scales_meta]
    n_scales = len(scale_abbrevs)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")

    # Arrange scales in a clean circle
    angles = np.linspace(0, 2 * np.pi, n_scales, endpoint=False)
    radius = 3.3

    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)

    # Plot outer circle boundary (solid)
    outer_circle = plt.Circle((0, 0), 3.6, fill=False, linestyle="-", 
                              linewidth=2.0, edgecolor="#457B9D", alpha=0.5)
    ax.add_patch(outer_circle)

    # Add scale names - all equal size with ellipse boxes
    for i, (abbrev, x, y) in enumerate(zip(scale_abbrevs, x_positions, y_positions)):
        ax.text(x, y, abbrev, ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4F8", 
                         edgecolor="#457B9D", linewidth=1.0, alpha=0.85),
                zorder=2)

    # Center label - larger, no box
    ax.text(0, 0, "30 Clinical Scales", ha="center", va="center", fontsize=16,
            fontweight="bold", color="#457B9D")

    # Axis formatting
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.set_title("Clinical Scales Included in Semantic Analysis",
                 fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()

    out_path = FIGURES_DIR / "clinical_scales_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    out_pdf = FIGURES_DIR / "clinical_scales_overview.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Scales overview figure saved: {out_path}")
    print(f"Scales overview figure saved: {out_pdf}")


if __name__ == "__main__":
    run()
