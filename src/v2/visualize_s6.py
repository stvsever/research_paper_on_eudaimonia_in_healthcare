"""
Generate supplementary figure: PubMed publication trends for top-10 hedonic
and top-10 eudaimonic scales — 2×2 layout.
  Row 1 (A, B): raw annual publication counts.
  Row 2 (C, D): baseline-corrected (each scale's count divided by the
                cross-scale mean for that year, giving a ratio relative
                to the "average" scale).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Load cosine similarities and compute mean delta per scale ─────────
cos = pd.read_csv(OUTPUT_DIR / "h1_cosine_similarities_all.csv")
mean_delta = cos.groupby("scale")["delta"].mean().reset_index()
mean_delta.columns = ["scale", "mean_delta"]

# Top 10 hedonic (largest positive delta)
top10_hedonic = mean_delta.nlargest(10, "mean_delta")["scale"].tolist()
# Top 10 eudaimonic (most negative delta)
top10_eudaimonic = mean_delta.nsmallest(10, "mean_delta")["scale"].tolist()

# ── Load usage counts ─────────────────────────────────────────────────
usage = pd.read_csv(OUTPUT_DIR / "usage_counts.csv")

# ── Baseline correction ──────────────────────────────────────────────
# For each year compute the mean count across ALL scales, then express
# each scale as a ratio to that mean.  Values >1 = above-average growth;
# <1 = below-average.  This removes the general upward publication trend.
year_mean = usage.groupby("year")["count"].mean().rename("year_mean")
usage = usage.merge(year_mean, on="year")
usage["corrected"] = usage["count"] / usage["year_mean"].replace(0, np.nan)

# ── Helper ────────────────────────────────────────────────────────────
def _plot(ax, scales, col, ylabel, title):
    for scale in scales:
        sub = usage[usage["scale"] == scale].sort_values("year")
        ax.plot(sub["year"], sub[col], marker="o", markersize=3,
                linewidth=1.4, label=scale)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_xlim(2000, 2025)
    ax.grid(True, alpha=0.3)

# ── 2×2 figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

_plot(axes[0, 0], top10_hedonic,    "count",     "PubMed Publication Count",
      "(A)  Top 10 Hedonic — Raw Counts")
_plot(axes[0, 1], top10_eudaimonic, "count",     "PubMed Publication Count",
      "(B)  Top 10 Eudaimonic — Raw Counts")
_plot(axes[1, 0], top10_hedonic,    "corrected", "Ratio to Cross-Scale Mean",
      "(C)  Top 10 Hedonic — Baseline-Corrected")
_plot(axes[1, 1], top10_eudaimonic, "corrected", "Ratio to Cross-Scale Mean",
      "(D)  Top 10 Eudaimonic — Baseline-Corrected")

# Add a reference line at 1.0 for the corrected panels
for ax in axes[1]:
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", zorder=0)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "s_top10_pubmed_trends.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "s_top10_pubmed_trends.pdf", bbox_inches="tight")
plt.close()

print(f"Saved: {FIGURES_DIR / 's_top10_pubmed_trends.png'}")
print(f"Saved: {FIGURES_DIR / 's_top10_pubmed_trends.pdf'}")
print(f"\nTop 10 Hedonic:    {top10_hedonic}")
print(f"Top 10 Eudaimonic: {top10_eudaimonic}")
