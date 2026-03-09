"""
Visualization module.
Produces a research-grade figure: violin + strip plots for cosine similarity
distributions per dimension, with significance bars.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def run():
    df = pd.read_csv(OUTPUT_DIR / "cosine_similarities.csv")
    df_stats = pd.read_csv(OUTPUT_DIR / "dimension_statistics.csv")
    with open(OUTPUT_DIR / "overall_statistics.json") as f:
        overall = json.load(f)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Reshape for plotting: long format with paradigm column
    df_long = pd.concat([
        df[["scale", "dimension", "cosine_sim_hedonic"]].rename(
            columns={"cosine_sim_hedonic": "cosine_similarity"}
        ).assign(paradigm="Hedonic"),
        df[["scale", "dimension", "cosine_sim_eudaimonic"]].rename(
            columns={"cosine_sim_eudaimonic": "cosine_similarity"}
        ).assign(paradigm="Eudaimonic"),
    ], ignore_index=True)

    dimensions = df_stats["dimension"].tolist()
    n_dims = len(dimensions)

    # --- Color palette ---
    color_hedonic = "#E63946"     # warm red
    color_eudaimonic = "#457B9D"  # steel blue
    colors = {"Hedonic": color_hedonic, "Eudaimonic": color_eudaimonic}

    # --- Figure setup ---
    fig, axes = plt.subplots(1, n_dims, figsize=(3.4 * n_dims, 7), sharey=True)
    if n_dims == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        subset = df_long[df_long["dimension"] == dim]

        # Violin plot
        parts = ax.violinplot(
            [subset[subset["paradigm"] == "Hedonic"]["cosine_similarity"].values,
             subset[subset["paradigm"] == "Eudaimonic"]["cosine_similarity"].values],
            positions=[0, 1],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.7,
        )

        for i, pc in enumerate(parts["bodies"]):
            c = color_hedonic if i == 0 else color_eudaimonic
            pc.set_facecolor(c)
            pc.set_alpha(0.25)
            pc.set_edgecolor(c)
            pc.set_linewidth(1.2)

        # Strip plot (raw data with jitter and gradient-like alpha)
        for i, (paradigm, color) in enumerate(colors.items()):
            vals = subset[subset["paradigm"] == paradigm]["cosine_similarity"].values
            jitter = np.random.default_rng(42 + idx + i).uniform(-0.12, 0.12, size=len(vals))

            # Density-based alpha: points closer to median are more opaque
            median_val = np.median(vals)
            distances = np.abs(vals - median_val)
            max_dist = distances.max() if distances.max() > 0 else 1
            alphas = 0.35 + 0.55 * (1 - distances / max_dist)

            for v, j, a in zip(vals, jitter, alphas):
                ax.scatter(i + j, v, color=color, alpha=a, s=28, edgecolors="white",
                           linewidth=0.4, zorder=3)

        # Mean markers
        for i, paradigm in enumerate(["Hedonic", "Eudaimonic"]):
            vals = subset[subset["paradigm"] == paradigm]["cosine_similarity"].values
            mean_val = np.mean(vals)
            ax.scatter(i, mean_val, color="white", s=60, zorder=5, edgecolors="black", linewidth=1.5, marker="D")

        # Significance bracket
        row = df_stats[df_stats["dimension"] == dim].iloc[0]
        p_val = row["wilcoxon_p"]
        sig = significance_label(p_val)

        y_max = subset["cosine_similarity"].max()
        bracket_y = y_max + 0.012
        bracket_h = 0.004

        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                color="black", linewidth=1.0, clip_on=False)

        # Combined delta + significance on the bracket
        delta = row["mean_delta"]
        ax.text(0.5, bracket_y + bracket_h + 0.004, f"Δ={delta:+.3f}  {sig}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")

        # Axis formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Hedonic", "Eudaimonic"], fontsize=9, fontweight="medium")
        ax.set_title(dim, fontsize=10, fontweight="bold", pad=22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    axes[0].set_ylabel("Cosine Similarity  (scale embedding · dimension embedding)", fontsize=10, fontweight="medium")

    # --- Suptitle ---
    fig.suptitle(
        "Semantic Proximity of 30 Major Clinical Scales\nto Hedonic vs. Eudaimonic Well-Being Dimensions",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_hedonic, alpha=0.4, label="Hedonic"),
        mpatches.Patch(facecolor=color_eudaimonic, alpha=0.4, label="Eudaimonic"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black", markerfacecolor="white",
               markersize=7, label="Mean"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="gray", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Individual scale"),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               frameon=True, fancybox=True, shadow=False, borderpad=0.8,
               bbox_to_anchor=(0.5, -0.04))

    # Significance legend
    fig.text(0.98, -0.06, "* p < .05   ** p < .01   *** p < .001   n.s. = not significant\nWilcoxon signed-rank test (one-sided: Hedonic > Eudaimonic)",
             ha="right", va="top", fontsize=7, color="#777777")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = FIGURES_DIR / "hedonic_eudaimonic_semantic_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    out_pdf = FIGURES_DIR / "hedonic_eudaimonic_semantic_analysis.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figure saved: {out_path}")
    print(f"Figure saved: {out_pdf}")


if __name__ == "__main__":
    run()
