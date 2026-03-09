"""
H1 Visualization:
 1. v1-style violin + strip plots for ALL scales (1×6 panels)
 2. v1-style violin + strip plots for TOP-50 scales (1×6 panels)
 3. 2×2 bar chart of per-dimension Δ across all 4 sub-analyses
 4. Effect-size comparison bar chart
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

COLOR_HEDONIC = "#E63946"
COLOR_EUDAIMONIC = "#457B9D"

ANALYSES = [
    ("h1a_topN_raw",       "A — Top-50 (raw)"),
    ("h1b_topN_weighted",  "B — Top-50 (usage-weighted)"),
    ("h1c_all_raw",        "C — All scales (raw)"),
    ("h1d_all_weighted",   "D — All scales (usage-weighted)"),
]


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── v1-style violin + strip helpers ──────────────────────────────────

def _draw_violin_panel(ax, subset, dim, stat_row, idx, dot_size=18,
                       weights=None):
    """Draw v1-style violin + strip for one dimension on one Axes."""
    h_vals = subset["sim_hedonic"].values
    e_vals = subset["sim_eudaimonic"].values
    scales = subset["scale"].values

    # Violin
    parts = ax.violinplot(
        [h_vals, e_vals], positions=[0, 1],
        showmeans=False, showmedians=False, showextrema=False, widths=0.7,
    )
    for i, pc in enumerate(parts["bodies"]):
        c = COLOR_HEDONIC if i == 0 else COLOR_EUDAIMONIC
        pc.set_facecolor(c)
        pc.set_alpha(0.25)
        pc.set_edgecolor(c)
        pc.set_linewidth(1.2)

    # Strip (jittered dots with density-based alpha)
    rng = np.random.default_rng(42 + idx)
    for i, (vals, color) in enumerate([(h_vals, COLOR_HEDONIC), (e_vals, COLOR_EUDAIMONIC)]):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        median_val = np.median(vals)
        distances = np.abs(vals - median_val)
        max_dist = distances.max() if distances.max() > 0 else 1
        alphas = 0.35 + 0.55 * (1 - distances / max_dist)
        for v, j, a in zip(vals, jitter, alphas):
            ax.scatter(i + j, v, color=color, alpha=a, s=dot_size,
                       edgecolors="white", linewidth=0.3, zorder=3)

    # Mean diamonds (weighted or unweighted)
    if weights is not None:
        w = np.array([weights.get(s, 0) for s in scales])
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
        mean_h = np.average(h_vals, weights=w_norm)
        mean_e = np.average(e_vals, weights=w_norm)
    else:
        mean_h = np.mean(h_vals)
        mean_e = np.mean(e_vals)
    for i, m in enumerate([mean_h, mean_e]):
        ax.scatter(i, m, color="white", s=55, zorder=5,
                   edgecolors="black", linewidth=1.4, marker="D")

    # Significance bracket
    p_col = "p_adjusted" if "p_adjusted" in stat_row.index else "wilcoxon_p"
    p_val = stat_row[p_col]
    sig = significance_label(p_val)
    delta_col = "mean_delta_weighted" if (weights is not None and "mean_delta_weighted" in stat_row.index) else "mean_delta"
    delta = stat_row[delta_col]
    y_max = max(h_vals.max(), e_vals.max())
    bracket_y = y_max + 0.012
    bracket_h = 0.004
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
            color="black", linewidth=1.0, clip_on=False)
    ax.text(0.5, bracket_y + bracket_h + 0.004, f"Δ={delta:+.3f}  {sig}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hedonic", "Eudaimonic"], fontsize=8, fontweight="medium")
    ax.set_title(dim, fontsize=9, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.2, linestyle="--")


def _make_violin_figure(df_sim, dim_stats_df, overall, title, filename,
                        n_scales, dot_size=18, weights=None):
    """Create a 1×6 violin + strip figure (v1 style)."""
    dimensions = dim_stats_df["dimension"].tolist()
    n_dims = len(dimensions)

    fig, axes = plt.subplots(1, n_dims, figsize=(3.4 * n_dims, 7), sharey=True)
    if n_dims == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for idx, dim in enumerate(dimensions):
        subset = df_sim[df_sim["dimension"] == dim]
        stat_row = dim_stats_df[dim_stats_df["dimension"] == dim].iloc[0]
        _draw_violin_panel(axes[idx], subset, dim, stat_row, idx,
                           dot_size=dot_size, weights=weights)

    axes[0].set_ylabel(
        "Cosine Similarity  (scale · dimension embedding)",
        fontsize=10, fontweight="medium",
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HEDONIC, alpha=0.4, label="Hedonic"),
        mpatches.Patch(facecolor=COLOR_EUDAIMONIC, alpha=0.4, label="Eudaimonic"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black",
               markerfacecolor="white", markersize=7,
               label="Weighted Mean" if weights else "Mean"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="gray",
               markerfacecolor="gray", markersize=5, alpha=0.6,
               label="Individual scale"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               frameon=True, fancybox=True, shadow=False, borderpad=0.8,
               bbox_to_anchor=(0.5, -0.04))

    sig_text = (
        f"N = {n_scales} scales   |   "
        f"Overall Δ = {overall['mean_delta']:+.4f}   d = {overall['cohen_d']:.2f}"
        f"   p = {overall['wilcoxon_p']:.2e}\n"
        "* p < .05   ** p < .01   *** p < .001   n.s. = not significant  |  "
        "Wilcoxon signed-rank (one-sided)"
    )
    fig.text(0.98, -0.06, sig_text, ha="right", va="top",
             fontsize=7, color="#777777")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{filename}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── bar chart helpers ────────────────────────────────────────────────

def _draw_bar_panel(ax, dim_df, overall, title):
    """Horizontal bar chart of per-dimension Δ for one sub-analysis."""
    dims = dim_df["dimension"].tolist()
    n = len(dims)
    y_pos = np.arange(n)

    d_col = "cohen_d_weighted" if "cohen_d_weighted" in dim_df.columns else "cohen_d"
    delta_col = "mean_delta_weighted" if "mean_delta_weighted" in dim_df.columns else "mean_delta"

    vals = dim_df[delta_col].values
    colors = [COLOR_HEDONIC if v > 0 else COLOR_EUDAIMONIC for v in vals]

    ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5,
            height=0.65)

    for i, (_, row) in enumerate(dim_df.iterrows()):
        sig = significance_label(row["wilcoxon_p"])
        d_val = row[d_col]
        label = f" d={d_val:.2f}  {sig}"
        ax.text(vals[i] + 0.001 * np.sign(vals[i]), i, label,
                va="center", fontsize=7,
                ha="left" if vals[i] >= 0 else "right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dims, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean Δ  (hedonic − eudaimonic)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle="--")

    o_delta = overall.get("mean_delta_weighted", overall["mean_delta"])
    o_d = overall.get("cohen_d_weighted", overall["cohen_d"])
    o_p = overall["wilcoxon_p"]
    ax.set_title(
        f"{title}\nOverall Δ={o_delta:+.4f}  d={o_d:.2f}  "
        f"{significance_label(o_p)}",
        fontsize=9, fontweight="bold",
    )


# ── main ─────────────────────────────────────────────────────────────

def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load similarity data
    df_sim = pd.read_csv(OUTPUT_DIR / "h1_cosine_similarities_all.csv")
    top_n_df = pd.read_csv(OUTPUT_DIR / "h1_top_n_scales.csv")
    top_n_set = set(top_n_df["scale"])

    # Load stats for all 4 analyses
    analysis_data = {}
    for label, _ in ANALYSES:
        dim_df = pd.read_csv(OUTPUT_DIR / f"{label}_dim_stats.csv")
        with open(OUTPUT_DIR / f"{label}_overall.json") as f:
            overall = json.load(f)
        analysis_data[label] = (dim_df, overall)

    # ===== 1. v1-style violin: ALL scales =====
    _make_violin_figure(
        df_sim,
        analysis_data["h1c_all_raw"][0],
        analysis_data["h1c_all_raw"][1],
        "H1 — Semantic Proximity of Most Common Clinical Scales\n"
        "to Hedonic vs. Eudaimonic Well-Being Dimensions",
        "h1_violin_all",
        n_scales=df_sim["scale"].nunique(),
        dot_size=14,
    )
    print(f"Saved H1 violin (all scales) to {FIGURES_DIR}")

    # ===== 1b. v1-style violin: ALL scales (usage-weighted mean) =====
    weights_df = pd.read_csv(OUTPUT_DIR / "h1_usage_weights.csv")
    all_weights = dict(zip(weights_df["scale"], weights_df["weight"]))
    _make_violin_figure(
        df_sim,
        analysis_data["h1d_all_weighted"][0],
        analysis_data["h1d_all_weighted"][1],
        "H1 — Semantic Proximity of Most Common Clinical Scales\n"
        "to Hedonic vs. Eudaimonic Dimensions (Usage-Weighted Mean)",
        "h1_violin_all_weighted",
        n_scales=df_sim["scale"].nunique(),
        dot_size=14,
        weights=all_weights,
    )
    print(f"Saved H1 violin weighted (all scales) to {FIGURES_DIR}")

    # ===== 2. v1-style violin: TOP-50 =====
    df_top = df_sim[df_sim["scale"].isin(top_n_set)]
    _make_violin_figure(
        df_top,
        analysis_data["h1a_topN_raw"][0],
        analysis_data["h1a_topN_raw"][1],
        "H1 — Semantic Proximity of 50 Most-Used Clinical Scales\n"
        "to Hedonic vs. Eudaimonic Well-Being Dimensions",
        "h1_violin_top50",
        n_scales=len(top_n_set),
        dot_size=22,
    )
    print(f"Saved H1 violin (top-50) to {FIGURES_DIR}")

    # ===== 2b. v1-style violin: TOP-50 (usage-weighted mean) =====
    top_n_weights = {s: all_weights.get(s, 0) for s in top_n_set}
    _make_violin_figure(
        df_top,
        analysis_data["h1b_topN_weighted"][0],
        analysis_data["h1b_topN_weighted"][1],
        "H1 — Semantic Proximity of 50 Most-Used Clinical Scales\n"
        "to Hedonic vs. Eudaimonic Dimensions (Usage-Weighted Mean)",
        "h1_violin_top50_weighted",
        n_scales=len(top_n_set),
        dot_size=22,
        weights=top_n_weights,
    )
    print(f"Saved H1 violin weighted (top-50) to {FIGURES_DIR}")

    # ===== 3. 2×2 bar chart =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    for idx, (label, title) in enumerate(ANALYSES):
        r, c = divmod(idx, 2)
        dim_df, overall = analysis_data[label]
        _draw_bar_panel(axes[r][c], dim_df, overall, title)

    fig.suptitle("H1 — Hedonic Bias Across Four Sub-Analyses",
                 fontsize=14, fontweight="bold", y=1.01)
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HEDONIC, alpha=0.7,
                       label="Hedonic-leaning (Δ > 0)"),
        mpatches.Patch(facecolor=COLOR_EUDAIMONIC, alpha=0.7,
                       label="Eudaimonic-leaning (Δ < 0)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.98, -0.04,
             "* p < .05   ** p < .01   *** p < .001   n.s. = not significant\n"
             "Wilcoxon signed-rank test (one-sided: Hedonic > Eudaimonic)",
             ha="right", va="top", fontsize=7, color="#777777")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"h1_four_panel.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved H1 four-panel figure to {FIGURES_DIR}")

    # ===== 4. Effect-size comparison =====
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor("white")

    dims = analysis_data["h1a_topN_raw"][0]["dimension"].tolist()
    n_dims = len(dims)
    x = np.arange(n_dims)
    w = 0.18

    labels_short = ["Top-50 raw", "Top-50 wt.", "All raw", "All wt."]
    bar_colors = ["#264653", "#2A9D8F", "#E76F51", "#E63946"]

    for i, (label, _) in enumerate(ANALYSES):
        dim_df = analysis_data[label][0]
        d_col = ("cohen_d_weighted"
                 if "cohen_d_weighted" in dim_df.columns else "cohen_d")
        ax2.bar(x + (i - 1.5) * w, dim_df[d_col].values, w,
                color=bar_colors[i], alpha=0.8, label=labels_short[i])

    ax2.set_xticks(x)
    ax2.set_xticklabels(dims, fontsize=8, rotation=25, ha="right")
    ax2.set_ylabel("Cohen's d", fontsize=10)
    ax2.set_title("H1 — Effect Size Comparison Across Sub-Analyses",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(FIGURES_DIR / f"h1_effect_sizes.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved H1 effect-size comparison to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
