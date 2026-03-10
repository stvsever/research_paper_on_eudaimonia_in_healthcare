"""
Supplementary visualisation for the mental-health book replication.

Produces two figures matching the style of the main clinical-scale plots:

1. ngram_h1_violin.pdf  — 1×6 violin + strip plot panels (H1 replication)
2. ngram_h2_combined.pdf — 1×2 panel (A smoothed raw, B decadal means)
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
from scipy.stats import t as t_dist
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

COLOR_HEDONIC = "#E63946"
COLOR_EUDAIMONIC = "#457B9D"

DIM_COLORS = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#457B9D",
]
COLOR_AGG = "#2B2D42"
COLOR_CI = "#8D99AE"

RECENT_YEARS = range(2000, 2020)  # broader window for book corpus coverage


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ══════════════════════════════════════════════════════════════════════
# H1 VIOLIN PLOT
# ══════════════════════════════════════════════════════════════════════

def _draw_violin_panel(ax, subset, dim, stat_row, idx, weights):
    h_vals = subset["sim_hedonic"].values
    e_vals = subset["sim_eudaimonic"].values
    word_list = subset["word"].values

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

    # Strip (jittered dots with density-based alpha) — subsample for speed
    rng = np.random.default_rng(42 + idx)
    max_dots = 400
    n_total = len(h_vals)
    if n_total > max_dots:
        sample_idx = rng.choice(n_total, max_dots, replace=False)
    else:
        sample_idx = np.arange(n_total)
    for i, (vals, color) in enumerate(
        [(h_vals, COLOR_HEDONIC), (e_vals, COLOR_EUDAIMONIC)]
    ):
        subset_vals = vals[sample_idx]
        jitter = rng.uniform(-0.12, 0.12, size=len(subset_vals))
        median_val = np.median(vals)
        distances = np.abs(subset_vals - median_val)
        max_dist = distances.max() if distances.max() > 0 else 1
        alphas = 0.35 + 0.55 * (1 - distances / max_dist)
        for v, j, a in zip(subset_vals, jitter, alphas):
            ax.scatter(i + j, v, color=color, alpha=float(a), s=6,
                       edgecolors="white", linewidth=0.15, zorder=3)

    # Weighted mean diamonds
    w = np.array([weights.get(word, 0) for word in word_list])
    w_sum = w.sum()
    w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
    mean_h = np.average(h_vals, weights=w_norm)
    mean_e = np.average(e_vals, weights=w_norm)
    for i, m in enumerate([mean_h, mean_e]):
        ax.scatter(i, m, color="white", s=55, zorder=5,
                   edgecolors="black", linewidth=1.4, marker="D")

    # Significance bracket
    p_val = stat_row["p_adjusted"]
    sig = significance_label(p_val)
    delta = stat_row["mean_delta_weighted"]
    y_max = max(h_vals.max(), e_vals.max())
    bracket_y = y_max + 0.012
    bracket_h = 0.004
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + bracket_h,
             bracket_y + bracket_h, bracket_y],
            color="black", linewidth=1.0, clip_on=False)
    ax.text(0.5, bracket_y + bracket_h + 0.004,
            f"Δ={delta:+.3f}  {sig}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hedonic", "Eudaimonic"], fontsize=8,
                       fontweight="medium")
    ax.set_title(dim, fontsize=9, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.2, linestyle="--")


def _make_ngram_h1_violin(df_sim, dim_df, overall, n_words, weights):
    dimensions = dim_df["dimension"].tolist()
    n_dims = len(dimensions)

    fig, axes = plt.subplots(1, n_dims, figsize=(3.4 * n_dims, 7),
                             sharey=True)
    if n_dims == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for idx, dim in enumerate(dimensions):
        subset = df_sim[df_sim["dimension"] == dim]
        stat_row = dim_df[dim_df["dimension"] == dim].iloc[0]
        _draw_violin_panel(axes[idx], subset, dim, stat_row, idx, weights)

    axes[0].set_ylabel(
        "Cosine Similarity  (word · dimension embedding)",
        fontsize=10, fontweight="medium",
    )
    fig.suptitle(
        "Mental-Health Book Replication — Hedonic Bias in "
        "Adjective/Verb Frequency (Usage-Weighted)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HEDONIC, alpha=0.4,
                       label="Hedonic"),
        mpatches.Patch(facecolor=COLOR_EUDAIMONIC, alpha=0.4,
                       label="Eudaimonic"),
        Line2D([0], [0], marker="D", color="w",
               markeredgecolor="black", markerfacecolor="white",
               markersize=7, label="Freq-Weighted Mean"),
        Line2D([0], [0], marker="o", color="w",
               markeredgecolor="gray", markerfacecolor="gray",
               markersize=5, alpha=0.6, label="Individual word"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, frameon=True, fancybox=True, shadow=False,
               borderpad=0.8, bbox_to_anchor=(0.5, -0.04))

    sig_text = (
        f"N = {n_words:,} words   |   "
        f"Overall Δw = {overall['mean_delta_weighted']:+.4f}   "
        f"dw = {overall['cohen_d_weighted']:.2f}"
        f"   p = {overall['wilcoxon_p']:.2e}\n"
        "* p < .05   ** p < .01   *** p < .001   n.s. = not significant  "
        "|  Wilcoxon signed-rank (one-sided)"
    )
    fig.text(0.98, -0.06, sig_text, ha="right", va="top",
             fontsize=7, color="#777777")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


# ══════════════════════════════════════════════════════════════════════
# H2 COMBINED PANEL
# ══════════════════════════════════════════════════════════════════════

def _make_ngram_h2_combined(df_yearly, df_trends, summary, dim_cols):
    years = df_yearly["year"].values
    n_dims = len(dim_cols)

    dim_matrix = np.column_stack([df_yearly[d].values for d in dim_cols])
    agg_vals = df_yearly["aggregate"].values.copy()

    # Smoothing window (centered rolling average)
    SMOOTH_WIN = 11

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")

    # ── Panel A: Raw yearly data with smoothed trend + OLS line ──────
    ax = axes[0]

    # CI ribbon from cross-dimension variability
    t_crit = t_dist.ppf(0.975, df=n_dims - 1)
    means = np.mean(dim_matrix, axis=1)
    ses = np.std(dim_matrix, axis=1, ddof=1) / np.sqrt(n_dims)
    ci_lo = means - t_crit * ses
    ci_hi = means + t_crit * ses
    ax.fill_between(years, ci_lo, ci_hi, color=COLOR_CI, alpha=0.15,
                    zorder=1)

    # Individual dimension lines — very faint
    for i, dim in enumerate(dim_cols):
        color = DIM_COLORS[i % len(DIM_COLORS)]
        ax.plot(years, dim_matrix[:, i], "-", color=color,
                linewidth=0.5, alpha=0.25, zorder=2)

    # Raw aggregate as small transparent dots
    ax.scatter(years, agg_vals, color=COLOR_AGG, s=8, alpha=0.25,
               zorder=3, edgecolors="none")

    # Smoothed aggregate (centered rolling average)
    agg_smooth = uniform_filter1d(agg_vals, size=SMOOTH_WIN, mode="nearest")
    ax.plot(years, agg_smooth, "-", color=COLOR_AGG, linewidth=2.5,
            zorder=6, label=f"Aggregate ({SMOOTH_WIN}-yr smooth)")

    # OLS regression line
    agg_trend = df_trends[df_trends["dimension"] == "aggregate"].iloc[0]
    ols_y = agg_trend["intercept"] + agg_trend["slope"] * years.astype(float)
    agg_sig = significance_label(agg_trend["p_ols"])
    ax.plot(years, ols_y, "--", color="#C0392B", linewidth=2.0, zorder=7,
            label=f"OLS trend  (β={agg_trend['slope']:+.2e}/yr {agg_sig})")

    # Annotations
    mk_p = summary.get("mann_kendall_p", 1)
    mk_p_str = f"p = {mk_p:.4f}" if mk_p >= 0.0001 else "p < .0001"
    anno = (
        f"R² = {summary.get('r_squared', 0):.3f}   "
        f"Mann-Kendall S = {summary.get('mann_kendall_S', 0):.0f},  "
        f"{mk_p_str}"
    )
    ax.text(0.98, 0.03, anno, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#333",
            fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))

    ax.text(-0.03, 1.04, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax.set_title("Yearly Values with Smoothed Trend", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Freq-Weighted Δ (hedonic − eudaimonic)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, frameon=True, fancybox=True)

    # ── Panel B: Baseline-corrected decadal means with CI error bars ──
    ax2 = axes[1]

    # Compute decadal means and SEs of the aggregate
    df_tmp = df_yearly.copy()
    df_tmp["decade"] = (df_tmp["year"] // 10) * 10
    dec_groups = df_tmp.groupby("decade")["aggregate"]
    dec_means = dec_groups.mean()
    dec_ses = dec_groups.std() / np.sqrt(dec_groups.count())
    dec_years = dec_means.index.values + 5  # mid-decade

    # Baseline-correct: subtract the first decade's mean
    baseline = dec_means.iloc[0]
    dec_bc = dec_means.values - baseline

    # Error bars (95% CI) on baseline-corrected values
    ax2.errorbar(dec_years, dec_bc,
                 yerr=1.96 * dec_ses.values,
                 fmt="o-", color=COLOR_AGG, linewidth=2.5, markersize=9,
                 capsize=6, capthick=1.5, markeredgecolor="white",
                 markeredgewidth=1.2, zorder=5,
                 label="Decadal mean ± 95% CI (baseline-corrected)")

    # OLS regression line (baseline-corrected)
    dec_x = np.linspace(years[0], years[-1], 100)
    dec_ols_y = (agg_trend["intercept"] + agg_trend["slope"] * dec_x) - baseline
    ax2.plot(dec_x, dec_ols_y, "--", color="#C0392B", linewidth=2.0,
             zorder=4, label=f"OLS trend  (β={agg_trend['slope']:+.2e}/yr)")

    # Zero baseline
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5,
                zorder=1)

    # Cumulative drift annotation
    total_shift = agg_trend["slope"] * (years[-1] - years[0])
    ax2.annotate(
        f"Cumulative shift: {total_shift:+.4f}\nover {int(years[-1]-years[0])} years",
        xy=(dec_years[-1], dec_bc[-1]),
        xytext=(1935, max(dec_bc) * 0.92),
        fontsize=8.5, fontweight="medium", color="#C0392B",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#C0392B", alpha=0.85),
        zorder=8)

    # Per-dimension baseline-corrected decadal means
    for i, dim in enumerate(dim_cols):
        dim_dec = df_tmp.groupby("decade")[dim].mean()
        dim_bc = dim_dec.values - dim_dec.iloc[0]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        trend = df_trends[df_trends["dimension"] == dim].iloc[0]
        sig = significance_label(trend["p_ols"])
        ax2.plot(dim_dec.index.values + 5, dim_bc, "o-",
                 color=color, linewidth=0.9, alpha=0.45, markersize=3,
                 label=f"{dim}  (β={trend['slope']:+.2e}/yr {sig})",
                 zorder=3)

    ax2.text(-0.03, 1.04, "B", transform=ax2.transAxes,
             fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax2.set_title("Baseline-Corrected Decadal Means", fontsize=11,
                 fontweight="bold")
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Change in Freq-Weighted Δ from 1900s", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")

    # Shared legend below both panels
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles2, labels2, loc="lower center", fontsize=7.5,
               ncol=4, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Mental-Health Book Replication — Temporal Evolution of "
        f"Hedonic Bias ({int(years[0])}–{int(years[-1])})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── H1 violin ────────────────────────────────────────────────────
    df_sim = pd.read_csv(OUTPUT_DIR / "ngram_h1_cosine_similarities.csv")
    dim_df = pd.read_csv(OUTPUT_DIR / "ngram_h1_dim_stats.csv")
    with open(OUTPUT_DIR / "ngram_h1_overall.json") as f:
        overall = json.load(f)

    # Build usage weights from mental-health book data
    ngram_df = pd.read_csv(OUTPUT_DIR / "mh_books_word_freq.csv")
    recent = ngram_df[ngram_df["year"].isin(RECENT_YEARS)]
    word_freq = recent.groupby("word")["count"].mean()
    total_freq = word_freq.sum()
    words = sorted(df_sim["word"].dropna().unique())
    weights = {w: word_freq.get(w, 0) / total_freq for w in words}

    fig1 = _make_ngram_h1_violin(
        df_sim, dim_df, overall, overall["n_words"], weights
    )
    for ext in ("png", "pdf"):
        fig1.savefig(FIGURES_DIR / f"ngram_h1_violin.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Saved ngram H1 violin to {FIGURES_DIR}")

    # ── H2 combined panel ────────────────────────────────────────────
    df_yearly = pd.read_csv(OUTPUT_DIR / "ngram_h2_yearly_delta.csv")
    df_trends = pd.read_csv(OUTPUT_DIR / "ngram_h2_dimension_trends.csv")
    with open(OUTPUT_DIR / "ngram_h2_summary.json") as f:
        h2_summary = json.load(f)

    dim_cols = [c for c in df_yearly.columns
                if c not in ("year", "aggregate")]

    fig2 = _make_ngram_h2_combined(df_yearly, df_trends, h2_summary,
                                    dim_cols)
    for ext in ("png", "pdf"):
        fig2.savefig(FIGURES_DIR / f"ngram_h2_combined.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved ngram H2 combined to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
