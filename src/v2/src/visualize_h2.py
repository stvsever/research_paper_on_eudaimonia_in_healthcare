"""
H2 Visualization:
 1. Dimension trend plot: 6 thin dimension lines + bold aggregate
    with pointwise 95% CI ribbon (from cross-dimension variability).
 2. Baseline-corrected version: subtract year-2000 values from each series.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

DIM_COLORS = [
    "#264653",  # dark teal
    "#2A9D8F",  # teal
    "#E9C46A",  # gold
    "#F4A261",  # sandy brown
    "#E76F51",  # burnt sienna
    "#457B9D",  # steel blue
]
COLOR_AGG = "#2B2D42"
COLOR_CI = "#8D99AE"


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _make_h2_figure(df_yearly, df_trends, summary, dim_cols,
                    baseline_correct=False):
    """Build the H2 dimension-trend figure (raw or baseline-corrected)."""
    years = df_yearly["year"].values
    n_dims = len(dim_cols)

    # Build matrix of dimension values (n_years × n_dims)
    dim_matrix = np.column_stack([df_yearly[d].values for d in dim_cols])
    agg_vals = df_yearly["aggregate"].values.copy()

    if baseline_correct:
        baseline_dims = dim_matrix[0].copy()
        dim_matrix = dim_matrix - baseline_dims[np.newaxis, :]
        agg_baseline = agg_vals[0]
        agg_vals = agg_vals - agg_baseline

    # Pointwise CI from cross-dimension variability
    t_crit = t_dist.ppf(0.975, df=n_dims - 1)  # ~2.571 for df=5
    means = np.mean(dim_matrix, axis=1)
    ses = np.std(dim_matrix, axis=1, ddof=1) / np.sqrt(n_dims)
    ci_lo = means - t_crit * ses
    ci_hi = means + t_crit * ses

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")

    # CI ribbon (behind everything)
    ax.fill_between(years, ci_lo, ci_hi, color=COLOR_CI, alpha=0.22,
                    label="Aggregate 95% CI", zorder=1)

    # Dimension lines (thin, semi-transparent)
    for i, dim in enumerate(dim_cols):
        color = DIM_COLORS[i % len(DIM_COLORS)]
        y = dim_matrix[:, i]
        trend = df_trends[df_trends["dimension"] == dim].iloc[0]
        sig = significance_label(trend["p_ols"])
        ax.plot(years, y, "-", color=color, linewidth=0.9, alpha=0.55,
                label=f"{dim}  (β={trend['slope']:+.5f}/yr {sig})", zorder=2)

    # Aggregate line (bold, prominent)
    agg_trend = df_trends[df_trends["dimension"] == "aggregate"].iloc[0]
    agg_sig = significance_label(agg_trend["p_ols"])
    ax.plot(years, agg_vals, "D-", color=COLOR_AGG, markersize=4,
            linewidth=2.5,
            label=f"Aggregate  (β={agg_trend['slope']:+.5f}/yr {agg_sig})",
            zorder=5)

    ax.set_xlabel("Year", fontsize=11)
    if baseline_correct:
        ax.set_ylabel("Δ Change from 2000 Baseline", fontsize=10)
        ax.set_title(
            "H2 — Change in Hedonic Measurement Bias Since 2000 "
            "(Baseline-Corrected)",
            fontsize=13, fontweight="bold",
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    else:
        ax.set_ylabel("Usage-Weighted Δ  (hedonic − eudaimonic)", fontsize=10)
        ax.set_title(
            "H2 — Temporal Evolution of Hedonic Measurement Bias "
            "per Dimension (2000–2025)",
            fontsize=13, fontweight="bold",
        )

    ax.legend(fontsize=7.5, loc="upper left", frameon=True, fancybox=True,
              ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    mk_p = summary.get('mann_kendall_p', 1)
    mk_p_str = f"p={mk_p:.4f}" if mk_p >= 0.0001 else "p < .0001"
    anno = (
        f"Aggregate: R²={summary.get('r_squared', 0):.4f}   "
        f"Mann-Kendall S={summary.get('mann_kendall_S', 0):.0f}, "
        f"{mk_p_str}"
    )
    ax.text(0.98, 0.03, anno, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color="#555")

    plt.tight_layout()
    return fig


def _make_h2_combined(df_yearly, df_trends, summary, dim_cols):
    """Build a 1×2 panel figure (usage-weighted only):
       A (left):  Raw usage-weighted Δ
       B (right): Baseline-corrected usage-weighted Δ

    Equal-weight panels are omitted because embeddings are fixed per scale;
    without usage weighting the series is flat by construction, adding no
    information beyond confirming that the temporal trend is adoption-driven.
    """
    years = df_yearly["year"].values
    n_dims = len(dim_cols)

    dim_matrix = np.column_stack([df_yearly[d].values for d in dim_cols])
    agg_vals = df_yearly["aggregate"].values.copy()

    dim_bc = dim_matrix - dim_matrix[0][np.newaxis, :]
    agg_bc = agg_vals - agg_vals[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")

    panel_configs = [
        (axes[0], dim_matrix, agg_vals, False,
         "Usage-Weighted Δ (hedonic − eudaimonic)", "Raw"),
        (axes[1], dim_bc, agg_bc, True,
         "Change in Usage-Weighted Δ from 2000", "Baseline-Corrected"),
    ]
    labels = ["A", "B"]

    for idx, (ax, dm, av, is_bc, ylabel, subtitle) in enumerate(panel_configs):
        t_crit = t_dist.ppf(0.975, df=n_dims - 1)
        means = np.mean(dm, axis=1)
        ses = np.std(dm, axis=1, ddof=1) / np.sqrt(n_dims)
        ci_lo = means - t_crit * ses
        ci_hi = means + t_crit * ses

        ax.fill_between(years, ci_lo, ci_hi, color=COLOR_CI, alpha=0.22,
                        label="Aggregate 95% CI" if idx == 0 else None, zorder=1)

        for i, dim in enumerate(dim_cols):
            color = DIM_COLORS[i % len(DIM_COLORS)]
            y = dm[:, i]
            if idx == 0:
                trend = df_trends[df_trends["dimension"] == dim].iloc[0]
                sig = significance_label(trend["p_ols"])
                lbl = f"{dim}  (β={trend['slope']:+.5f}/yr {sig})"
            else:
                lbl = None
            ax.plot(years, y, "-", color=color, linewidth=0.9, alpha=0.55,
                    label=lbl, zorder=2)

        agg_trend = df_trends[df_trends["dimension"] == "aggregate"].iloc[0]
        agg_sig = significance_label(agg_trend["p_ols"])
        if idx == 0:
            lbl_agg = f"Aggregate  (β={agg_trend['slope']:+.5f}/yr {agg_sig})"
        else:
            lbl_agg = None
        ax.plot(years, av, "D-", color=COLOR_AGG, markersize=3,
                linewidth=2.2, label=lbl_agg, zorder=5)

        if is_bc:
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.text(-0.03, 1.04, labels[idx], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Shared legend below both panels
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", fontsize=7.5,
               ncol=4, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def run():
    df_yearly = pd.read_csv(OUTPUT_DIR / "h2_yearly_delta.csv")
    df_trends = pd.read_csv(OUTPUT_DIR / "h2_dimension_trends.csv")
    with open(OUTPUT_DIR / "h2_summary.json") as f:
        summary = json.load(f)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dim_cols = [c for c in df_yearly.columns
                if c not in ("year", "aggregate")]

    # Figure 1: Raw dimension trends
    fig1 = _make_h2_figure(df_yearly, df_trends, summary, dim_cols,
                           baseline_correct=False)
    for ext in ("png", "pdf"):
        fig1.savefig(FIGURES_DIR / f"h2_dimension_trends.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # Figure 2: Baseline-corrected
    fig2 = _make_h2_figure(df_yearly, df_trends, summary, dim_cols,
                           baseline_correct=True)
    for ext in ("png", "pdf"):
        fig2.savefig(FIGURES_DIR / f"h2_baseline_corrected.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    print(f"Saved H2 figures to {FIGURES_DIR}")

    # Figure 3: Combined A/B panel (raw left, baseline-corrected right)
    fig3 = _make_h2_combined(df_yearly, df_trends, summary, dim_cols)
    for ext in ("png", "pdf"):
        fig3.savefig(FIGURES_DIR / f"h2_combined.{ext}", dpi=300,
                     bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"Saved H2 combined figure to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
