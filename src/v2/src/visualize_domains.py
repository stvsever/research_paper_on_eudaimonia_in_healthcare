"""
Post-hoc visualization.
1. Domain-level H1 bar chart.
2. Domain-level H2 small multiples.
3. Permutation null distribution.
4. Sensitivity to N.
5. Usage vs. delta scatter.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#457B9D", "#E63946", "#A8DADC"]


def significance_label(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ===== 1. Domain H1 bar chart =====
    df_h1 = pd.read_csv(OUTPUT_DIR / "posthoc_domain_h1.csv")
    df_h1 = df_h1.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    domains = df_h1["domain"].tolist()
    n = len(domains)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.barh(range(n), df_h1["mean_delta"],
            color=[PALETTE[i % len(PALETTE)] for i in range(n)],
            edgecolor="white", linewidth=0.5)
    p_col = "p_adjusted" if "p_adjusted" in df_h1.columns else "wilcoxon_p"
    for i, (_, row) in enumerate(df_h1.iterrows()):
        sig = significance_label(row[p_col])
        suffix = " (Holm)" if p_col == "p_adjusted" else ""
        label = f"  d={row['cohen_d']:.2f}  {sig}{suffix}"
        ax.text(row["mean_delta"] + 0.002, i, label, va="center", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{d}  (n={int(row['n'])})" for d, (_, row) in zip(domains, df_h1.iterrows())], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Δ  (hedonic − eudaimonic)", fontsize=10)
    ax.set_title("Post-Hoc — Domain-Level Hedonic Bias (H1)", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle="--")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"posthoc_domain_h1.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ===== 2. Domain H2 small multiples =====
    df_h2 = pd.read_csv(OUTPUT_DIR / "posthoc_domain_h2.csv")
    if not df_h2.empty:
        # Regenerate yearly data from raw inputs
        usage_df = pd.read_csv(OUTPUT_DIR / "usage_counts.csv")
        domain_path = DATA_DIR / "domains" / "domain_assignments.csv"
        domain_map = dict(pd.read_csv(domain_path).values) if domain_path.exists() else {}

        with open(OUTPUT_DIR / "embeddings.json") as f:
            data = json.load(f)
        dimension_names = data["dimension_names"]
        hedonic_embs = data["hedonic_embeddings"]
        eudaimonic_embs = data["eudaimonic_embeddings"]
        scale_embs = data["scale_embeddings"]
        scales_meta = data["scales_metadata"]
        scale_abbrevs = [s["abbreviation"] for s in scales_meta if s["abbreviation"] in scale_embs]

        scale_mean_delta = {}
        for a in scale_abbrevs:
            e = scale_embs[a]
            ds = [1.0 - cosine(e, hedonic_embs[d]) - (1.0 - cosine(e, eudaimonic_embs[d])) for d in dimension_names]
            scale_mean_delta[a] = np.mean(ds)

        years = list(range(2000, 2026))
        n_dom = len(df_h2)
        ncols = min(3, n_dom)
        nrows = max(1, (n_dom + ncols - 1) // ncols)

        fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=False)
        fig2.patch.set_facecolor("white")
        if n_dom == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()

        for idx, (_, row) in enumerate(df_h2.iterrows()):
            domain = row["domain"]
            ax = axes_flat[idx]
            dom_scales = [a for a in scale_abbrevs if domain_map.get(a) == domain]

            yearly_deltas = []
            for year in years:
                yu = usage_df[usage_df["year"] == year].set_index("scale")["count"]
                total = sum(yu.get(a, 0) for a in dom_scales)
                if total == 0:
                    yearly_deltas.append(np.nan)
                    continue
                wd = sum((yu.get(a, 0) / total) * scale_mean_delta[a] for a in dom_scales)
                yearly_deltas.append(wd)

            y_arr = np.array(yearly_deltas)
            mask = ~np.isnan(y_arr)
            y_clean = y_arr[mask]
            x_clean = np.array(years)[mask]
            color = PALETTE[idx % len(PALETTE)]
            ax.plot(x_clean, y_clean, "o-", color=color, markersize=3, linewidth=1.2)
            if len(x_clean) >= 5:
                sl, ic, _, _, _ = linregress(x_clean.astype(float), y_clean)
                ax.plot(x_clean, ic + sl * x_clean, "--", color="#E63946", linewidth=1.2)

            p_col_h2 = "p_adjusted" if "p_adjusted" in df_h2.columns else "p_ols"
            sig = significance_label(row[p_col_h2])
            total_chg = row.get("total_change_25yr", row["slope"] * 25)
            ax.set_title(f"{domain}  {sig}", fontsize=9, fontweight="bold")
            ax.text(0.02, 0.02,
                    f"Δ/decade={row.get('slope_per_decade', row['slope']*10):+.5f}  "
                    f"total={total_chg:+.5f}  R²={row['r_squared']:.3f}",
                    transform=ax.transAxes, fontsize=7, color="#555", va="bottom")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.15, linestyle="--")
            ax.tick_params(labelsize=7)

        for idx in range(n_dom, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig2.suptitle("Post-Hoc — Domain-Level Temporal Trends (H2)", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig2.savefig(FIGURES_DIR / f"posthoc_domain_h2.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

    # ===== 3. Permutation null distribution =====
    perm_path = OUTPUT_DIR / "posthoc_permutation.json"
    if perm_path.exists():
        with open(perm_path) as f:
            perm = json.load(f)
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        fig3.patch.set_facecolor("white")
        # Simulate null from mean/sd (we stored summary stats, not full dist)
        rng = np.random.default_rng(42)
        null_samples = rng.normal(perm["null_mean"], perm["null_sd"], 50_000)
        ax3.hist(null_samples, bins=80, color="#8D99AE", edgecolor="white", linewidth=0.3, density=True, alpha=0.7)
        ax3.axvline(perm["observed_delta"], color="#E63946", linewidth=2.2, linestyle="-",
                    label=f"Observed Δ = {perm['observed_delta']:+.5f}")
        ax3.set_xlabel("Mean Δ  (hedonic − eudaimonic)", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        perm_p_str = f"p = {perm['perm_p']:.5f}" if perm['perm_p'] >= 0.0001 else "p < .0001"
        ax3.set_title(f"Permutation Test (n = 10,000)  —  {perm_p_str}",
                      fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig3.savefig(FIGURES_DIR / f"posthoc_permutation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig3)

    # ===== 4. Sensitivity to N =====
    sens_path = OUTPUT_DIR / "posthoc_sensitivity_n.csv"
    if sens_path.exists():
        df_sens = pd.read_csv(sens_path)
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        fig4.patch.set_facecolor("white")
        ax4.plot(df_sens["top_n"], df_sens["mean_delta_raw"], "o-", color="#264653", linewidth=1.5, label="Raw Δ")
        ax4.plot(df_sens["top_n"], df_sens["mean_delta_weighted"], "s--", color="#E76F51", linewidth=1.5, label="Weighted Δ")
        ax4.set_xlabel("Top-N scales included", fontsize=10)
        ax4.set_ylabel("Mean Δ", fontsize=10)
        ax4.set_title("Sensitivity Analysis — Effect Stability Across N", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.grid(axis="y", alpha=0.2, linestyle="--")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig4.savefig(FIGURES_DIR / f"posthoc_sensitivity_n.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig4)

    # ===== 5. Usage vs. delta scatter =====
    scatter_path = OUTPUT_DIR / "posthoc_usage_delta_scatter.csv"
    if scatter_path.exists():
        df_sc = pd.read_csv(scatter_path)
        with open(OUTPUT_DIR / "posthoc_usage_correlation.json") as f:
            corr = json.load(f)

        # Filter to scales with non-zero usage for log scale
        df_sc_pos = df_sc[df_sc["usage_weight"] > 0].copy()

        fig5, ax5 = plt.subplots(figsize=(8, 6))
        fig5.patch.set_facecolor("white")
        unique_domains = df_sc_pos["domain"].unique()
        for i, dom in enumerate(sorted(unique_domains)):
            sub = df_sc_pos[df_sc_pos["domain"] == dom]
            ax5.scatter(sub["usage_weight"], sub["mean_delta"],
                        color=PALETTE[i % len(PALETTE)], s=30, alpha=0.7, label=dom, edgecolors="white", linewidth=0.3)
        ax5.axhline(0, color="black", linewidth=0.5, linestyle="-")

        # Add OLS regression line + polynomial fit on log-transformed usage
        log_x = np.log10(df_sc_pos["usage_weight"].values)
        y_vals = df_sc_pos["mean_delta"].values
        x_sorted = np.sort(log_x)
        x_plot = np.linspace(x_sorted[0], x_sorted[-1], 200)

        # Linear fit
        slope, intercept, _, _, _ = linregress(log_x, y_vals)
        ax5.plot(10**x_plot, slope * x_plot + intercept, "--",
                 color="#E63946", linewidth=1.5, alpha=0.8, label="Linear fit")

        # Polynomial (degree 2) fit
        poly_coeffs = np.polyfit(log_x, y_vals, 2)
        poly_y = np.polyval(poly_coeffs, x_plot)
        ax5.plot(10**x_plot, poly_y, "-",
                 color="#457B9D", linewidth=1.8, alpha=0.8, label="Quadratic fit")

        ax5.set_xscale("log")
        ax5.set_xlabel("Usage Weight — log scale (proportion of total)", fontsize=10)
        ax5.set_ylabel("Mean Δ  (hedonic − eudaimonic)", fontsize=10)
        ax5.set_title(f"Scale Usage vs. Hedonic Loading  (ρ = {corr['spearman_rho']:.3f}, p = {corr['spearman_p']:.4f})",
                      fontsize=12, fontweight="bold")
        ax5.legend(fontsize=7, ncol=2, loc="upper right", frameon=True)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        ax5.grid(alpha=0.15, linestyle="--")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig5.savefig(FIGURES_DIR / f"posthoc_usage_scatter.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig5)

    # ===== 6. Combined Domain H1 + H2 (A/B panel) =====
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(18, 6))
    fig6.patch.set_facecolor("white")

    # Panel A — Domain H1 bar chart
    df_h1_s = df_h1.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    domains_s = df_h1_s["domain"].tolist()
    n_s = len(domains_s)
    ax6a.barh(range(n_s), df_h1_s["mean_delta"],
              color=[PALETTE[i % len(PALETTE)] for i in range(n_s)],
              edgecolor="white", linewidth=0.5)
    p_col_6a = "p_adjusted" if "p_adjusted" in df_h1_s.columns else "wilcoxon_p"
    for i, (_, row) in enumerate(df_h1_s.iterrows()):
        sig = significance_label(row[p_col_6a])
        label = f"  d = {row['cohen_d']:.2f}  {sig}"
        ax6a.text(row["mean_delta"] + 0.002, i, label, va="center", fontsize=8)
    ax6a.set_yticks(range(n_s))
    ax6a.set_yticklabels([f"{d}  (n = {int(row['n'])})"
                          for d, (_, row) in zip(domains_s, df_h1_s.iterrows())], fontsize=9)
    ax6a.invert_yaxis()
    ax6a.set_xlabel("Mean Δ  (hedonic − eudaimonic)", fontsize=10)
    ax6a.set_title("Cross-Sectional Hedonic Bias by Domain", fontsize=11, fontweight="bold")
    ax6a.axvline(0, color="black", linewidth=0.6)
    ax6a.spines["top"].set_visible(False)
    ax6a.spines["right"].set_visible(False)
    ax6a.grid(axis="x", alpha=0.15, linestyle="--")
    ax6a.text(-0.03, 1.04, "A", transform=ax6a.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    # Panel B — Domain H2 summary (slope per decade as horizontal bars)
    if not df_h2.empty:
        df_h2_s = df_h2.sort_values("slope", ascending=False).reset_index(drop=True)
        slope_decade = df_h2_s["slope"].values * 10
        domains_h2 = df_h2_s["domain"].tolist()
        n_h2 = len(domains_h2)
        colors_h2 = ["#2A9D8F" if s >= 0 else "#E76F51" for s in slope_decade]
        ax6b.barh(range(n_h2), slope_decade, color=colors_h2,
                  edgecolor="white", linewidth=0.5)
        p_col_6b = "p_adjusted" if "p_adjusted" in df_h2_s.columns else "p_ols"
        for i, (_, row) in enumerate(df_h2_s.iterrows()):
            sig = significance_label(row[p_col_6b])
            total_chg = row.get("total_change_25yr", row["slope"] * 25)
            label = f"  R² = {row['r_squared']:.2f}  {sig}"
            xpos = row["slope"] * 10
            ax6b.text(xpos + 0.0002 if xpos >= 0 else xpos - 0.0002, i,
                      label, va="center", fontsize=8,
                      ha="left" if xpos >= 0 else "right")
        ax6b.set_yticks(range(n_h2))
        ax6b.set_yticklabels([f"{d}  (n = {int(row['n'])})"
                              for d, (_, row) in zip(domains_h2, df_h2_s.iterrows())], fontsize=9)
        ax6b.invert_yaxis()
        ax6b.set_xlabel("Slope (Δ per decade)", fontsize=10)
        ax6b.set_title("Temporal Trend by Domain (2000–2025)", fontsize=11, fontweight="bold")
        ax6b.axvline(0, color="black", linewidth=0.6)
        ax6b.spines["top"].set_visible(False)
        ax6b.spines["right"].set_visible(False)
        ax6b.grid(axis="x", alpha=0.15, linestyle="--")
        ax6b.text(-0.03, 1.04, "B", transform=ax6b.transAxes,
                  fontsize=16, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig6.savefig(FIGURES_DIR / f"posthoc_domain_combined.{ext}",
                     dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig6)

    # ===== 7. Combined Sensitivity + Usage Scatter (A/B panel) =====
    if sens_path.exists() and scatter_path.exists():
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(16, 6))
        fig7.patch.set_facecolor("white")

        # Panel A — Sensitivity to N
        df_sens2 = pd.read_csv(sens_path)
        ax7a.plot(df_sens2["top_n"], df_sens2["mean_delta_raw"], "o-",
                  color="#264653", linewidth=1.5, markersize=6, label="Raw Δ")
        ax7a.plot(df_sens2["top_n"], df_sens2["mean_delta_weighted"], "s--",
                  color="#E76F51", linewidth=1.5, markersize=6, label="Weighted Δ")
        # Add Cohen's d on secondary y-axis
        ax7a_d = ax7a.twinx()
        ax7a_d.plot(df_sens2["top_n"], df_sens2["cohen_d_raw"], "^:",
                    color="#8D99AE", linewidth=1.2, markersize=5, label="Cohen's d (raw)")
        ax7a_d.plot(df_sens2["top_n"], df_sens2["cohen_d_weighted"], "v:",
                    color="#2A9D8F", linewidth=1.2, markersize=5, label="Cohen's d (weighted)")
        ax7a_d.set_ylabel("Cohen's d", fontsize=9, color="#8D99AE")
        ax7a_d.tick_params(axis="y", labelcolor="#8D99AE")
        ax7a.set_xlabel("Top-N Scales Included", fontsize=10)
        ax7a.set_ylabel("Mean Δ", fontsize=10)
        ax7a.set_title("Sensitivity to Corpus Size", fontsize=11, fontweight="bold")
        # Merge legends
        h1, l1 = ax7a.get_legend_handles_labels()
        h2, l2 = ax7a_d.get_legend_handles_labels()
        ax7a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
        ax7a.spines["top"].set_visible(False)
        ax7a.grid(axis="y", alpha=0.2, linestyle="--")
        ax7a.text(-0.03, 1.04, "A", transform=ax7a.transAxes,
                  fontsize=16, fontweight="bold", va="bottom", ha="right")

        # Panel B — Usage scatter
        df_sc2 = pd.read_csv(scatter_path)
        with open(OUTPUT_DIR / "posthoc_usage_correlation.json") as f:
            corr2 = json.load(f)
        df_sc2_pos = df_sc2[df_sc2["usage_weight"] > 0].copy()
        unique_doms = sorted(df_sc2_pos["domain"].unique())
        for i, dom in enumerate(unique_doms):
            sub = df_sc2_pos[df_sc2_pos["domain"] == dom]
            ax7b.scatter(sub["usage_weight"], sub["mean_delta"],
                         color=PALETTE[i % len(PALETTE)], s=30, alpha=0.7,
                         label=dom, edgecolors="white", linewidth=0.3)
        ax7b.axhline(0, color="black", linewidth=0.5)
        log_x = np.log10(df_sc2_pos["usage_weight"].values)
        y_vals = df_sc2_pos["mean_delta"].values
        x_sorted = np.sort(log_x)
        x_plot = np.linspace(x_sorted[0], x_sorted[-1], 200)
        slope_lr, intercept_lr, _, _, _ = linregress(log_x, y_vals)
        ax7b.plot(10**x_plot, slope_lr * x_plot + intercept_lr, "--",
                  color="#E63946", linewidth=1.5, alpha=0.8, label="Linear fit")
        # Quadratic fit
        poly_coeffs = np.polyfit(log_x, y_vals, 2)
        poly_y = np.polyval(poly_coeffs, x_plot)
        ax7b.plot(10**x_plot, poly_y, "-",
                  color="#457B9D", linewidth=1.8, alpha=0.8, label="Quadratic fit")
        corr_p_str = f"p = {corr2['spearman_p']:.4f}" if corr2['spearman_p'] >= 0.0001 else "p < .0001"
        ax7b.set_xscale("log")
        ax7b.set_xlabel("Usage Weight (log scale)", fontsize=10)
        ax7b.set_ylabel("Mean Δ  (hedonic − eudaimonic)", fontsize=10)
        ax7b.set_title(f"Usage vs. Hedonic Loading  (ρ = {corr2['spearman_rho']:.3f}, {corr_p_str})",
                       fontsize=11, fontweight="bold")
        ax7b.legend(fontsize=6.5, ncol=2, loc="upper right", frameon=True)
        ax7b.spines["top"].set_visible(False)
        ax7b.spines["right"].set_visible(False)
        ax7b.grid(alpha=0.15, linestyle="--")
        ax7b.text(-0.03, 1.04, "B", transform=ax7b.transAxes,
                  fontsize=16, fontweight="bold", va="bottom", ha="right")

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig7.savefig(FIGURES_DIR / f"posthoc_sensitivity_usage_combined.{ext}",
                         dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig7)

    print(f"Saved post-hoc figures to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
