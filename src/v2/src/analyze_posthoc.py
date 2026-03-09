"""
Post-hoc & robustness analyses.

1. Domain-level H1: per-domain mean Δ and Wilcoxon.
2. Permutation test: shuffle hedonic/eudaimonic labels 10,000 times → null Δ.
3. Sensitivity to N: effect size as function of top-N (20, 50, 100, 150, 200+).
4. Scale-level scatter: usage weight vs. Δ (does popularity → hedonic loading?).
5. Domain-level H2: per-domain temporal trend.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, linregress, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
YEARS = list(range(2000, 2026))
N_PERMUTATIONS = 10_000
N_BOOT_SLOPE = 5_000
RNG_SEED = 42


def _holm_adjust(pvals):
    """Holm-Bonferroni correction for multiple comparisons."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted = np.minimum(sorted_p * (n - np.arange(n)), 1.0)
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


def cosine_similarity(a, b):
    return 1.0 - cosine(a, b)


def run():
    with open(OUTPUT_DIR / "embeddings.json") as f:
        data = json.load(f)

    dimension_names = data["dimension_names"]
    hedonic_embs = data["hedonic_embeddings"]
    eudaimonic_embs = data["eudaimonic_embeddings"]
    scale_embs = data["scale_embeddings"]
    scales_meta = data["scales_metadata"]

    usage_df = pd.read_csv(OUTPUT_DIR / "usage_counts.csv")
    scale_abbrevs = [s["abbreviation"] for s in scales_meta if s["abbreviation"] in scale_embs]

    # Load domain assignments
    domain_path = DATA_DIR / "domains" / "domain_assignments.csv"
    if domain_path.exists():
        domain_map = dict(pd.read_csv(domain_path).values)
    else:
        domain_map = {}

    # Pre-compute per-scale mean delta (across 6 dimensions)
    scale_mean_delta = {}
    scale_dim_deltas = {}
    for a in scale_abbrevs:
        s_emb = scale_embs[a]
        dims_d = []
        for dim in dimension_names:
            d = cosine_similarity(s_emb, hedonic_embs[dim]) - cosine_similarity(s_emb, eudaimonic_embs[dim])
            scale_dim_deltas[(a, dim)] = d
            dims_d.append(d)
        scale_mean_delta[a] = np.mean(dims_d)

    # Usage weights
    recent = usage_df[usage_df["year"].isin([2024, 2025])].groupby("scale")["count"].mean()
    total_usage = sum(recent.get(a, 0) for a in scale_abbrevs)
    weights = {a: recent.get(a, 0) / total_usage if total_usage > 0 else 0 for a in scale_abbrevs}
    sorted_by_usage = sorted(scale_abbrevs, key=lambda a: weights.get(a, 0), reverse=True)

    # ============ 1. DOMAIN-LEVEL H1 ============
    print("\n=== POST-HOC 1: DOMAIN H1 ===")
    domains = sorted(set(domain_map.get(a, "Unassigned") for a in scale_abbrevs))
    domain_h1 = []
    for domain in domains:
        dom_scales = [a for a in scale_abbrevs if domain_map.get(a, "Unassigned") == domain]
        if len(dom_scales) < 3:
            continue
        deltas = np.array([scale_mean_delta[a] for a in dom_scales])
        mean_d = np.mean(deltas)
        std_d = np.std(deltas, ddof=1)
        d_eff = mean_d / std_d if std_d > 0 else 0
        if len(deltas) >= 6:
            _, p = wilcoxon(deltas, alternative="greater")
        else:
            p = np.nan
        domain_h1.append({"domain": domain, "n": len(dom_scales), "mean_delta": mean_d,
                          "cohen_d": d_eff, "wilcoxon_p": p})

    # Holm-Bonferroni correction
    raw_ps = [r["wilcoxon_p"] for r in domain_h1]
    adj_ps = _holm_adjust([p if not np.isnan(p) else 1.0 for p in raw_ps])
    for i, rec in enumerate(domain_h1):
        rec["p_adjusted"] = float(adj_ps[i]) if not np.isnan(raw_ps[i]) else np.nan
        sig = "***" if rec["p_adjusted"] < 0.001 else "**" if rec["p_adjusted"] < 0.01 else "*" if rec["p_adjusted"] < 0.05 else "n.s."
        print(f"  {rec['domain']:42s}  n={rec['n']:3d}  Δ={rec['mean_delta']:+.4f}  d={rec['cohen_d']:.3f}  {sig} (Holm-adj)")

    pd.DataFrame(domain_h1).to_csv(OUTPUT_DIR / "posthoc_domain_h1.csv", index=False)

    # ============ 2. PERMUTATION TEST ============
    print("\n=== POST-HOC 2: PERMUTATION TEST ===")
    rng = np.random.default_rng(RNG_SEED)

    # Observed mean Δ across all scales and dimensions
    observed_delta = np.mean([scale_mean_delta[a] for a in scale_abbrevs])

    # Under null: randomly swap hedonic/eudaimonic label for each dimension
    null_deltas = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm_delta = 0
        for dim in dimension_names:
            if rng.random() < 0.5:
                # Swap labels for this dimension
                for a in scale_abbrevs:
                    perm_delta += -scale_dim_deltas[(a, dim)]
            else:
                for a in scale_abbrevs:
                    perm_delta += scale_dim_deltas[(a, dim)]
        null_deltas[i] = perm_delta / (len(scale_abbrevs) * len(dimension_names))

    perm_p = np.mean(null_deltas >= observed_delta)
    print(f"  Observed Δ: {observed_delta:+.5f}")
    print(f"  Permutation p (one-sided): {perm_p:.5f}")
    print(f"  Null distribution: mean={np.mean(null_deltas):.5f} sd={np.std(null_deltas):.5f}")

    perm_result = {
        "observed_delta": float(observed_delta),
        "perm_p": float(perm_p),
        "null_mean": float(np.mean(null_deltas)),
        "null_sd": float(np.std(null_deltas)),
    }
    with open(OUTPUT_DIR / "posthoc_permutation.json", "w") as f:
        json.dump(perm_result, f, indent=2)

    # ============ 3. SENSITIVITY TO N ============
    print("\n=== POST-HOC 3: SENSITIVITY TO N ===")
    n_values = [20, 50, 100, 150, len(scale_abbrevs)]
    sensitivity = []
    for n_val in n_values:
        top_scales = sorted_by_usage[:n_val]
        deltas = np.array([scale_mean_delta[a] for a in top_scales])
        mean_d = np.mean(deltas)
        std_d = np.std(deltas, ddof=1)
        d_eff = mean_d / std_d if std_d > 0 else 0
        # Weighted
        w = np.array([weights.get(a, 0) for a in top_scales])
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
        w_mean = np.average(deltas, weights=w_norm)
        # Weighted Cohen's d
        w_var = np.average((deltas - w_mean)**2, weights=w_norm)
        w_std = np.sqrt(w_var) if w_var > 0 else 1e-12
        d_weighted = w_mean / w_std
        sensitivity.append({"top_n": n_val, "mean_delta_raw": mean_d, "cohen_d_raw": d_eff,
                            "mean_delta_weighted": w_mean, "cohen_d_weighted": d_weighted})
        print(f"  Top-{n_val:3d}: Δ_raw={mean_d:+.4f}  d_raw={d_eff:.3f}  Δ_weighted={w_mean:+.4f}  d_weighted={d_weighted:.3f}")

    pd.DataFrame(sensitivity).to_csv(OUTPUT_DIR / "posthoc_sensitivity_n.csv", index=False)

    # ============ 4. SCALE-LEVEL SCATTER: usage ↔ delta ============
    print("\n=== POST-HOC 4: USAGE vs DELTA CORRELATION ===")
    usage_vals = np.array([weights.get(a, 0) for a in scale_abbrevs])
    delta_vals = np.array([scale_mean_delta[a] for a in scale_abbrevs])
    rho, p_rho = spearmanr(usage_vals, delta_vals)
    print(f"  Spearman rho: {rho:.4f}  p={p_rho:.6f}")

    scatter_df = pd.DataFrame({
        "scale": scale_abbrevs,
        "usage_weight": usage_vals,
        "mean_delta": delta_vals,
        "domain": [domain_map.get(a, "Unassigned") for a in scale_abbrevs],
    })
    scatter_df.to_csv(OUTPUT_DIR / "posthoc_usage_delta_scatter.csv", index=False)

    corr_result = {"spearman_rho": float(rho), "spearman_p": float(p_rho)}
    with open(OUTPUT_DIR / "posthoc_usage_correlation.json", "w") as f:
        json.dump(corr_result, f, indent=2)

    # ============ 5. DOMAIN-LEVEL H2 ============
    print("\n=== POST-HOC 5: DOMAIN H2 (temporal) ===")
    rng_boot = np.random.default_rng(RNG_SEED + 99)
    domain_h2 = []
    for domain in domains:
        dom_scales = [a for a in scale_abbrevs if domain_map.get(a, "Unassigned") == domain]
        if len(dom_scales) < 3:
            continue
        yearly = []
        for year in YEARS:
            yu = usage_df[usage_df["year"] == year].set_index("scale")["count"]
            t = sum(yu.get(a, 0) for a in dom_scales)
            if t == 0:
                continue
            wd = sum((yu.get(a, 0) / t) * scale_mean_delta[a] for a in dom_scales)
            yearly.append({"year": year, "delta": wd})
        if len(yearly) < 5:
            continue
        df_y = pd.DataFrame(yearly)
        yrs = df_y["year"].values.astype(float)
        deltas = df_y["delta"].values
        sl, ic, r, p, se = linregress(yrs, deltas)
        n_years = len(yrs)
        total_change = sl * (yrs[-1] - yrs[0])
        slope_decade = sl * 10

        # Bootstrap CI on slope
        boot_slopes = np.empty(N_BOOT_SLOPE)
        for bi in range(N_BOOT_SLOPE):
            idx = rng_boot.choice(n_years, n_years, replace=True)
            boot_slopes[bi] = linregress(yrs[idx], deltas[idx]).slope
        ci_lo = float(np.percentile(boot_slopes, 2.5))
        ci_hi = float(np.percentile(boot_slopes, 97.5))

        domain_h2.append({
            "domain": domain, "n": len(dom_scales), "slope": sl,
            "slope_per_decade": slope_decade, "total_change_25yr": total_change,
            "slope_ci_lo": ci_lo, "slope_ci_hi": ci_hi,
            "r_squared": r**2, "p_ols": p,
        })

    # Holm-Bonferroni correction for domain H2
    raw_ps_h2 = [r["p_ols"] for r in domain_h2]
    adj_ps_h2 = _holm_adjust(raw_ps_h2)
    for i, rec in enumerate(domain_h2):
        rec["p_adjusted"] = float(adj_ps_h2[i])
        sig = "***" if rec["p_adjusted"] < 0.001 else "**" if rec["p_adjusted"] < 0.01 else "*" if rec["p_adjusted"] < 0.05 else "n.s."
        print(f"  {rec['domain']:42s}  slope/decade={rec['slope_per_decade']:+.5f}  "
              f"total_Δ25yr={rec['total_change_25yr']:+.5f}  R²={rec['r_squared']:.4f}  {sig} (Holm-adj)")

    pd.DataFrame(domain_h2).to_csv(OUTPUT_DIR / "posthoc_domain_h2.csv", index=False)

    return {
        "domain_h1": domain_h1,
        "permutation": perm_result,
        "sensitivity": sensitivity,
        "correlation": corr_result,
        "domain_h2": domain_h2,
    }


if __name__ == "__main__":
    run()
