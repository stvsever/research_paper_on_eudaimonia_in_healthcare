"""
H2 Analysis — Temporal trend in hedonic measurement bias (2000-2025).

For each year, computes usage-weighted mean Δ (hedonic − eudaimonic cosine
similarity) across all 200+ scales, per dimension.  Tests whether each
dimension's Δ_weighted increases over time (OLS + Mann-Kendall + bootstrap).

Output: yearly Δ per dimension → one figure with 6 coloured lines.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import linregress, norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
YEARS = list(range(2000, 2026))
N_BOOTSTRAP = 10_000
RNG_SEED = 42


def cosine_similarity(a, b):
    return 1.0 - cosine(a, b)


def mann_kendall(x):
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i, j in combinations(range(n), 2))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * norm.sf(abs(z))
    return float(s), float(p)


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

    # Pre-compute per-scale, per-dimension deltas
    scale_dim_deltas = {}
    for a in scale_abbrevs:
        s_emb = scale_embs[a]
        for dim in dimension_names:
            sim_h = cosine_similarity(s_emb, hedonic_embs[dim])
            sim_e = cosine_similarity(s_emb, eudaimonic_embs[dim])
            scale_dim_deltas[(a, dim)] = sim_h - sim_e

    # Compute yearly usage-weighted Δ per dimension
    yearly_records = []
    for year in YEARS:
        year_usage = usage_df[usage_df["year"] == year].set_index("scale")["count"]
        total = sum(year_usage.get(a, 0) for a in scale_abbrevs)
        if total == 0:
            continue

        rec = {"year": year}
        for dim in dimension_names:
            w_d = sum(
                (year_usage.get(a, 0) / total) * scale_dim_deltas[(a, dim)]
                for a in scale_abbrevs
            )
            rec[dim] = w_d

        # Also compute the aggregate (mean across dimensions)
        rec["aggregate"] = np.mean([rec[dim] for dim in dimension_names])
        yearly_records.append(rec)

    df_yearly = pd.DataFrame(yearly_records)
    df_yearly.to_csv(OUTPUT_DIR / "h2_yearly_delta.csv", index=False)

    # Trend statistics per dimension + aggregate
    years_arr = df_yearly["year"].values.astype(float)
    rng = np.random.default_rng(RNG_SEED)
    n = len(years_arr)

    trend_records = []
    cols_to_test = dimension_names + ["aggregate"]
    for col in cols_to_test:
        y = df_yearly[col].values
        slope, intercept, r, p, se = linregress(years_arr, y)
        mk_s, mk_p = mann_kendall(y)

        # Bootstrap CI on slope
        boot_slopes = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.choice(n, n, replace=True)
            boot_slopes[i] = linregress(years_arr[idx], y[idx]).slope
        ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

        # Durbin-Watson autocorrelation diagnostic
        residuals = y - (intercept + slope * years_arr)
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))

        trend_records.append({
            "dimension": col,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r ** 2,
            "p_ols": p,
            "se": se,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "mann_kendall_S": mk_s,
            "mann_kendall_p": mk_p,
            "durbin_watson": dw,
        })

    df_trends = pd.DataFrame(trend_records)
    df_trends.to_csv(OUTPUT_DIR / "h2_dimension_trends.csv", index=False)

    # Summary JSON
    agg_row = df_trends[df_trends["dimension"] == "aggregate"].iloc[0]
    h2_summary = {k: float(v) if isinstance(v, (np.floating, float)) else v
                  for k, v in agg_row.to_dict().items()}
    h2_summary["n_years"] = int(n)
    with open(OUTPUT_DIR / "h2_summary.json", "w") as f:
        json.dump(h2_summary, f, indent=2)

    # Print
    print("\n=== H2: TEMPORAL TRENDS ===")
    for _, row in df_trends.iterrows():
        sig = "***" if row["p_ols"] < 0.001 else "**" if row["p_ols"] < 0.01 else "*" if row["p_ols"] < 0.05 else "n.s."
        print(f"  {row['dimension']:25s}  slope={row['slope']:+.6f}/yr  "
              f"R²={row['r_squared']:.4f}  {sig}  "
              f"CI=[{row['ci_95_lo']:+.6f}, {row['ci_95_hi']:+.6f}]  "
              f"MK p={row['mann_kendall_p']:.4f}  "
              f"DW={row['durbin_watson']:.3f}")

    return df_yearly, df_trends


if __name__ == "__main__":
    run()
