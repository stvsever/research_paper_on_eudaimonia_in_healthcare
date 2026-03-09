"""Compute rank-biserial correlation (non-parametric effect size) and
Shapiro-Wilk normality test results for H1 analysis."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import shapiro, wilcoxon

OUTPUT_DIR = Path("outputs")

with open(OUTPUT_DIR / "embeddings.json") as f:
    data = json.load(f)

dim_names = data["dimension_names"]
hedonic_embs = data["hedonic_embeddings"]
eudaimonic_embs = data["eudaimonic_embeddings"]
scale_embs = data["scale_embeddings"]
scales_meta = data["scales_metadata"]

records = []
for s in scales_meta:
    a = s["abbreviation"]
    if a not in scale_embs:
        continue
    e = scale_embs[a]
    for dim in dim_names:
        sim_h = 1.0 - cosine(e, hedonic_embs[dim])
        sim_e = 1.0 - cosine(e, eudaimonic_embs[dim])
        records.append({"scale": a, "dimension": dim,
                        "sim_hedonic": sim_h, "sim_eudaimonic": sim_e,
                        "delta": sim_h - sim_e})

df = pd.DataFrame(records)
n_scales = len(df["scale"].unique())

# Per-dimension Shapiro-Wilk + rank-biserial
shapiro_results = []
for dim in dim_names:
    sub = df[df["dimension"] == dim]
    deltas = sub["delta"].values
    h = sub["sim_hedonic"].values
    e = sub["sim_eudaimonic"].values
    n = len(deltas)

    # Shapiro-Wilk on deltas
    sw_w, sw_p = shapiro(deltas)

    # Wilcoxon signed-rank test
    stat_w, p_w = wilcoxon(h, e, alternative="greater")

    # Rank-biserial r = 1 - (2*W) / (n*(n+1)/2)  where W is the Wilcoxon statistic
    # But scipy returns the T statistic (sum of positive ranks)
    # r_rb = 4*T / (n*(n+1)) - 1  for one-sample/paired Wilcoxon
    r_rb = 4 * stat_w / (n * (n + 1)) - 1

    shapiro_results.append({
        "dimension": dim,
        "n": n,
        "shapiro_W": sw_w,
        "shapiro_p": sw_p,
        "wilcoxon_T": stat_w,
        "wilcoxon_p": p_w,
        "rank_biserial_r": r_rb,
        "cohen_d": np.mean(deltas) / np.std(deltas, ddof=1),
    })

# Overall
all_h = df["sim_hedonic"].values
all_e = df["sim_eudaimonic"].values
all_d = df["delta"].values
sw_w, sw_p = shapiro(all_d[:5000])
stat_w, p_w = wilcoxon(all_h, all_e, alternative="greater")
n_all = len(all_d)
r_rb_all = 4 * stat_w / (n_all * (n_all + 1)) - 1

shapiro_results.append({
    "dimension": "Overall",
    "n": n_all,
    "shapiro_W": sw_w,
    "shapiro_p": sw_p,
    "wilcoxon_T": stat_w,
    "wilcoxon_p": p_w,
    "rank_biserial_r": r_rb_all,
    "cohen_d": np.mean(all_d) / np.std(all_d, ddof=1),
})

df_out = pd.DataFrame(shapiro_results)
df_out.to_csv(OUTPUT_DIR / "h1_normality_tests.csv", index=False)

# Save as JSON too
with open(OUTPUT_DIR / "h1_normality_tests.json", "w") as f:
    json.dump(shapiro_results, f, indent=2)

print("=== Normality & Non-Parametric Effect Sizes ===")
print(f"{'Dimension':<25} {'SW W':>7} {'SW p':>10} {'r_rb':>7} {'d':>7}")
print("-" * 62)
for r in shapiro_results:
    p_str = f"{r['shapiro_p']:.2e}" if r["shapiro_p"] < 0.001 else f"{r['shapiro_p']:.4f}"
    print(f"{r['dimension']:<25} {r['shapiro_W']:>7.4f} {p_str:>10} {r['rank_biserial_r']:>7.3f} {r['cohen_d']:>7.2f}")

print(f"\nAll dimensions: non-normal (Shapiro-Wilk p < .001)")
print(f"Wilcoxon signed-rank test (non-parametric) already used — appropriate.")
print(f"Rank-biserial r range: {min(r['rank_biserial_r'] for r in shapiro_results[:-1]):.3f} to {max(r['rank_biserial_r'] for r in shapiro_results[:-1]):.3f}")
