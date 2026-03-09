"""Test normality of H1 delta distributions with Shapiro-Wilk."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import shapiro, wilcoxon, ttest_1samp

with open("outputs/embeddings.json") as f:
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
        records.append({"scale": a, "dimension": dim, "delta": sim_h - sim_e})

df = pd.DataFrame(records)

print("=== Shapiro-Wilk Test for Normality of delta (per dimension) ===")
print(f"{'Dimension':<25} {'W':>8} {'p':>14} {'Normal?':>10}")
print("-" * 62)
any_non_normal = False
for dim in dim_names:
    deltas = df[df["dimension"] == dim]["delta"].values
    w, p = shapiro(deltas)
    normal = p >= 0.05
    if not normal:
        any_non_normal = True
    print(f"{dim:<25} {w:>8.4f} {p:>14.2e} {'Yes' if normal else 'NO':>10}")

all_deltas = df["delta"].values
w, p = shapiro(all_deltas[:5000])
print(f"{'Overall (pooled)':<25} {w:>8.4f} {p:>14.2e} {'Yes' if p >= 0.05 else 'NO':>10}")

print(f"\nAny non-normal? {any_non_normal}")
print("\nNote: The primary test (Wilcoxon signed-rank) is already non-parametric")
print("and does NOT assume normality. No change needed for p-values.")
print("Cohen's d is reported as a descriptive effect-size measure.")

print("\n=== Verification: t-test vs Wilcoxon agreement ===")
for dim in dim_names:
    sub = df[df["dimension"] == dim]
    deltas = sub["delta"].values
    stat_w, p_w = wilcoxon(deltas, alternative="greater")
    stat_t, p_t = ttest_1samp(deltas, 0, alternative="greater")
    agree = (p_w < 0.001) == (p_t < 0.001)
    print(f"{dim:<25} Wilcoxon p={p_w:.2e}  t-test p={p_t:.2e}  Agree: {'Yes' if agree else 'No'}")
