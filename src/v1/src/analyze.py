"""
Analysis module.
Computes cosine similarities between clinical scale embeddings and hedonic/eudaimonic
dimension embeddings, then runs statistical significance tests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, ttest_rel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine(a, b)


def run():
    embeddings_path = OUTPUT_DIR / "embeddings.json"
    with open(embeddings_path) as f:
        data = json.load(f)

    dimension_names = data["dimension_names"]
    hedonic_embs = data["hedonic_embeddings"]
    eudaimonic_embs = data["eudaimonic_embeddings"]
    scale_embs = data["scale_embeddings"]
    scales_meta = data["scales_metadata"]

    # --- Compute cosine similarities ---
    records = []
    for scale in scales_meta:
        abbrev = scale["abbreviation"]
        s_emb = scale_embs[abbrev]

        for dim in dimension_names:
            sim_h = cosine_similarity(s_emb, hedonic_embs[dim])
            sim_e = cosine_similarity(s_emb, eudaimonic_embs[dim])
            records.append({
                "scale": abbrev,
                "scale_name": scale["name"],
                "dimension": dim,
                "cosine_sim_hedonic": sim_h,
                "cosine_sim_eudaimonic": sim_e,
                "delta": sim_h - sim_e,  # positive = closer to hedonic
            })

    df = pd.DataFrame(records)

    # --- Per-dimension statistics (paired Wilcoxon signed-rank test) ---
    dim_stats = []
    for dim in dimension_names:
        d = df[df["dimension"] == dim]
        hedonic_sims = d["cosine_sim_hedonic"].values
        eudaimonic_sims = d["cosine_sim_eudaimonic"].values

        # Wilcoxon signed-rank test (non-parametric paired test)
        stat_w, p_w = wilcoxon(hedonic_sims, eudaimonic_sims, alternative="greater")
        # Paired t-test as complement
        stat_t, p_t = ttest_rel(hedonic_sims, eudaimonic_sims)

        dim_stats.append({
            "dimension": dim,
            "mean_hedonic": np.mean(hedonic_sims),
            "mean_eudaimonic": np.mean(eudaimonic_sims),
            "mean_delta": np.mean(hedonic_sims - eudaimonic_sims),
            "std_delta": np.std(hedonic_sims - eudaimonic_sims, ddof=1),
            "wilcoxon_stat": stat_w,
            "wilcoxon_p": p_w,
            "ttest_stat": stat_t,
            "ttest_p_twosided": p_t,
            "n_scales": len(hedonic_sims),
            "effect_size_d": np.mean(hedonic_sims - eudaimonic_sims) / np.std(hedonic_sims - eudaimonic_sims, ddof=1),
        })

    df_stats = pd.DataFrame(dim_stats)

    # --- Overall aggregate test ---
    all_h = df["cosine_sim_hedonic"].values
    all_e = df["cosine_sim_eudaimonic"].values
    stat_w_all, p_w_all = wilcoxon(all_h, all_e, alternative="greater")
    stat_t_all, p_t_all = ttest_rel(all_h, all_e)

    overall = {
        "mean_hedonic": float(np.mean(all_h)),
        "mean_eudaimonic": float(np.mean(all_e)),
        "mean_delta": float(np.mean(all_h - all_e)),
        "wilcoxon_p": float(p_w_all),
        "ttest_p": float(p_t_all),
        "effect_size_d": float(np.mean(all_h - all_e) / np.std(all_h - all_e, ddof=1)),
    }

    # --- Save ---
    df.to_csv(OUTPUT_DIR / "cosine_similarities.csv", index=False)
    df_stats.to_csv(OUTPUT_DIR / "dimension_statistics.csv", index=False)
    with open(OUTPUT_DIR / "overall_statistics.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\n=== PER-DIMENSION STATISTICS ===")
    for _, row in df_stats.iterrows():
        sig = "***" if row["wilcoxon_p"] < 0.001 else "**" if row["wilcoxon_p"] < 0.01 else "*" if row["wilcoxon_p"] < 0.05 else "n.s."
        print(f"  {row['dimension']:25s}  Δ={row['mean_delta']:+.4f}  p={row['wilcoxon_p']:.4f} {sig}  d={row['effect_size_d']:.3f}")

    print(f"\n=== OVERALL ===")
    print(f"  Mean hedonic sim:    {overall['mean_hedonic']:.4f}")
    print(f"  Mean eudaimonic sim: {overall['mean_eudaimonic']:.4f}")
    print(f"  Mean Δ:              {overall['mean_delta']:+.4f}")
    print(f"  Wilcoxon p:          {overall['wilcoxon_p']:.6f}")
    print(f"  Cohen's d:           {overall['effect_size_d']:.3f}")

    return df, df_stats, overall


if __name__ == "__main__":
    run()
