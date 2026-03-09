"""
H1 Analysis — Cross-sectional hedonic bias in clinical measurement.

Four sub-analyses:
  H1a: Top-N most-used scales only (no eudaimonic forced) — raw cosine Δ
  H1b: Top-N most-used scales only — usage-weighted cosine Δ
  H1c: All 200+ scales (incl. eudaimonic) — raw cosine Δ
  H1d: All 200+ scales — usage-weighted cosine Δ

The contrast between H1c-raw (modest bias) and H1d-weighted (strong bias) is
the paper's core insight: eudaimonic tools *exist*, but adoption crushes them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TOP_N = 50  # "most clinically used" subset


def cosine_similarity(a, b):
    return 1.0 - cosine(a, b)


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


def _compute_scale_similarities(scale_embs, hedonic_embs, eudaimonic_embs,
                                 dimension_names, scales_meta):
    """Return DataFrame with per-scale, per-dimension cosine similarities."""
    records = []
    for scale in scales_meta:
        abbrev = scale["abbreviation"]
        if abbrev not in scale_embs:
            continue
        s_emb = scale_embs[abbrev]
        for dim in dimension_names:
            sim_h = cosine_similarity(s_emb, hedonic_embs[dim])
            sim_e = cosine_similarity(s_emb, eudaimonic_embs[dim])
            records.append({
                "scale": abbrev,
                "dimension": dim,
                "sim_hedonic": sim_h,
                "sim_eudaimonic": sim_e,
                "delta": sim_h - sim_e,
            })
    return pd.DataFrame(records)


def _usage_weights(usage_df, scale_list, recent_years=(2024, 2025)):
    """Normalised usage weight per scale (mean of recent years)."""
    recent = usage_df[usage_df["year"].isin(recent_years)].groupby("scale")["count"].mean()
    total = sum(recent.get(s, 0) for s in scale_list)
    if total == 0:
        return {s: 1.0 / len(scale_list) for s in scale_list}
    return {s: recent.get(s, 0) / total for s in scale_list}


def _dim_stats(df, weights=None):
    """Per-dimension paired test + effect size. If weights provided: weighted."""
    dimension_names = df["dimension"].unique()
    rows = []
    for dim in dimension_names:
        d = df[df["dimension"] == dim]
        h = d["sim_hedonic"].values
        e = d["sim_eudaimonic"].values
        delta = h - e

        stat_w, p_w = wilcoxon(h, e, alternative="greater")
        stat_t, p_t = ttest_rel(h, e)
        mean_d = np.mean(delta)
        std_d = np.std(delta, ddof=1)
        d_uw = mean_d / std_d if std_d > 0 else 0

        row = {
            "dimension": dim,
            "mean_hedonic": np.mean(h),
            "mean_eudaimonic": np.mean(e),
            "mean_delta": mean_d,
            "cohen_d": d_uw,
            "wilcoxon_stat": stat_w,
            "wilcoxon_p": p_w,
            "ttest_p": p_t,
            "n_scales": len(d["scale"].unique()),
        }

        if weights is not None:
            w = np.array([weights.get(s, 0) for s in d["scale"].values])
            w_sum = w.sum()
            if w_sum > 0:
                w_norm = w / w_sum
            else:
                w_norm = np.ones_like(w) / len(w)
            w_mean = np.average(delta, weights=w_norm)
            w_var = np.average((delta - w_mean) ** 2, weights=w_norm)
            w_std = np.sqrt(w_var) if w_var > 0 else 1e-12
            row["mean_delta_weighted"] = w_mean
            row["cohen_d_weighted"] = w_mean / w_std

        rows.append(row)
    dim_df = pd.DataFrame(rows)
    # Holm-Bonferroni correction across dimensions
    dim_df["p_adjusted"] = _holm_adjust(dim_df["wilcoxon_p"].values)
    return dim_df


def _overall_stats(df, weights=None):
    h = df["sim_hedonic"].values
    e = df["sim_eudaimonic"].values
    delta = h - e
    stat_w, p_w = wilcoxon(h, e, alternative="greater")
    stat_t, p_t = ttest_rel(h, e)
    std_d = np.std(delta, ddof=1)

    result = {
        "mean_delta": float(np.mean(delta)),
        "cohen_d": float(np.mean(delta) / std_d) if std_d > 0 else 0,
        "wilcoxon_p": float(p_w),
        "ttest_p": float(p_t),
        "n_observations": int(len(delta)),
    }

    if weights is not None:
        w = np.array([weights.get(s, 0) for s in df["scale"].values])
        w_sum = w.sum()
        if w_sum > 0:
            w_norm = w / w_sum
        else:
            w_norm = np.ones_like(w) / len(w)
        w_mean = float(np.average(delta, weights=w_norm))
        w_var = np.average((delta - w_mean) ** 2, weights=w_norm)
        w_std = np.sqrt(w_var) if w_var > 0 else 1e-12
        result["mean_delta_weighted"] = w_mean
        result["cohen_d_weighted"] = float(w_mean / w_std)

    return result


def run():
    with open(OUTPUT_DIR / "embeddings.json") as f:
        data = json.load(f)

    dimension_names = data["dimension_names"]
    hedonic_embs = data["hedonic_embeddings"]
    eudaimonic_embs = data["eudaimonic_embeddings"]
    scale_embs = data["scale_embeddings"]
    scales_meta = data["scales_metadata"]

    usage_df = pd.read_csv(OUTPUT_DIR / "usage_counts.csv")

    # All scale abbreviations that have both embeddings and usage data
    all_abbrevs = [s["abbreviation"] for s in scales_meta if s["abbreviation"] in scale_embs]
    all_weights = _usage_weights(usage_df, all_abbrevs)

    # Determine top-N by recent usage (no eudaimonic forcing)
    sorted_by_usage = sorted(all_abbrevs, key=lambda a: all_weights.get(a, 0), reverse=True)
    top_n_abbrevs = sorted_by_usage[:TOP_N]
    top_n_set = set(top_n_abbrevs)

    # === Compute similarities ===
    df_all = _compute_scale_similarities(scale_embs, hedonic_embs, eudaimonic_embs,
                                          dimension_names, scales_meta)
    df_topn = df_all[df_all["scale"].isin(top_n_set)].copy()

    top_n_weights = _usage_weights(usage_df, top_n_abbrevs)

    # === Four sub-analyses ===
    analyses = {}
    for label, df, w in [
        ("h1a_topN_raw", df_topn, None),
        ("h1b_topN_weighted", df_topn, top_n_weights),
        ("h1c_all_raw", df_all, None),
        ("h1d_all_weighted", df_all, all_weights),
    ]:
        dim_df = _dim_stats(df, w)
        overall = _overall_stats(df, w)

        dim_df.to_csv(OUTPUT_DIR / f"{label}_dim_stats.csv", index=False)
        with open(OUTPUT_DIR / f"{label}_overall.json", "w") as f:
            json.dump(overall, f, indent=2)

        analyses[label] = {"dim_stats": dim_df, "overall": overall}

        print(f"\n=== {label.upper()} ===")
        for _, row in dim_df.iterrows():
            sig = "***" if row["p_adjusted"] < 0.001 else "**" if row["p_adjusted"] < 0.01 else "*" if row["p_adjusted"] < 0.05 else "n.s."
            d_col = "cohen_d_weighted" if w else "cohen_d"
            d_val = row.get(d_col, row["cohen_d"])
            print(f"  {row['dimension']:25s}  Δ={row['mean_delta']:+.4f}  d={d_val:.3f}  {sig} (Holm)")
        print(f"  Overall: Δ={overall['mean_delta']:+.4f}  d={overall['cohen_d']:.3f}  p={overall['wilcoxon_p']:.6f}")

    # Save raw similarity data and top-N list
    df_all.to_csv(OUTPUT_DIR / "h1_cosine_similarities_all.csv", index=False)
    pd.DataFrame({"scale": top_n_abbrevs, "rank": range(1, TOP_N + 1)}).to_csv(
        OUTPUT_DIR / "h1_top_n_scales.csv", index=False
    )

    # Save all weights for downstream use
    pd.DataFrame([
        {"scale": s, "weight": all_weights.get(s, 0)} for s in all_abbrevs
    ]).to_csv(OUTPUT_DIR / "h1_usage_weights.csv", index=False)

    return analyses


if __name__ == "__main__":
    run()
