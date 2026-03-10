"""
H1 replication on mental-health book data — usage-weighted only.

For each word appearing in Internet Archive mental-health books,
computes cosine similarity to each of the 6 hedonic and 6 eudaimonic
dimension embeddings (same dimensional framework as the clinical-scale
analysis).

Uses frequency weighting (2010-2019 mean frequency from mental-health
books) as the usage proxy, mirroring the PubMed-based usage weighting
in the main analysis.

The corpus is filtered at the **book level**: only books whose Internet
Archive subject metadata matches psychology, psychiatry, or related
mental-health fields are included, ensuring that the word frequencies
reflect mental-health literature specifically.

Output:
  - ngram_h1_dim_stats.csv  (per-dimension statistics)
  - ngram_h1_overall.json   (aggregate statistics)
  - ngram_h1_cosine_similarities.csv  (word-level similarities)
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RECENT_YEARS = range(2000, 2020)  # broader window for book corpus coverage


def cosine_similarity(a, b):
    return 1.0 - cosine(a, b)


def _holm_adjust(pvals):
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


def run():
    # Load dimension embeddings from main pipeline
    with open(OUTPUT_DIR / "embeddings.json") as f:
        main_data = json.load(f)
    dimension_names = main_data["dimension_names"]
    hedonic_embs = main_data["hedonic_embeddings"]
    eudaimonic_embs = main_data["eudaimonic_embeddings"]

    # Load ngram embeddings
    with open(OUTPUT_DIR / "ngram_embeddings.json") as f:
        ngram_data = json.load(f)
    word_embs = ngram_data["word_embeddings"]

    # Load mental-health book frequency data (book-level filtered)
    ngram_df = pd.read_csv(OUTPUT_DIR / "mh_books_word_freq.csv")
    recent = ngram_df[ngram_df["year"].isin(RECENT_YEARS)]

    # Keep only words that appear in the mental-health book corpus
    # AND have embeddings — no word-level semantic filtering
    mh_words = set(ngram_df["word"].unique())
    words = sorted(w for w in mh_words if w in word_embs)
    word_embs = {w: word_embs[w] for w in words}
    print(f"  Words in MH books with embeddings: {len(words)}")

    # Usage weights from mental-health books
    word_freq = recent.groupby("word")["count"].mean()
    total_freq = word_freq.sum()
    usage_weights = {w: word_freq.get(w, 0) / total_freq for w in words}

    # Compute cosine similarities per (word, dimension)
    records = []
    for word in words:
        if word not in word_embs:
            continue
        w_emb = word_embs[word]
        for dim in dimension_names:
            sim_h = cosine_similarity(w_emb, hedonic_embs[dim])
            sim_e = cosine_similarity(w_emb, eudaimonic_embs[dim])
            records.append({
                "word": word,
                "dimension": dim,
                "sim_hedonic": sim_h,
                "sim_eudaimonic": sim_e,
                "delta": sim_h - sim_e,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "ngram_h1_cosine_similarities.csv", index=False)
    print(f"  Computed similarities for {len(words)} words × "
          f"{len(dimension_names)} dimensions = {len(df)} pairs")

    # Per-dimension statistics (usage-weighted)
    dim_rows = []
    for dim in dimension_names:
        d = df[df["dimension"] == dim]
        h = d["sim_hedonic"].values
        e = d["sim_eudaimonic"].values
        delta = h - e
        word_list = d["word"].values

        stat_w, p_w = wilcoxon(h, e, alternative="greater")
        stat_t, p_t = ttest_rel(h, e)

        # Raw stats
        mean_d = np.mean(delta)
        std_d = np.std(delta, ddof=1)
        d_raw = mean_d / std_d if std_d > 0 else 0

        # Weighted stats
        w = np.array([usage_weights.get(word, 0) for word in word_list])
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
        w_mean = np.average(delta, weights=w_norm)
        w_var = np.average((delta - w_mean) ** 2, weights=w_norm)
        w_std = np.sqrt(w_var) if w_var > 0 else 1e-12

        dim_rows.append({
            "dimension": dim,
            "mean_hedonic": np.mean(h),
            "mean_eudaimonic": np.mean(e),
            "mean_delta": mean_d,
            "cohen_d": d_raw,
            "mean_delta_weighted": w_mean,
            "cohen_d_weighted": w_mean / w_std,
            "wilcoxon_stat": stat_w,
            "wilcoxon_p": p_w,
            "ttest_p": p_t,
            "n_words": len(d["word"].unique()),
        })

    dim_df = pd.DataFrame(dim_rows)
    dim_df["p_adjusted"] = _holm_adjust(dim_df["wilcoxon_p"].values)
    dim_df.to_csv(OUTPUT_DIR / "ngram_h1_dim_stats.csv", index=False)

    # Overall statistics (usage-weighted)
    h_all = df["sim_hedonic"].values
    e_all = df["sim_eudaimonic"].values
    delta_all = h_all - e_all
    word_all = df["word"].values

    stat_w, p_w = wilcoxon(h_all, e_all, alternative="greater")
    stat_t, p_t = ttest_rel(h_all, e_all)
    std_all = np.std(delta_all, ddof=1)

    w_all = np.array([usage_weights.get(word, 0) for word in word_all])
    w_sum = w_all.sum()
    w_norm = w_all / w_sum if w_sum > 0 else np.ones_like(w_all) / len(w_all)
    w_mean_all = float(np.average(delta_all, weights=w_norm))
    w_var_all = np.average((delta_all - w_mean_all) ** 2, weights=w_norm)
    w_std_all = np.sqrt(w_var_all) if w_var_all > 0 else 1e-12

    overall = {
        "mean_delta": float(np.mean(delta_all)),
        "cohen_d": float(np.mean(delta_all) / std_all) if std_all > 0 else 0,
        "mean_delta_weighted": w_mean_all,
        "cohen_d_weighted": float(w_mean_all / w_std_all),
        "wilcoxon_p": float(p_w),
        "ttest_p": float(p_t),
        "n_observations": int(len(delta_all)),
        "n_words": len(words),
    }
    with open(OUTPUT_DIR / "ngram_h1_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Print summary
    print("\n=== NGRAM H1: CROSS-SECTIONAL (USAGE-WEIGHTED) ===")
    for _, row in dim_df.iterrows():
        sig = "***" if row["p_adjusted"] < 0.001 else (
              "**" if row["p_adjusted"] < 0.01 else (
              "*" if row["p_adjusted"] < 0.05 else "n.s."))
        print(f"  {row['dimension']:25s}  Δw={row['mean_delta_weighted']:+.4f}  "
              f"dw={row['cohen_d_weighted']:.3f}  {sig} (Holm)")
    print(f"  Overall: Δw={overall['mean_delta_weighted']:+.4f}  "
          f"dw={overall['cohen_d_weighted']:.3f}  p={overall['wilcoxon_p']:.2e}")

    return dim_df, overall


if __name__ == "__main__":
    run()
