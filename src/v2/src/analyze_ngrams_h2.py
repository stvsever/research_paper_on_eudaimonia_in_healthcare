"""
H2 replication on mental-health book data — usage-weighted only.

For each year (1900–2019), computes frequency-weighted mean Δ (hedonic −
eudaimonic cosine similarity) across the words appearing in Internet
Archive mental-health books, per dimension and aggregate.

The corpus is filtered at the **book level**: only books whose Internet
Archive subject metadata matches psychology, psychiatry, or related
mental-health fields are included.

Tests whether each dimension's frequency-weighted Δ shows a temporal
trend over the 120-year period, mirroring the clinical H2 analysis
(OLS + Mann-Kendall + bootstrap CI + Durbin-Watson).

Output:
  - ngram_h2_yearly_delta.csv       (per-year Δ per dimension)
  - ngram_h2_dimension_trends.csv   (trend statistics)
  - ngram_h2_summary.json           (aggregate summary)
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

YEARS = list(range(1900, 2020))
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

    # Keep only words that appear in the MH book corpus
    # AND have embeddings — no word-level semantic filtering
    mh_words = set(ngram_df["word"].unique())
    words = sorted(w for w in mh_words if w in word_embs)
    word_embs = {w: word_embs[w] for w in words}
    print(f"  Words in MH books with embeddings: {len(words)}")

    # Load ngram frequency data
    # (already loaded as ngram_df above — reuse it)

    # Pre-compute per-word, per-dimension deltas
    word_dim_deltas = {}
    for w in words:
        if w not in word_embs:
            continue
        w_emb = word_embs[w]
        for dim in dimension_names:
            sim_h = cosine_similarity(w_emb, hedonic_embs[dim])
            sim_e = cosine_similarity(w_emb, eudaimonic_embs[dim])
            word_dim_deltas[(w, dim)] = sim_h - sim_e

    print(f"  Pre-computed {len(word_dim_deltas)} (word, dim) deltas")

    # Aggregate frequency per (word, year) — sum across POS tags
    freq_by_word_year = (
        ngram_df.groupby(["word", "year"])["count"]
        .sum()
        .reset_index()
        .set_index(["word", "year"])["count"]
    )

    # Compute yearly frequency-weighted Δ per dimension
    yearly_records = []
    for year in YEARS:
        # Get frequency for this year for all words
        year_freqs = {}
        for w in words:
            try:
                year_freqs[w] = freq_by_word_year.get((w, year), 0)
            except KeyError:
                year_freqs[w] = 0

        total = sum(year_freqs.values())
        if total == 0:
            continue

        rec = {"year": year}
        for dim in dimension_names:
            w_d = sum(
                (year_freqs[w] / total) * word_dim_deltas.get((w, dim), 0)
                for w in words
                if w in word_embs
            )
            rec[dim] = w_d

        rec["aggregate"] = np.mean([rec[dim] for dim in dimension_names])
        yearly_records.append(rec)

    df_yearly = pd.DataFrame(yearly_records)
    df_yearly.to_csv(OUTPUT_DIR / "ngram_h2_yearly_delta.csv", index=False)
    print(f"  Computed yearly Δ for {len(df_yearly)} years")

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

        # Durbin-Watson
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
    df_trends.to_csv(OUTPUT_DIR / "ngram_h2_dimension_trends.csv", index=False)

    # Summary JSON
    agg_row = df_trends[df_trends["dimension"] == "aggregate"].iloc[0]
    h2_summary = {k: float(v) if isinstance(v, (np.floating, float)) else v
                  for k, v in agg_row.to_dict().items()}
    h2_summary["n_years"] = int(n)
    h2_summary["year_range"] = f"{int(years_arr[0])}-{int(years_arr[-1])}"
    with open(OUTPUT_DIR / "ngram_h2_summary.json", "w") as f:
        json.dump(h2_summary, f, indent=2)

    # Print
    print("\n=== NGRAM H2: TEMPORAL TRENDS ===")
    for _, row in df_trends.iterrows():
        sig = "***" if row["p_ols"] < 0.001 else (
              "**" if row["p_ols"] < 0.01 else (
              "*" if row["p_ols"] < 0.05 else "n.s."))
        print(f"  {row['dimension']:25s}  slope={row['slope']:+.6f}/yr  "
              f"R²={row['r_squared']:.4f}  {sig}  "
              f"CI=[{row['ci_95_lo']:+.6f}, {row['ci_95_hi']:+.6f}]  "
              f"MK p={row['mann_kendall_p']:.4f}  "
              f"DW={row['durbin_watson']:.3f}")

    return df_yearly, df_trends


if __name__ == "__main__":
    run()
