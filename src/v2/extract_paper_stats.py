"""Extract all paper statistics from pipeline output files.

Run after run_pipeline.py completes to get every number needed for main.tex.
"""
import csv, json, sys
from pathlib import Path

OUT = Path(__file__).parent / "outputs"

def load_json(name):
    with open(OUT / name) as f:
        return json.load(f)

def load_csv(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))

# ── H1 ──────────────────────────────────────────────────────────────
print("=" * 70)
print("H1: CROSS-SECTIONAL STATS")
print("=" * 70)

for tag, label in [
    ("h1a_topN_raw", "H1a: Top-50 raw"),
    ("h1b_topN_weighted", "H1b: Top-50 weighted"),
    ("h1c_all_raw", "H1c: All raw"),
    ("h1d_all_weighted", "H1d: All weighted"),
]:
    overall = load_json(f"{tag}_overall.json")
    dims = load_csv(f"{tag}_dim_stats.csv")
    print(f"\n--- {label} ---")
    print(f"  Overall Δ = {float(overall['mean_delta']):+.3f}")
    print(f"  Overall d = {float(overall['cohen_d']):.2f}")
    if "mean_delta_weighted" in overall:
        print(f"  Overall Δ_w = {float(overall['mean_delta_weighted']):+.3f}")
        print(f"  Overall d_w = {float(overall['cohen_d_weighted']):.2f}")
    print(f"  N obs = {overall['n_observations']}")
    print(f"  Wilcoxon p = {float(overall['wilcoxon_p']):.6f}")
    print()
    for d in dims:
        dim = d["dimension"]
        delta = float(d["mean_delta"])
        cd = float(d["cohen_d"])
        p = float(d["p_adjusted"])
        extra = ""
        if "mean_delta_weighted" in d:
            dw = float(d["mean_delta_weighted"])
            cdw = float(d["cohen_d_weighted"])
            extra = f"  Δ_w={dw:+.3f}  d_w={cdw:.2f}"
        sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
        print(f"  {dim:25s}  Δ={delta:+.3f}  d={cd:.2f}{extra}  {sig}")


# ── LaTeX: Table 2 (per-dim stats, all scales) ─────────────────────
print("\n" + "=" * 70)
print("LATEX TABLE 2 (per-dimension, all scales)")
print("=" * 70)
raw_dims = {d["dimension"]: d for d in load_csv("h1c_all_raw_dim_stats.csv")}
wt_dims = {d["dimension"]: d for d in load_csv("h1d_all_weighted_dim_stats.csv")}
raw_overall = load_json("h1c_all_raw_overall.json")
wt_overall = load_json("h1d_all_weighted_overall.json")

dim_order = [
    "Foundational claim", "Evaluative criterion", "Time horizon",
    "Adversity", "Measurement proxies", "Central tension",
]
for dim in dim_order:
    r = raw_dims[dim]
    w = wt_dims[dim]
    rd, rcd = float(r["mean_delta"]), float(r["cohen_d"])
    wd, wcd = float(w["mean_delta_weighted"]), float(w["cohen_d_weighted"])
    print(f"{dim:25s} & ${rd:+.3f}$ & ${rcd:.2f}$ & ${wd:+.3f}$ & ${wcd:.2f}$ & $< .001$ \\\\")

ro_d = float(raw_overall["mean_delta"])
ro_cd = float(raw_overall["cohen_d"])
wo_d = float(wt_overall["mean_delta_weighted"])
wo_cd = float(wt_overall["cohen_d_weighted"])
print(f"\\midrule")
print(f"\\textbf{{Overall}} & $\\boldsymbol{{{ro_d:+.3f}}}$ & $\\boldsymbol{{{ro_cd:.2f}}}$ & $\\boldsymbol{{{wo_d:+.3f}}}$ & $\\boldsymbol{{{wo_cd:.2f}}}$ & $\\boldsymbol{{< .001}}$ \\\\")


# ── H2 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("H2: TEMPORAL TREND STATS")
print("=" * 70)

h2_summary = load_json("h2_summary.json")
h2_dims = load_csv("h2_dimension_trends.csv")

print(f"\nAggregate:")
for k in ["slope", "slope_per_decade", "r_squared", "p_ols", "mann_kendall_S",
           "mann_kendall_p", "durbin_watson", "ci_95_lo", "ci_95_hi"]:
    if k in h2_summary:
        print(f"  {k} = {h2_summary[k]}")

# Total change over 25 years
slope = float(h2_summary["slope"])
total_25 = slope * 25
print(f"  total_change_25yr = {total_25:+.4f}")
print(f"  slope × 10^4 = {slope * 1e4:+.2f}")

print("\nPer-dimension trends:")
dw_values = []
for d in h2_dims:
    dim = d["dimension"]
    s = float(d["slope"])
    r2 = float(d["r_squared"])
    dw = float(d["durbin_watson"])
    mk_s = d.get("mann_kendall_S", "?")
    mk_p = d.get("mann_kendall_p", "?")
    dw_values.append(dw)
    print(f"  {dim:25s}  slope×1e4={s*1e4:+.2f}  R²={r2:.3f}  DW={dw:.2f}  MK_S={mk_s}  MK_p={mk_p}")

print(f"\n  DW range: {min(dw_values):.2f}--{max(dw_values):.2f}")

# ── LaTeX: Table 3 (H2 trends) ────────────────────────────────────
print("\n" + "=" * 70)
print("LATEX TABLE 3 (H2 temporal trends)")
print("=" * 70)
for d in h2_dims:
    dim = d["dimension"]
    s = float(d["slope"]) * 1e4
    r2 = float(d["r_squared"])
    mk_s = d.get("mann_kendall_S", "?")
    dw = float(d["durbin_watson"])
    mk_p_raw = d.get("mann_kendall_p", "?")
    mk_p_str = "< .001" if float(mk_p_raw) < 0.001 else f"{float(mk_p_raw):.3f}"
    p_str = "< .001" if float(d["p_ols"]) < 0.001 else f"{float(d['p_ols']):.3f}"
    print(f"{dim:25s} & ${s:+.2f}$ & $.{r2:.3f}$[fix] & ${p_str}$ & ${mk_s}$ & ${mk_p_str}$ & ${dw:.2f}$ \\\\")

# Aggregate row
s_agg = float(h2_summary["slope"]) * 1e4
r2_agg = float(h2_summary["r_squared"])
mk_s_agg = h2_summary.get("mann_kendall_S", "?")
mk_p_agg = h2_summary.get("mann_kendall_p", "?")
dw_agg = float(h2_summary["durbin_watson"])
print(f"\\textbf{{Aggregate}} & $\\boldsymbol{{{s_agg:+.2f}}}$ & $\\boldsymbol{{.{r2_agg:.3f}[fix]}}$ & ... \\\\")


# ── POST-HOC ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("POST-HOC STATS")
print("=" * 70)

# Permutation
perm = load_json("posthoc_permutation.json")
print(f"\nPermutation test:")
print(f"  observed Δ = {perm['observed_delta']:+.5f}")
print(f"  p = {perm['perm_p']:.3f}")
print(f"  null mean = {perm['null_mean']:+.4f}")
print(f"  null SD = {perm['null_sd']:.3f}")

# Sensitivity
print(f"\nSensitivity to N:")
sens = load_csv("posthoc_sensitivity_n.csv")
d_raw_values = []
for s in sens:
    n = s["top_n"]
    dr = float(s["mean_delta_raw"])
    cd = float(s["cohen_d_raw"])
    dw = float(s["mean_delta_weighted"])
    d_raw_values.append(cd)
    print(f"  Top-{n:>3s}: Δ_raw={dr:+.3f}  d_raw={cd:.2f}  Δ_wt={dw:+.3f}")
print(f"  d_raw range: {min(d_raw_values):.2f}--{max(d_raw_values):.2f}")

# ── LaTeX: sensitivity table ──────────────────────────────────────
print(f"\nLATEX SENSITIVITY TABLE:")
for s in sens:
    n = s["top_n"]
    dr = float(s["mean_delta_raw"])
    cd = float(s["cohen_d_raw"])
    dw = float(s["mean_delta_weighted"])
    print(f"{n:>3s} & ${dr:+.3f}$ & ${cd:.2f}$ & ${dw:+.3f}$ \\\\")

# Usage correlation
corr = load_json("posthoc_usage_correlation.json")
print(f"\nUsage-bias correlation:")
print(f"  Spearman rho = {corr['spearman_rho']:.3f}")
print(f"  p = {corr['spearman_p']:.3f}")

# Domain H1
print(f"\nDomain-level H1:")
dom_h1 = load_csv("posthoc_domain_h1.csv")
for d in sorted(dom_h1, key=lambda x: -float(x["cohen_d"])):
    domain = d["domain"]
    n = d["n"]
    delta = float(d["mean_delta"])
    cd = float(d["cohen_d"])
    p = float(d["p_adjusted"])
    sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
    print(f"  {domain:45s}  n={n:>2s}  Δ={delta:+.3f}  d={cd:.2f}  p_adj={p:.3f}  {sig}")

# ── LaTeX: Domain H1 ─────────────────────────────────────────────
print(f"\nLATEX DOMAIN H1:")
for d in sorted(dom_h1, key=lambda x: -float(x["cohen_d"])):
    domain = d["domain"]
    n = d["n"]
    delta = float(d["mean_delta"])
    cd = float(d["cohen_d"])
    p = float(d["p_adjusted"])
    p_str = "$< .001$" if p < 0.001 else f"${p:.3f}$"
    print(f"{domain:45s} & {n} & ${delta:+.3f}$ & ${cd:.2f}$ & {p_str} \\\\")


# Domain H2
print(f"\nDomain-level H2:")
dom_h2 = load_csv("posthoc_domain_h2.csv")
for d in sorted(dom_h2, key=lambda x: -float(x["slope_per_decade"])):
    domain = d["domain"]
    n = d["n"]
    spd = float(d["slope_per_decade"])
    tot = float(d["total_change_25yr"])
    r2 = float(d["r_squared"])
    p = float(d["p_adjusted"])
    sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
    print(f"  {domain:45s}  n={n:>2s}  Δ/dec={spd:+.4f}  tot25={tot:+.3f}  R²={r2:.3f}  {sig}")

# ── LaTeX: Domain H2 ─────────────────────────────────────────────
print(f"\nLATEX DOMAIN H2:")
for d in sorted(dom_h2, key=lambda x: -float(x["slope_per_decade"])):
    domain = d["domain"]
    n = d["n"]
    spd = float(d["slope_per_decade"])
    tot = float(d["total_change_25yr"])
    r2 = float(d["r_squared"])
    p = float(d["p_adjusted"])
    p_str = "$< .001$" if p < 0.001 else (f"$.{p:.3f}$" if p < 1 else f"${p:.3f}$")
    print(f"{domain:45s} & {n} & ${spd:+.4f}$ & ${tot:+.3f}$ & ${r2:.3f}$ & {p_str} \\\\")


# ── S8: Top hedonic/eudaimonic scales ─────────────────────────────
print("\n" + "=" * 70)
print("S8: TOP HEDONIC / EUDAIMONIC SCALES")
print("=" * 70)

sims = load_csv("h1_cosine_similarities_all.csv")
# Compute mean Δ per scale
from collections import defaultdict
scale_deltas = defaultdict(list)
for row in sims:
    scale_deltas[row["scale"]].append(float(row["delta"]))

mean_deltas = {s: sum(ds)/len(ds) for s, ds in scale_deltas.items()}
sorted_hedonic = sorted(mean_deltas.items(), key=lambda x: -x[1])
sorted_eudaimonic = sorted(mean_deltas.items(), key=lambda x: x[1])

print("\nTop 20 Hedonic:")
for i, (s, d) in enumerate(sorted_hedonic[:20], 1):
    print(f"  {i:2d}. {s:15s}  Δ = {d:+.3f}")

print("\nTop 10 Eudaimonic:")
for i, (s, d) in enumerate(sorted_eudaimonic[:10], 1):
    print(f"  {i:2d}. {s:15s}  Δ = {d:+.3f}")

# LaTeX S8
print("\nLATEX S8 TABLE ROWS:")
for i in range(20):
    hs, hd = sorted_hedonic[i]
    if i < 10:
        es, ed = sorted_eudaimonic[i]
        print(f"{i+1:2d} & {hs:15s} & ${hd:+.3f}$ & {i+1:2d} & {es:15s} & ${ed:+.3f}$ \\\\")
    else:
        print(f"{i+1:2d} & {hs:15s} & ${hd:+.3f}$ &    &                  &          \\\\")


# ── S9: Descriptive stats by dimension ────────────────────────────
print("\n" + "=" * 70)
print("S9: PER-DIMENSION DESCRIPTIVE STATS")
print("=" * 70)

dim_hedonic = defaultdict(list)
dim_eudaimonic = defaultdict(list)
for row in sims:
    dim_hedonic[row["dimension"]].append(float(row["sim_hedonic"]))
    dim_eudaimonic[row["dimension"]].append(float(row["sim_eudaimonic"]))

n_scales = len(mean_deltas)
print(f"\nN scales = {n_scales}")

print("\nLATEX S9 TABLE ROWS:")
for dim in dim_order:
    h_vals = dim_hedonic[dim]
    e_vals = dim_eudaimonic[dim]
    import statistics
    hm, hs_d = statistics.mean(h_vals), statistics.stdev(h_vals)
    em, es_d = statistics.mean(e_vals), statistics.stdev(e_vals)
    h_min, h_max = min(h_vals), max(h_vals)
    e_min, e_max = min(e_vals), max(e_vals)
    print(f"{dim:25s} & .{hm:.3f}[fix] & .{hs_d:.3f}[fix] & [{h_min:.2f}, {h_max:.2f}] & .{em:.3f}[fix] & .{es_d:.3f}[fix] & [{e_min:.2f}, {e_max:.2f}] \\\\")


# ── S10: Top 20 most-used scales ─────────────────────────────────
print("\n" + "=" * 70)
print("S10: TOP 20 MOST-USED SCALES")
print("=" * 70)

# Load usage weights
weights = {r["scale"]: float(r["weight"]) for r in load_csv("h1_usage_weights.csv")}

# Load usage_counts.csv and compute total pubs per scale
usage_rows = load_csv("usage_counts.csv")
total_pubs = defaultdict(int)
for r in usage_rows:
    total_pubs[r["scale"]] += int(r["count"])

# Load domain assignments
domains = {}
with open(Path(__file__).parent / "data" / "domains" / "domain_assignments.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        domains[r["scale"]] = r["domain"]

# Load scale full names
scale_names = {}
with open(Path(__file__).parent / "data" / "scales" / "clinical_scales_200.txt") as f:
    for line in f:
        line = line.strip()
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            scale_names[parts[0]] = parts[1]

# Sort by weight desc, take top 20
sorted_by_weight = sorted(weights.items(), key=lambda x: -x[1])[:20]

# Domain abbreviation map
domain_abbrev = {
    "Anxiety & Stress": "Anxiety",
    "Mood & Depression": "Mood",
    "Trauma & PTSD": "Trauma",
    "Substance Use & Addiction": "Substance Use",
    "Psychosis & Severe Mental Illness": "Psychosis",
    "General Distress & Psychopathology": "General Distr.",
    "Functional & Quality of Life": "Functional",
    "Eudaimonic & Flourishing": "Eudaimonic",
}

print()
hedonic_count = 0
eud_count = 0
for i, (scale, wt) in enumerate(sorted_by_weight, 1):
    delta = mean_deltas.get(scale, 0)
    pubs = total_pubs.get(scale, 0)
    domain = domains.get(scale, "?")
    name = scale_names.get(scale, "?")
    dom_short = domain_abbrev.get(domain, domain)
    sign = "H" if delta > 0 else "E"
    if delta > 0:
        hedonic_count += 1
    else:
        eud_count += 1
    print(f"  {i:2d}. {scale:12s}  pubs={pubs:>7,d}  wt={wt:.4f}  Δ={delta:+.3f} ({sign})  {dom_short}")

print(f"\n  Hedonic: {hedonic_count}/20  Eudaimonic: {eud_count}/20")

# LaTeX S10 rows
print(f"\nLATEX S10 TABLE ROWS:")
for i, (scale, wt) in enumerate(sorted_by_weight, 1):
    delta = mean_deltas.get(scale, 0)
    pubs = total_pubs.get(scale, 0)
    domain = domains.get(scale, "?")
    name = scale_names.get(scale, "?")
    dom_short = domain_abbrev.get(domain, domain)
    # Format pubs with comma thousands
    pubs_str = f"{pubs:,d}"
    print(f"{i:2d} & {scale} & {name} & {dom_short} & {pubs_str} & ${delta:+.3f}$ \\\\")

# ── Effect size range (d) for Discussion ─────────────────────────
print("\n" + "=" * 70)
print("MISC STATS FOR PAPER TEXT")
print("=" * 70)
wt_dims_list = load_csv("h1d_all_weighted_dim_stats.csv")
d_wt_values = [float(d["cohen_d_weighted"]) for d in wt_dims_list]
print(f"  Weighted d range: {min(d_wt_values):.2f}--{max(d_wt_values):.2f}")

# Time horizon raw d
raw_dims_list = load_csv("h1c_all_raw_dim_stats.csv")
for d in raw_dims_list:
    if d["dimension"] == "Time horizon":
        print(f"  Time horizon raw d = {float(d['cohen_d']):.2f}")
for d in wt_dims_list:
    if d["dimension"] == "Time horizon":
        print(f"  Time horizon weighted d = {float(d['cohen_d_weighted']):.2f}")

# Top-50 stats
h1a = load_json("h1a_topN_raw_overall.json")
print(f"  Top-50 raw: Δ={float(h1a['mean_delta']):+.3f}  d={float(h1a['cohen_d']):.2f}")

# N observations (all raw = scales * 6)
print(f"  N observations (all): {raw_overall['n_observations']}")

# Slope in per-year format for abstract
print(f"  Aggregate slope: {float(h2_summary['slope']):.4e}/yr")
print(f"  Aggregate slope × 1e4: {float(h2_summary['slope'])*1e4:.1f}")

# 3 strongest domain d values
print(f"\n  Top 3 domain d values:")
for d in sorted(dom_h1, key=lambda x: -float(x["cohen_d"]))[:3]:
    print(f"    {d['domain']}: d={float(d['cohen_d']):.2f}")

# Eudaimonic domain p
for d in dom_h1:
    if d["domain"] == "Eudaimonic & Flourishing":
        print(f"  Eudaimonic domain: Δ={float(d['mean_delta']):+.3f}  d={float(d['cohen_d']):.2f}  p={float(d['p_adjusted']):.3f}")

print("\nDONE - all stats extracted.")
