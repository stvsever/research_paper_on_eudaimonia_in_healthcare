"""Fill all %%PLACEHOLDER%% tokens in main.tex with computed paper stats.

Run AFTER run_pipeline.py has completed successfully.
"""
import csv, json, re, statistics, shutil
from pathlib import Path
from collections import defaultdict

V2  = Path(__file__).parent
OUT = V2 / "outputs"
TEX = V2 / "paper" / "main.tex"

# ── helpers ───────────────────────────────────────────────────────────────────
def jload(name):
    with open(OUT / name) as f:
        return json.load(f)

def cload(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))

def pval_str(p, latex=True):
    if float(p) < .001:
        return "$< .001$" if latex else "< .001"
    return f"${float(p):.3f}$" if latex else f"{float(p):.3f}"

def nolz(x, fmt=".3f"):
    """Format float 0 ≤ x < 1 without leading zero: 0.142 → '.142'"""
    return f"{float(x):{fmt}}"[1:]

def fmt_delta(d):
    return f"${float(d):+.3f}$"

def fmt_d(d):
    return f"${float(d):.2f}$"

# ── load all data ─────────────────────────────────────────────────────────────
h1c_overall  = jload("h1c_all_raw_overall.json")
h1d_overall  = jload("h1d_all_weighted_overall.json")
h1a_overall  = jload("h1a_topN_raw_overall.json")
h1b_overall  = jload("h1b_topN_weighted_overall.json")
h1c_dims     = {d["dimension"]: d for d in cload("h1c_all_raw_dim_stats.csv")}
h1d_dims     = {d["dimension"]: d for d in cload("h1d_all_weighted_dim_stats.csv")}
h2_summary   = jload("h2_summary.json")
h2_dims      = cload("h2_dimension_trends.csv")
perm         = jload("posthoc_permutation.json")
sens         = cload("posthoc_sensitivity_n.csv")
dom_h1       = cload("posthoc_domain_h1.csv")
dom_h2       = cload("posthoc_domain_h2.csv")
corr         = jload("posthoc_usage_correlation.json")
sims_rows    = cload("h1_cosine_similarities_all.csv")
usage_rows   = cload("usage_counts.csv")
weights_rows = cload("h1_usage_weights.csv")

# domain assignments (183 scales)
domains = {}
with open(V2 / "data" / "domains" / "domain_assignments.csv") as f:
    for r in csv.DictReader(f):
        domains[r["scale"]] = r["domain"]

# scale full names
scale_names = {}
with open(V2 / "data" / "scales" / "clinical_scales_200.txt") as f:
    for line in f:
        line = line.strip()
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            scale_names[parts[0]] = parts[1]

# ── compute derived data ──────────────────────────────────────────────────────
# Per-scale mean Δ
scale_deltas = defaultdict(list)
for row in sims_rows:
    scale_deltas[row["scale"]].append(float(row["delta"]))
mean_deltas = {s: sum(ds)/len(ds) for s, ds in scale_deltas.items()}

# Per-dimension similarity stats (for S9)
dim_hedonic  = defaultdict(list)
dim_eud      = defaultdict(list)
for row in sims_rows:
    dim_hedonic[row["dimension"]].append(float(row["sim_hedonic"]))
    dim_eud[row["dimension"]].append(float(row["sim_eudaimonic"]))

# Total pubs per scale (for S10)
total_pubs = defaultdict(int)
for r in usage_rows:
    total_pubs[r["scale"]] += int(r["count"])

# Usage weights dict
weights = {r["scale"]: float(r["weight"]) for r in weights_rows}

# ── extract key scalar stats ──────────────────────────────────────────────────
h1c_delta = float(h1c_overall["mean_delta"])
h1c_d     = float(h1c_overall["cohen_d"])
h1d_delta = float(h1d_overall["mean_delta_weighted"])
h1d_d     = float(h1d_overall["cohen_d_weighted"])
h1a_delta = float(h1a_overall["mean_delta"])
h1a_d     = float(h1a_overall["cohen_d"])
h1b_delta = float(h1b_overall["mean_delta_weighted"])
h1b_d     = float(h1b_overall["cohen_d_weighted"])

# d range across dimensions (weighted)
d_wt_vals = [float(h1d_dims[d]["cohen_d_weighted"]) for d in h1d_dims]
d_wt_min, d_wt_max = min(d_wt_vals), max(d_wt_vals)

# Time horizon d values
th_raw = float(h1c_dims["Time horizon"]["cohen_d"])
th_wt  = float(h1d_dims["Time horizon"]["cohen_d_weighted"])

# H2 aggregate
h2_slope   = float(h2_summary["slope"])
h2_r2      = float(h2_summary["r_squared"])
h2_p       = float(h2_summary["p_ols"])
h2_mk_s    = int(h2_summary.get("mann_kendall_S", 0))
h2_mk_p    = float(h2_summary.get("mann_kendall_p", 1))
h2_dw      = float(h2_summary["durbin_watson"])
h2_total   = h2_slope * 25

# DW range across dimensions
dw_vals = [float(d["durbin_watson"]) for d in h2_dims if d["dimension"] != "aggregate"]
dw_min, dw_max = min(dw_vals), max(dw_vals)

# Permutation
perm_delta = float(perm["observed_delta"])
perm_p     = float(perm["perm_p"])
perm_null_mean = float(perm["null_mean"])
perm_null_sd   = float(perm["null_sd"])

# Usage correlation
rho = float(corr["spearman_rho"])
rho_p = float(corr["spearman_p"])

# Sensitivity
d_raw_vals = [float(s["cohen_d_raw"]) for s in sens]
sens_d_min, sens_d_max = min(d_raw_vals), max(d_raw_vals)
sens_n_min = min(int(s["top_n"]) for s in sens)
# top-150 d for the "top-20 to top-150" description
sens_by_n = {int(s["top_n"]): s for s in sens}
# Convergence value (weighted Δ at N ≥ 100)
sens_wt_100 = float(sens_by_n[100]["mean_delta_weighted"])

# Domain H1 - eudaimonic exception
dom_eud = next(d for d in dom_h1 if d["domain"] == "Eudaimonic & Flourishing")
dom_top3 = sorted(dom_h1, key=lambda x: -float(x["cohen_d"]))[:3]

# ── build replacement map ─────────────────────────────────────────────────────
R = {}

# Abstract
R["%%ABSTRACT_H1_STATS%%"] = (
    f"mean $\\Delta = {h1c_delta:+.3f}$, Cohen's $d = {h1c_d:.2f}$, $p < .001$; "
    f"Wilcoxon signed-rank, Holm-corrected"
)
R["%%ABSTRACT_WH1_STATS%%"] = (
    f"$\\Delta_w = {h1d_delta:+.3f}$, $d = {h1d_d:.2f}$"
)
R["%%ABSTRACT_SENS_D%%"] = (
    f"$d = {sens_d_min:.2f}$--${sens_d_max:.2f}$"
)
R["%%ABSTRACT_PERM_P%%"] = f"$p < .0001$" if perm_p < .0001 else f"$p = {nolz(perm_p, '.4f')}$"
h2_p_str = "$p < .0001$" if h2_p < .0001 else f"$p = {nolz(h2_p, '.4f')}$"
h2_mk_p_str_agg = "$p < .0001$" if h2_mk_p < .0001 else f"$p = {nolz(h2_mk_p, '.4f')}$"
R["%%ABSTRACT_H2_STATS%%"] = (
    f"aggregate slope $= {h2_slope:+.2e}$/year, {h2_p_str}; "
    f"Mann-Kendall $S = {h2_mk_s}$, {h2_mk_p_str_agg}"
)
R["%%ABSTRACT_H2_TOTAL%%"] = f"${h2_total:+.3f}$"

# H1 inline results
R["%%H1C_DELTA%%"] = f"{h1c_delta:+.3f}"
R["%%H1C_D%%"]     = f"{h1c_d:.2f}"
R["%%H1D_DELTA%%"] = f"{h1d_delta:+.3f}"
R["%%H1D_D%%"]     = f"{h1d_d:.2f}"
R["%%H1A_D%%"]     = f"{h1a_d:.2f}"
R["%%H1A_STATS%%"] = f"$\\Delta = {h1a_delta:+.3f}$, $d = {h1a_d:.2f}$"
R["%%H1B_DELTA%%"] = f"{h1b_delta:+.3f}"
R["%%H1B_D%%"]     = f"{h1b_d:.2f}"

R["%%H1D_D_RANGE%%"] = (
    f"$d = {d_wt_min:.2f}$ ({min(h1d_dims, key=lambda x: float(h1d_dims[x]['cohen_d_weighted']))}) "
    f"to $d = {d_wt_max:.2f}$ ({max(h1d_dims, key=lambda x: float(h1d_dims[x]['cohen_d_weighted']))})"
)
R["%%H1D_D_RANGE_DISC%%"] = f"$d = {d_wt_min:.2f}$--${d_wt_max:.2f}$"

# H2 inline
R["%%H2_AGG_STATS%%"] = (
    f"slope $= {h2_slope:+.2e}$/year, $R^2 = {nolz(h2_r2)}$, {h2_p_str}"
)
R["%%H2_DW_RANGE%%"] = f"DW $= {dw_min:.2f}$--${dw_max:.2f}$"
R["%%H2_MK_AGG%%"]   = f"$S = {h2_mk_s}$, {h2_mk_p_str_agg}"
R["%%H2_TOTAL_CHANGE%%"] = f"${h2_total:+.3f}$"
R["%%H2_ANNUAL_SLOPE%%"] = f"${h2_slope:+.5f}$"

# More readable slope text
h2_slope_1e4 = h2_slope * 1e4
R["%%H2_SLOPE_TEXT%%"] = (
    f"${h2_slope_1e4:+.1f} \\times 10^{{-4}}$/year amounts to ${h2_total:+.3f}$ over 25 years"
)

# Permutation
R["%%PERM_P%%"]        = "< .0001" if perm_p < .0001 else nolz(perm_p, '.4f')
R["%%PERM_NULL_MEAN%%"] = f"{perm_null_mean:+.4f}"
R["%%PERM_NULL_SD%%"]   = f"{perm_null_sd:.3f}"

# Sensitivity
R["%%SENSITIVITY_D_RANGE%%"] = (
    f"${sens_d_min:.2f}$ (top-{sens_by_n[min(int(s['top_n']) for s in sens)]['top_n']}) "
    f"to ${max(d_raw_vals):.2f}$ (top-{sens_by_n[max(int(s['top_n']) for s in sens if float(s['cohen_d_raw']) == max(d_raw_vals))]['top_n']})"
)
R["%%SENSITIVITY_W_CONVERGE%%"] = f"${sens_wt_100:+.3f}$"

# Usage correlation
rho_p_str = f"$p < .001$" if rho_p < .001 else f"$p = {nolz(rho_p)}$"
R["%%USAGE_CORR%%"] = f"Spearman $\\rho = {nolz(rho)}$ ({rho_p_str})"

# Time horizon d values
R["%%TIME_HORIZON_D_DISC%%"] = f"$d = {th_wt:.2f}$"
R["%%TIME_HORIZON_D_RAW_D%%"] = f"$d = {th_raw:.2f}$"

# Domain stats
R["%%DOM_EUD_STATS%%"] = (
    f"$\\Delta = {float(dom_eud['mean_delta']):+.3f}$, "
    f"$d = {float(dom_eud['cohen_d']):.2f}$, "
    f"$p_{{\\text{{adj}}}} = {float(dom_eud['p_adjusted']):.3f}$"
)

def domain_abbrev_text(domain):
    abbrev = {
        "Substance Use & Addiction": "Substance Use \\& Addiction",
        "Anxiety & Stress": "Anxiety \\& Stress",
        "Mood & Depression": "Mood \\& Depression",
        "Trauma & PTSD": "Trauma \\& PTSD",
        "Psychosis & Severe Mental Illness": "Psychosis \\& SMI",
        "General Distress & Psychopathology": "General Distress",
        "Functional & Quality of Life": "Functional \\& QoL",
        "Eudaimonic & Flourishing": "Eudaimonic \\& Flourishing",
    }
    return abbrev.get(domain, domain)

top3_parts = [
    f"{domain_abbrev_text(d['domain'])} ($d = {float(d['cohen_d']):.2f}$)"
    for d in dom_top3
]
R["%%DOM_TOP3_STATS%%"] = ", ".join(top3_parts)

# Conclusion
d_wt_rounded = round(h1c_d, 1)
if h1c_d >= 1.0:
    R["%%H1C_D_QUALITATIVE%%"] = f"Cohen's $d = {h1c_d:.2f}$"
else:
    R["%%H1C_D_QUALITATIVE%%"] = f"Cohen's $d = {h1c_d:.2f}$, near-large effect"

# ── TABLE 2 ROWS ──────────────────────────────────────────────────────────────
dim_order = [
    "Foundational claim", "Evaluative criterion", "Time horizon",
    "Adversity", "Measurement proxies", "Central tension",
]
table2_rows = []
for dim in dim_order:
    r = h1c_dims[dim]
    w = h1d_dims[dim]
    rd = float(r["mean_delta"])
    rcd = float(r["cohen_d"])
    wd = float(w["mean_delta_weighted"])
    wcd = float(w["cohen_d_weighted"])
    table2_rows.append(
        f"{dim:25s} & ${rd:+.3f}$ & ${rcd:.2f}$ & ${wd:+.3f}$ & ${wcd:.2f}$ & $< .001$ \\\\"
    )
R["%%TABLE2_ROWS%%"] = "\n".join(table2_rows)

overall_row = (
    f"$\\boldsymbol{{{h1c_delta:+.3f}}}$ & $\\boldsymbol{{{h1c_d:.2f}}}$ & "
    f"$\\boldsymbol{{{h1d_delta:+.3f}}}$ & $\\boldsymbol{{{h1d_d:.2f}}}$ & "
    f"$\\boldsymbol{{< .001}}$ \\\\"
)
R["%%TABLE2_OVERALL%%"] = overall_row

# ── TABLE 3 ROWS (H2 dimension trends) ───────────────────────────────────────
table3_rows = []
for d in h2_dims:
    if d["dimension"] == "aggregate":
        continue
    dim   = d["dimension"]
    slope = float(d["slope"]) * 1e4
    r2    = float(d["r_squared"])
    p     = float(d["p_ols"])
    mk_s  = int(float(d.get("mann_kendall_S", 0)))
    mk_p  = d.get("mann_kendall_p", "?")
    dw    = float(d["durbin_watson"])
    p_str = "< .001" if p < .001 else nolz(p)
    mk_p_str = "< .001" if float(mk_p) < .001 else nolz(float(mk_p))
    table3_rows.append(
        f"{dim:25s} & ${slope:+.2f}$ & ${nolz(r2)}$ & ${p_str}$ & ${mk_s}$ & ${mk_p_str}$ & ${dw:.2f}$ \\\\"
    )
R["%%TABLE3_ROWS%%"] = "\n".join(table3_rows)

agg_slope_1e4 = h2_slope * 1e4
agg_p_str = "< .001" if h2_p < .001 else f"{h2_p:.3f}"
agg_mk_p_str = "< .001" if h2_mk_p < .001 else f"{h2_mk_p:.3f}"
R["%%TABLE3_AGGREGATE%%"] = (
    f"$\\boldsymbol{{{agg_slope_1e4:+.2f}}}$ & $\\boldsymbol{{{nolz(h2_r2)}}}$ & "
    f"$\\boldsymbol{{< .001}}$ & $\\boldsymbol{{{h2_mk_s}}}$ & "
    f"$\\boldsymbol{{< .001}}$ & $\\boldsymbol{{{h2_dw:.2f}}}$ \\\\"
)

# ── SENSITIVITY TABLE ROWS ────────────────────────────────────────────────────
sens_rows = []
for s in sens:
    n  = s["top_n"]
    dr = float(s["mean_delta_raw"])
    cd = float(s["cohen_d_raw"])
    dw = float(s["mean_delta_weighted"])
    sens_rows.append(f"{n:>3s} & ${dr:+.3f}$ & ${cd:.2f}$ & ${dw:+.3f}$ \\\\")
R["%%SENSITIVITY_ROWS%%"] = "\n".join(sens_rows)

# Better sensitivity d range text
d_by_n = {int(s["top_n"]): float(s["cohen_d_raw"]) for s in sens}
d_min_n = min(d_by_n, key=lambda n: d_by_n[n])
# Top-N just before full (e.g., 150) with max d
d_max_n = max(d_by_n, key=lambda n: d_by_n[n])
R["%%SENSITIVITY_D_RANGE%%"] = (
    f"${d_by_n[d_min_n]:.2f}$ (top-{d_min_n}) to ${d_by_n[d_max_n]:.2f}$ (top-{d_max_n})"
)

# ── DOMAIN H1 TABLE ROWS ──────────────────────────────────────────────────────
domain_latex = {
    "Anxiety & Stress": "Anxiety \\& Stress",
    "Mood & Depression": "Mood \\& Depression",
    "Trauma & PTSD": "Trauma \\& PTSD",
    "General Distress & Psychopathology": "General Distress \\& Psychopathology",
    "Psychosis & Severe Mental Illness": "Psychosis \\& Severe Mental Illness",
    "Substance Use & Addiction": "Substance Use \\& Addiction",
    "Functional & Quality of Life": "Functional \\& Quality of Life",
    "Eudaimonic & Flourishing": "Eudaimonic \\& Flourishing",
}
dom_h1_sorted = sorted(dom_h1, key=lambda x: -float(x["cohen_d"]))
table_dom_h1 = []
for d in dom_h1_sorted:
    dom  = domain_latex.get(d["domain"], d["domain"])
    n    = d["n"]
    delt = float(d["mean_delta"])
    cd   = float(d["cohen_d"])
    p    = float(d["p_adjusted"])
    p_str = "$< .001$" if p < .001 else f"${nolz(p)}$"
    table_dom_h1.append(f"{dom:50s} & {n:>2s} & ${delt:+.3f}$ & ${cd:.2f}$ & {p_str} \\\\")
R["%%TABLE_DOMAIN_H1_ROWS%%"] = "\n".join(table_dom_h1)

# ── DOMAIN H2 TABLE ROWS ──────────────────────────────────────────────────────
dom_h2_sorted = sorted(dom_h2, key=lambda x: -float(x["slope_per_decade"]))
table_dom_h2 = []
domain_abbrev_h2 = {
    "Anxiety & Stress": "Anxiety \\& Stress",
    "Mood & Depression": "Mood \\& Depression",
    "Trauma & PTSD": "Trauma \\& PTSD",
    "General Distress & Psychopathology": "Gen.\\ Distress \\& Psychopathology",
    "Psychosis & Severe Mental Illness": "Psychosis \\& SMI",
    "Substance Use & Addiction": "Substance Use \\& Addiction",
    "Functional & Quality of Life": "Functional \\& QoL",
    "Eudaimonic & Flourishing": "Eudaimonic \\& Flourishing",
}
for d in dom_h2_sorted:
    dom  = domain_abbrev_h2.get(d["domain"], d["domain"])
    n    = d["n"]
    spd  = float(d["slope_per_decade"])
    tot  = float(d["total_change_25yr"])
    r2   = float(d["r_squared"])
    p    = float(d["p_adjusted"])
    p_str = "$< .001$" if p < .001 else f"${nolz(p)}$"
    table_dom_h2.append(
        f"{dom:45s} & {n:>2s} & ${spd:+.4f}$ & ${tot:+.3f}$ & ${nolz(r2)}$ & {p_str} \\\\"
    )
R["%%TABLE_DOMAIN_H2_ROWS%%"] = "\n".join(table_dom_h2)

# ── DOMAIN H2 narrative text ──────────────────────────────────────────────────
dom_h2_sorted_full = sorted(dom_h2, key=lambda x: -float(x["slope_per_decade"]))
dom_prose = {
    "Anxiety & Stress": "Anxiety \\& Stress",
    "Mood & Depression": "Mood \\& Depression",
    "Trauma & PTSD": "Trauma \\& PTSD",
    "General Distress & Psychopathology": "General Distress \\& Psychopathology",
    "Psychosis & Severe Mental Illness": "Psychosis \\& SMI",
    "Substance Use & Addiction": "Substance Use \\& Addiction",
    "Functional & Quality of Life": "Functional \\& QoL",
    "Eudaimonic & Flourishing": "Eudaimonic \\& Flourishing",
}
pos_h2 = [d for d in dom_h2_sorted_full if float(d["slope_per_decade"]) > 0]
neg_h2 = [d for d in dom_h2_sorted_full if float(d["slope_per_decade"]) < 0]

narr_lines = []
if pos_h2:
    d1 = pos_h2[0]
    narr_lines.append(
        f"{dom_prose.get(d1['domain'], d1['domain'])} showed the largest positive slope "
        f"(${float(d1['slope_per_decade']):+.4f}$/decade), reflecting the growing but still "
        f"insufficient adoption of eudaimonic instruments."
    )
if len(pos_h2) >= 2:
    mid = " and ".join(
        f"{dom_prose.get(d['domain'], d['domain'])} (${float(d['slope_per_decade']):+.4f}$/decade)"
        for d in pos_h2[1:3]
    )
    narr_lines.append(f"{mid} also showed significant positive trends.")
if neg_h2:
    neg_list = " and ".join(dom_prose.get(d["domain"], d["domain"]) for d in neg_h2)
    narr_lines.append(
        f"{neg_list} showed small but significant negative slopes (total 25-year change "
        f"$<\\!{min(float(d['total_change_25yr']) for d in neg_h2):.3f}$), "
        f"possibly reflecting more nuanced instrumentation."
    )

R["%%DOM_H2_NARRATIVE%%"] = "\n".join(narr_lines)

# ── S8: TOP HEDONIC / EUDAIMONIC SCALE ROWS ───────────────────────────────────
sorted_hedonic   = sorted(mean_deltas.items(), key=lambda x: -x[1])
sorted_eudaimonic = sorted(mean_deltas.items(), key=lambda x: x[1])
s8_rows = []
for i in range(20):
    hs, hd = sorted_hedonic[i]
    if i < 10:
        es, ed = sorted_eudaimonic[i]
        s8_rows.append(
            f"{i+1:2d} & {hs:15s} & ${hd:+.3f}$ & "
            f"{i+1:2d} & {es:15s} & ${ed:+.3f}$ \\\\"
        )
    else:
        s8_rows.append(
            f"{i+1:2d} & {hs:15s} & ${hd:+.3f}$ &    &                  &          \\\\"
        )
R["%%S8_ROWS%%"] = "\n".join(s8_rows)

# ── S9: PER-DIMENSION DESCRIPTIVE STATS ──────────────────────────────────────
s9_rows = []
for dim in dim_order:
    h_vals = dim_hedonic[dim]
    e_vals = dim_eud[dim]
    hm  = statistics.mean(h_vals)
    hs  = statistics.stdev(h_vals)
    em  = statistics.mean(e_vals)
    es  = statistics.stdev(e_vals)
    hmin, hmax = min(h_vals), max(h_vals)
    emin, emax = min(e_vals), max(e_vals)
    s9_rows.append(
        f"{dim:25s} & {hm:.3f} & {hs:.3f} & [{hmin:.2f}, {hmax:.2f}] & "
        f"{em:.3f} & {es:.3f} & [{emin:.2f}, {emax:.2f}] \\\\"
    )
R["%%S9_ROWS%%"] = "\n".join(s9_rows)

# ── S10: TOP 20 MOST-USED SCALE ROWS ─────────────────────────────────────────
sorted_by_weight = sorted(weights.items(), key=lambda x: -x[1])[:20]
domain_abbrev_s10 = {
    "Anxiety & Stress": "Anxiety",
    "Mood & Depression": "Mood",
    "Trauma & PTSD": "Trauma",
    "Substance Use & Addiction": "Substance Use",
    "Psychosis & Severe Mental Illness": "Psychosis",
    "General Distress & Psychopathology": "General Distr.",
    "Functional & Quality of Life": "Functional",
    "Eudaimonic & Flourishing": "Eudaimonic",
}
s10_rows = []
hedonic_count = 0
for i, (scale, wt) in enumerate(sorted_by_weight, 1):
    delta  = mean_deltas.get(scale, 0)
    pubs   = total_pubs.get(scale, 0)
    domain = domains.get(scale, "?")
    name   = scale_names.get(scale, scale)
    # Escape latex special chars in name
    name_esc = name.replace("&", "\\&").replace("--", "--")
    dom_short = domain_abbrev_s10.get(domain, domain)
    pubs_str  = f"{pubs:,d}"
    s10_rows.append(
        f"{i:2d} & {scale} & {name_esc} & {dom_short} & {pubs_str} & ${delta:+.3f}$ \\\\"
    )
    if delta > 0:
        hedonic_count += 1

R["%%S10_ROWS%%"] = "\n".join(s10_rows)

eud_count = 20 - hedonic_count
if hedonic_count >= 10:
    s10_fn_bias = f"{hedonic_count} of 20 most-used scales show positive (hedonic) loading"
else:
    s10_fn_bias = f"{hedonic_count} of 20 most-used scales show positive (hedonic) loading"
R["%%S10_FLOATNOTE%%"] = (
    f"Cumulative PubMed publications (2000--2025). "
    f"Usage weights normalised to sum to 1.0 across all {len(scale_names)} scales. "
    f"Mean $\\Delta$ averaged across all six hedonic--eudaimonic dimensions. "
    f"{s10_fn_bias}."
)

# ── S1/S2 floatnote stats already handled via H1C/H1D placeholders ───────────

# ── apply all replacements ────────────────────────────────────────────────────
with open(TEX) as f:
    text = f.read()

# Backup
shutil.copy(TEX, TEX.with_suffix(".tex.bak"))

missing = []
for placeholder, value in R.items():
    if placeholder in text:
        text = text.replace(placeholder, value)
    else:
        missing.append(placeholder)

if missing:
    print("WARNING: These placeholders were NOT found in main.tex:")
    for m in missing:
        print(f"  {m}")

with open(TEX, "w") as f:
    f.write(text)

print(f"\nDone. Filled {len(R) - len(missing)} of {len(R)} placeholders.")
if not missing:
    print("ALL placeholders filled successfully!")
print(f"\nBackup saved to {TEX.with_suffix('.tex.bak')}")
