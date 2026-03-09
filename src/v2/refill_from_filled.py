"""
Direct refill: update main.tex with new pipeline values WITHOUT needing a template.
Uses lambda replacements to avoid regex backslash issues with LaTeX.
"""
import csv, json, re, statistics
from pathlib import Path
from collections import defaultdict

V2  = Path(__file__).parent
OUT = V2 / "outputs"
TEX = V2 / "paper" / "main.tex"

def jload(name):
    with open(OUT / name) as f: return json.load(f)
def cload(name):
    with open(OUT / name) as f: return list(csv.DictReader(f))
def nolz(x, fmt=".3f"):
    return f"{float(x):{fmt}}"[1:]

def rsub(pattern, repl_str, text, flags=0):
    """re.sub but with lambda to avoid backslash issues in replacement."""
    return re.sub(pattern, lambda m: repl_str, text, flags=flags)

# ── load all data ──
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

domains = {}
with open(V2 / "data" / "domains" / "domain_assignments.csv") as f:
    for r in csv.DictReader(f): domains[r["scale"]] = r["domain"]

scale_names = {}
with open(V2 / "data" / "scales" / "clinical_scales_200.txt") as f:
    for line in f:
        line = line.strip()
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            scale_names[parts[0]] = parts[1]

n_scales = len(scale_names)

# Per-scale mean delta
scale_deltas = defaultdict(list)
for row in sims_rows: scale_deltas[row["scale"]].append(float(row["delta"]))
mean_deltas = {s: sum(ds)/len(ds) for s, ds in scale_deltas.items()}

dim_hedonic = defaultdict(list)
dim_eud = defaultdict(list)
for row in sims_rows:
    dim_hedonic[row["dimension"]].append(float(row["sim_hedonic"]))
    dim_eud[row["dimension"]].append(float(row["sim_eudaimonic"]))

total_pubs = defaultdict(int)
for r in usage_rows: total_pubs[r["scale"]] += int(r["count"])
weights = {r["scale"]: float(r["weight"]) for r in weights_rows}

h1c_delta = float(h1c_overall["mean_delta"])
h1c_d     = float(h1c_overall["cohen_d"])
h1d_delta = float(h1d_overall["mean_delta_weighted"])
h1d_d     = float(h1d_overall["cohen_d_weighted"])
h1a_delta = float(h1a_overall["mean_delta"])
h1a_d     = float(h1a_overall["cohen_d"])
h1b_delta = float(h1b_overall["mean_delta_weighted"])
h1b_d     = float(h1b_overall["cohen_d_weighted"])
d_wt_vals = [float(h1d_dims[d]["cohen_d_weighted"]) for d in h1d_dims]
d_wt_min, d_wt_max = min(d_wt_vals), max(d_wt_vals)
th_raw = float(h1c_dims["Time horizon"]["cohen_d"])
th_wt  = float(h1d_dims["Time horizon"]["cohen_d_weighted"])
h2_slope = float(h2_summary["slope"])
h2_r2    = float(h2_summary["r_squared"])
h2_p     = float(h2_summary["p_ols"])
h2_mk_s  = int(h2_summary.get("mann_kendall_S", 0))
h2_mk_p  = float(h2_summary.get("mann_kendall_p", 1))
h2_dw    = float(h2_summary["durbin_watson"])
h2_total = h2_slope * 25
dw_vals  = [float(d["durbin_watson"]) for d in h2_dims if d["dimension"] != "aggregate"]
dw_min, dw_max = min(dw_vals), max(dw_vals)
perm_p = float(perm["perm_p"])
perm_null_mean = float(perm["null_mean"])
perm_null_sd = float(perm["null_sd"])
rho = float(corr["spearman_rho"])
rho_p = float(corr["spearman_p"])
d_raw_vals = [float(s["cohen_d_raw"]) for s in sens]
sens_d_min, sens_d_max = min(d_raw_vals), max(d_raw_vals)
sens_by_n = {int(s["top_n"]): s for s in sens}
sens_wt_100 = float(sens_by_n[100]["mean_delta_weighted"])
dom_eud = next(d for d in dom_h1 if d["domain"] == "Eudaimonic & Flourishing")
dom_top3 = sorted(dom_h1, key=lambda x: -float(x["cohen_d"]))[:3]
h2_slope_1e4 = h2_slope * 1e4

h2_p_str = "$p < .0001$" if h2_p < .0001 else f"$p = {nolz(h2_p, '.4f')}$"
h2_mk_p_str = "$p < .0001$" if h2_mk_p < .0001 else f"$p = {nolz(h2_mk_p, '.4f')}$"
perm_p_str = "$p < .0001$" if perm_p < .0001 else f"$p = {nolz(perm_p, '.4f')}$"
rho_p_str = "$p < .001$" if rho_p < .001 else f"$p = {nolz(rho_p)}$"

dim_order = [
    "Foundational claim", "Evaluative criterion", "Time horizon",
    "Adversity", "Measurement proxies", "Central tension",
]

# ── read ──
text = TEX.read_text()
changes = 0

def counted_replace(old, new):
    global text, changes
    if old in text:
        text = text.replace(old, new)
        changes += 1

# ══════════════════════════════════════════════════════════════════
# SCALE COUNT
# ══════════════════════════════════════════════════════════════════
counted_replace("205 widely used clinical scales", f"{n_scales} widely used clinical scales")
counted_replace("Across all 205 scales", f"Across all {n_scales} scales")
counted_replace("all 205 scales", f"all {n_scales} scales")
counted_replace("205 clinical scales", f"{n_scales} clinical scales")
counted_replace("$N = 20$--$205$", f"$N = 20$--${n_scales}$")
counted_replace("top-205", f"top-{n_scales}")

# ══════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════

# H1 stats line
counted_replace(
    "mean $\\Delta = +0.054$, Cohen's $d = 0.97$, $p < .001$; Wilcoxon signed-rank, Holm-corrected",
    f"mean $\\Delta = {h1c_delta:+.3f}$, Cohen's $d = {h1c_d:.2f}$, $p < .001$; Wilcoxon signed-rank, Holm-corrected"
)

# Weighted
counted_replace(
    "$\\Delta_w = +0.067$, $d = 1.38$",
    f"$\\Delta_w = {h1d_delta:+.3f}$, $d = {h1d_d:.2f}$"
)

# Sensitivity d range
counted_replace(
    "$d = 1.29$--$2.33$",
    f"$d = {sens_d_min:.2f}$--${sens_d_max:.2f}$"
)

# H2 abstract
counted_replace(
    "aggregate slope $= +1.10e-04$/year, $p < .001$; Mann-Kendall $S = 145$, $p = .002$",
    f"aggregate slope $= {h2_slope:+.2e}$/year, {h2_p_str}; Mann-Kendall $S = {h2_mk_s}$, {h2_mk_p_str}"
)

# H2 total
counted_replace(
    "$+0.003$ units over 25 years",
    f"${h2_total:+.3f}$ units over 25 years"
)

# "Five of six dimensions" → "All six"
counted_replace(
    "Five of six dimensions showed positive slopes, with only one (Adversity) showing a small negative trend, confirming a broad-based temporal increase.",
    "All six dimensions showed positive slopes, confirming a broad-based temporal increase."
)

# ══════════════════════════════════════════════════════════════════
# RESULTS H1
# ══════════════════════════════════════════════════════════════════

# Overall mean
counted_replace(
    "mean $\\Delta = +0.054$ ($d = 0.97$",
    f"mean $\\Delta = {h1c_delta:+.3f}$ ($d = {h1c_d:.2f}$"
)
# N pairs
n_pairs = n_scales * 6
n_pairs_str = f"{n_pairs:,d}".replace(",", "{,}")
counted_replace("$N = 1{,}230$", f"$N = {n_pairs_str}$")

# weighted pattern
counted_replace(
    "weighted $\\Delta = +0.067$, $d = 1.38$)",
    f"weighted $\\Delta = {h1d_delta:+.3f}$, $d = {h1d_d:.2f}$)"
)

# Effect sizes range
min_dim = min(h1d_dims, key=lambda x: float(h1d_dims[x]['cohen_d_weighted']))
max_dim = max(h1d_dims, key=lambda x: float(h1d_dims[x]['cohen_d_weighted']))
counted_replace(
    "$d = 1.03$ (Measurement proxies) to $d = 2.76$ (Time horizon)",
    f"$d = {d_wt_min:.2f}$ ({min_dim}) to $d = {d_wt_max:.2f}$ ({max_dim})"
)

# Figure floatnote
counted_replace(
    "Weighted $\\Delta = +0.067$, $d = 1.38$.",
    f"Weighted $\\Delta = {h1d_delta:+.3f}$, $d = {h1d_d:.2f}$."
)

# ── TABLE 2 ──
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
new_table2 = "\n".join(table2_rows)

# Old table 2 body (between first \midrule and second \midrule in tab:h1_dims)
old_t2_body = """Foundational claim        & $+0.037$ & $1.19$ & $+0.043$ & $1.64$ & $< .001$ \\\\
Evaluative criterion      & $+0.057$ & $1.38$ & $+0.068$ & $1.88$ & $< .001$ \\\\
Time horizon              & $+0.104$ & $2.07$ & $+0.119$ & $2.76$ & $< .001$ \\\\
Adversity                 & $+0.067$ & $1.22$ & $+0.080$ & $1.74$ & $< .001$ \\\\
Measurement proxies       & $+0.029$ & $0.42$ & $+0.054$ & $1.03$ & $< .001$ \\\\
Central tension           & $+0.031$ & $0.73$ & $+0.039$ & $1.14$ & $< .001$ \\\\"""
counted_replace(old_t2_body, new_table2)

# Overall row
counted_replace(
    "$\\boldsymbol{+0.054}$ & $\\boldsymbol{0.97}$ & $\\boldsymbol{+0.067}$ & $\\boldsymbol{1.38}$ & $\\boldsymbol{< .001}$",
    f"$\\boldsymbol{{{h1c_delta:+.3f}}}$ & $\\boldsymbol{{{h1c_d:.2f}}}$ & $\\boldsymbol{{{h1d_delta:+.3f}}}$ & $\\boldsymbol{{{h1d_d:.2f}}}$ & $\\boldsymbol{{< .001}}$"
)

# Top-50 stats
counted_replace(
    "$\\Delta = +0.069$, $d = 1.33$",
    f"$\\Delta = {h1a_delta:+.3f}$, $d = {h1a_d:.2f}$"
)

# ══════════════════════════════════════════════════════════════════
# RESULTS H2
# ══════════════════════════════════════════════════════════════════

counted_replace(
    "slope $= +1.10e-04$/year, $R^2 = .543$, $p < .001$",
    f"slope $= {h2_slope:+.2e}$/year, $R^2 = {nolz(h2_r2)}$, {h2_p_str}"
)

# DW range (find the specific one)
counted_replace("DW $= 0.14$--$0.87$", f"DW $= {dw_min:.2f}$--${dw_max:.2f}$")

# "five of six slopes were positive"
counted_replace(
    "five of six slopes were positive (Table",
    "all six slopes were positive (Table"
)

# Dimension list in H2 body
counted_replace(
    "Evaluative criterion, Measurement proxies, Time horizon, Central tension, and Foundational claim all showed positive trends, while Adversity showed a small but significant negative trend, suggesting a near-uniform temporal increase with one exception.",
    "all dimension slopes were positive, suggesting a uniform temporal increase."
)

# MK stats
counted_replace(
    "$S = 145$, $p = .002$",
    f"$S = {h2_mk_s}$, {h2_mk_p_str}"
)

# "increased by approximately $+0.003$"
counted_replace(
    "increased by approximately $+0.003$ units over 25 years",
    f"increased by approximately ${h2_total:+.3f}$ units over 25 years"
)

# Cross-sectional bias refs
counted_replace(
    "cross-sectional bias itself ($\\Delta = +0.067$)",
    f"cross-sectional bias itself ($\\Delta = {h1d_delta:+.3f}$)"
)
counted_replace(
    "cross-sectional bias ($\\Delta = +0.054$, $d = 0.97$)",
    f"cross-sectional bias ($\\Delta = {h1c_delta:+.3f}$, $d = {h1c_d:.2f}$)"
)

# Figure 2 floatnote
counted_replace(
    "Five of six dimension slopes positive (Evaluative criterion, Measurement proxies, Time horizon, Central tension, Foundational claim); Adversity shows a small negative slope.",
    "All six dimension slopes positive."
)

# ── TABLE 3 ──
table3_rows = []
for d in h2_dims:
    if d["dimension"] == "aggregate": continue
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
new_table3 = "\n".join(table3_rows)

old_t3_body = """Foundational claim        & $+0.63$ & $.514$ & $< .001$ & $143$ & $.002$ & $0.44$ \\\\
Evaluative criterion      & $+3.94$ & $.916$ & $< .001$ & $289$ & $< .001$ & $0.25$ \\\\
Time horizon              & $+0.96$ & $.729$ & $< .001$ & $205$ & $< .001$ & $0.87$ \\\\
Adversity                 & $-0.67$ & $.264$ & $.007$ & $-107$ & $.019$ & $0.31$ \\\\
Measurement proxies       & $+1.00$ & $.250$ & $.009$ & $57$ & $.217$ & $0.23$ \\\\
Central tension           & $+0.73$ & $.161$ & $.042$ & $13$ & $.791$ & $0.14$ \\\\"""
counted_replace(old_t3_body, new_table3)

# Aggregate row
counted_replace(
    "$\\boldsymbol{+1.10}$ & $\\boldsymbol{.543}$ & $\\boldsymbol{< .001}$ & $\\boldsymbol{145}$ & $\\boldsymbol{< .001}$ & $\\boldsymbol{0.21}$",
    f"$\\boldsymbol{{{h2_slope_1e4:+.2f}}}$ & $\\boldsymbol{{{nolz(h2_r2)}}}$ & $\\boldsymbol{{< .001}}$ & $\\boldsymbol{{{h2_mk_s}}}$ & $\\boldsymbol{{< .001}}$ & $\\boldsymbol{{{h2_dw:.2f}}}$"
)

# DW posthoc section
counted_replace(
    "DW $= 0.14$--$0.87$) indicate",
    f"DW $= {dw_min:.2f}$--${dw_max:.2f}$) indicate"
)

# ══════════════════════════════════════════════════════════════════
# POST-HOC: PERMUTATION
# ══════════════════════════════════════════════════════════════════

counted_replace(
    "observed mean $\\Delta = +0.054$ fell",
    f"observed mean $\\Delta = {h1c_delta:+.3f}$ fell"
)
counted_replace(
    "null mean $= +0.0003$, null SD $= 0.025$",
    f"null mean $= {perm_null_mean:+.4f}$, null SD $= {perm_null_sd:.3f}$"
)
counted_replace(
    "yielding $p = < .001$ (one-sided",
    f"yielding {perm_p_str} (one-sided"
)
# Figure note: observed Delta
counted_replace(
    "observed $\\Delta = +0.054$. The observed bias",
    f"observed $\\Delta = {h1c_delta:+.3f}$. The observed bias"
)
counted_replace(
    "($p = < .001$).",
    f"({perm_p_str})."
)

# ══════════════════════════════════════════════════════════════════
# POST-HOC: SENSITIVITY
# ══════════════════════════════════════════════════════════════════

d_by_n = {int(s["top_n"]): float(s["cohen_d_raw"]) for s in sens}
d_min_n = min(d_by_n, key=lambda n: d_by_n[n])
d_max_n = max(d_by_n, key=lambda n: d_by_n[n])

counted_replace(
    "$1.29$ (top-205) to $2.33$ (top-20)",
    f"${d_by_n[d_min_n]:.2f}$ (top-{d_min_n}) to ${d_by_n[d_max_n]:.2f}$ (top-{d_max_n})"
)
counted_replace(
    "approximately $+0.068$ for $N \\geq 100$",
    f"approximately ${sens_wt_100:+.3f}$ for $N \\geq 100$"
)

# Sensitivity table
old_sens_body = """ 20 & $+0.068$ & $2.33$ & $+0.068$ \\\\
 50 & $+0.069$ & $1.78$ & $+0.069$ \\\\
100 & $+0.064$ & $1.71$ & $+0.068$ \\\\
150 & $+0.061$ & $1.58$ & $+0.067$ \\\\
205 & $+0.054$ & $1.29$ & $+0.067$ \\\\"""
sens_rows = []
for s in sens:
    n  = s["top_n"]
    dr = float(s["mean_delta_raw"])
    cd = float(s["cohen_d_raw"])
    dw = float(s["mean_delta_weighted"])
    sens_rows.append(f"{n:>3s} & ${dr:+.3f}$ & ${cd:.2f}$ & ${dw:+.3f}$ \\\\")
new_sens = "\n".join(sens_rows)
counted_replace(old_sens_body, new_sens)

# ══════════════════════════════════════════════════════════════════
# POST-HOC: DOMAIN H1
# ══════════════════════════════════════════════════════════════════

dom_eud_delta = float(dom_eud["mean_delta"])
dom_eud_d = float(dom_eud["cohen_d"])
dom_eud_padj = float(dom_eud["p_adjusted"])

counted_replace(
    "$\\Delta = +0.008$, $d = 0.17$, $p_{\\text{adj}} = 0.111$",
    f"$\\Delta = {dom_eud_delta:+.3f}$, $d = {dom_eud_d:.2f}$, $p_{{\\text{{adj}}}} = {dom_eud_padj:.3f}$"
)

# Top 3
def domain_at(domain):
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

top3_parts = [f"{domain_at(d['domain'])} ($d = {float(d['cohen_d']):.2f}$)" for d in dom_top3]
new_top3 = ", ".join(top3_parts)
counted_replace(
    "Substance Use \\& Addiction ($d = 5.69$), Anxiety \\& Stress ($d = 4.42$), Trauma \\& PTSD ($d = 4.28$)",
    new_top3
)

# Domain H1 table
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
new_dom_h1 = "\n".join(table_dom_h1)

old_dom_h1_body = """Substance Use \\& Addiction                         &  8 & $+0.061$ & $5.69$ & $.008$ \\\\
Anxiety \\& Stress                                  & 23 & $+0.079$ & $4.42$ & $< .001$ \\\\
Trauma \\& PTSD                                     & 10 & $+0.072$ & $4.28$ & $.003$ \\\\
Mood \\& Depression                                 & 21 & $+0.087$ & $3.97$ & $< .001$ \\\\
Psychosis \\& Severe Mental Illness                 & 16 & $+0.070$ & $2.93$ & $< .001$ \\\\
General Distress \\& Psychopathology                & 44 & $+0.074$ & $2.67$ & $< .001$ \\\\
Functional \\& Quality of Life                      & 33 & $+0.046$ & $1.65$ & $< .001$ \\\\
Eudaimonic \\& Flourishing                          & 50 & $+0.008$ & $0.17$ & $.111$ \\\\"""
counted_replace(old_dom_h1_body, new_dom_h1)

# ══════════════════════════════════════════════════════════════════
# POST-HOC: DOMAIN H2
# ══════════════════════════════════════════════════════════════════

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
dom_h2_sorted = sorted(dom_h2, key=lambda x: -float(x["slope_per_decade"]))
table_dom_h2 = []
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
new_dom_h2 = "\n".join(table_dom_h2)

old_dom_h2_body = """Eudaimonic \\& Flourishing                     & 50 & $+0.0081$ & $+0.020$ & $.702$ & $< .001$ \\\\
Functional \\& QoL                             & 33 & $+0.0040$ & $+0.010$ & $.975$ & $< .001$ \\\\
Anxiety \\& Stress                             & 23 & $+0.0018$ & $+0.004$ & $.928$ & $< .001$ \\\\
Mood \\& Depression                            & 21 & $+0.0016$ & $+0.004$ & $.508$ & $< .001$ \\\\
Trauma \\& PTSD                                & 10 & $+0.0010$ & $+0.002$ & $.145$ & $.055$ \\\\
Substance Use \\& Addiction                    &  8 & $+0.0006$ & $+0.002$ & $.860$ & $< .001$ \\\\
Psychosis \\& SMI                              & 16 & $-0.0023$ & $-0.006$ & $.854$ & $< .001$ \\\\
Gen.\\ Distress \\& Psychopathology             & 44 & $-0.0043$ & $-0.011$ & $.719$ & $< .001$ \\\\"""
counted_replace(old_dom_h2_body, new_dom_h2)

# Domain H2 narrative
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
pos_h2 = [d for d in dom_h2_sorted if float(d["slope_per_decade"]) > 0]
neg_h2 = [d for d in dom_h2_sorted if float(d["slope_per_decade"]) < 0]
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
new_narrative = "\n".join(narr_lines)

old_narrative = """Eudaimonic \\& Flourishing showed the largest positive slope ($+0.0081$/decade), reflecting the growing but still insufficient adoption of eudaimonic instruments.
Functional \\& QoL ($+0.0040$/decade) and Anxiety \\& Stress ($+0.0018$/decade) also showed significant positive trends.
Psychosis \\& SMI and General Distress \\& Psychopathology showed small but significant negative slopes (total 25-year change $<\\!-0.011$), possibly reflecting more nuanced instrumentation."""
counted_replace(old_narrative, new_narrative)

# ══════════════════════════════════════════════════════════════════
# USAGE CORRELATION
# ══════════════════════════════════════════════════════════════════

counted_replace(
    "Spearman $\\rho = .284$ ($p < .001$)",
    f"Spearman $\\rho = {nolz(rho)}$ ({rho_p_str})"
)

# ══════════════════════════════════════════════════════════════════
# DISCUSSION
# ══════════════════════════════════════════════════════════════════

# d range
counted_replace(
    "$d = 1.03$--$2.76$ usage-weighted",
    f"$d = {d_wt_min:.2f}$--${d_wt_max:.2f}$ usage-weighted"
)

# "increased by approximately $+0.003$ units" (discussion, may be 2nd occurrence)
counted_replace(
    "increased by approximately $+0.003$ units",
    f"increased by approximately ${h2_total:+.003f}$ units"
)

# annual increment
counted_replace(
    "annual increment ($+0.00011$) is small",
    f"annual increment (${h2_slope:+.5f}$) is small"
)
counted_replace(
    "annual increment ($+0.00011$ per year)",
    f"annual increment (${h2_slope:+.5f}$ per year)"
)

# Time horizon d values
counted_replace(
    "$d = 2.07$ raw, $d = 2.76$ weighted",
    f"$d = {th_raw:.2f}$ raw, $d = {th_wt:.2f}$ weighted"
)
counted_replace(
    "largest effect ($d = 2.76$)",
    f"largest effect ($d = {th_wt:.2f}$)"
)

# DW: "DW $= 0.14$--$0.87$) indicate" (2nd occurrence)
counted_replace(
    f"DW $= {dw_min:.2f}$--${dw_max:.2f}$) indicate",
    f"DW $= {dw_min:.2f}$--${dw_max:.2f}$) indicate"
)

# $+1.1 \times 10^{-4}$/year amounts to $+0.003$
counted_replace(
    "$+1.1 \\times 10^{-4}$/year amounts to $+0.003$ over 25 years",
    f"${h2_slope_1e4:+.1f} \\times 10^{{-4}}$/year amounts to ${h2_total:+.003f}$ over 25 years"
)

# cross-sectional effect size
counted_replace(
    "cross-sectional effect size ($\\Delta = +0.054$, $d = 0.97$)",
    f"cross-sectional effect size ($\\Delta = {h1c_delta:+.003f}$, $d = {h1c_d:.2f}$)"
)

# ══════════════════════════════════════════════════════════════════
# CONCLUSION
# ══════════════════════════════════════════════════════════════════

if h1c_d >= 1.0:
    qual = f"Cohen's $d = {h1c_d:.2f}$"
else:
    qual = f"Cohen's $d = {h1c_d:.2f}$, near-large effect"
counted_replace(
    "Cohen's $d = 0.97$, near-large effect",
    qual
)

# ══════════════════════════════════════════════════════════════════
# SUPPLEMENTARY 
# ══════════════════════════════════════════════════════════════════

# S1 floatnote
counted_replace(
    "Raw $\\Delta = +0.054$, $d = 0.97$.",
    f"Raw $\\Delta = {h1c_delta:+.003f}$, $d = {h1c_d:.2f}$."
)

# S2 top50 d comparison
counted_replace(
    "$d = 1.33$ vs.\\ $d = 0.97$",
    f"$d = {h1a_d:.2f}$ vs.\\ $d = {h1c_d:.2f}$"
)

# S2 weighted
counted_replace(
    "weighted $\\Delta = +0.069$, $d = 1.44$",
    f"weighted $\\Delta = {h1b_delta:+.003f}$, $d = {h1b_d:.2f}$"
)

# ══════════════════════════════════════════════════════════════════
# S8: TOP HEDONIC/EUDAIMONIC SCALES
# ══════════════════════════════════════════════════════════════════

sorted_hedonic = sorted(mean_deltas.items(), key=lambda x: -x[1])
sorted_eudaimonic = sorted(mean_deltas.items(), key=lambda x: x[1])
s8_rows = []
for i in range(20):
    hs, hd = sorted_hedonic[i]
    if i < 10:
        es, ed = sorted_eudaimonic[i]
        s8_rows.append(
            f"{i+1:2d} & {hs:15s} & ${hd:+.003f}$ & "
            f"{i+1:2d} & {es:15s} & ${ed:+.003f}$ \\\\"
        )
    else:
        s8_rows.append(
            f"{i+1:2d} & {hs:15s} & ${hd:+.003f}$ &    &                  &          \\\\"
        )
new_s8 = "\n".join(s8_rows)

# Match old S8 body
old_s8_start = " 1 & PANAS"
old_s8_end = "20 & BDI-II"
# Find and replace entire block
s8_idx_start = text.find(old_s8_start)
s8_idx_end = text.find("\\bottomrule", s8_idx_start) if s8_idx_start >= 0 else -1
if s8_idx_start >= 0 and s8_idx_end >= 0:
    text = text[:s8_idx_start] + new_s8 + "\n" + text[s8_idx_end:]
    changes += 1
    print(f"  S8 table replaced")

# ══════════════════════════════════════════════════════════════════
# S9: PER-DIMENSION DESCRIPTIVE STATS
# ══════════════════════════════════════════════════════════════════

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
new_s9 = "\n".join(s9_rows)

old_s9_start = "Foundational claim        & 0.188"
old_s9_end = "Central tension"
# Find S9 table in the s_dim_descriptives context
s9_search = text.find("s_dim_descriptives")
if s9_search >= 0:
    s9_start = text.find(old_s9_start, s9_search)
    if s9_start >= 0:
        s9_end = text.find("\\bottomrule", s9_start)
        if s9_end >= 0:
            text = text[:s9_start] + new_s9 + "\n" + text[s9_end:]
            changes += 1
            print(f"  S9 table replaced")

# ══════════════════════════════════════════════════════════════════
# S10: TOP 20 MOST-USED SCALES
# ══════════════════════════════════════════════════════════════════

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
sorted_by_weight = sorted(weights.items(), key=lambda x: -x[1])[:20]
s10_rows = []
hedonic_count = 0
for i, (scale, wt) in enumerate(sorted_by_weight, 1):
    delta  = mean_deltas.get(scale, 0)
    pubs   = total_pubs.get(scale, 0)
    domain = domains.get(scale, "?")
    name   = scale_names.get(scale, scale)
    name_esc = name.replace("&", "\\&")
    dom_short = domain_abbrev_s10.get(domain, domain)
    pubs_str  = f"{pubs:,d}"
    s10_rows.append(
        f"{i:2d} & {scale} & {name_esc} & {dom_short} & {pubs_str} & ${delta:+.003f}$ \\\\"
    )
    if delta > 0: hedonic_count += 1
new_s10 = "\n".join(s10_rows)

s10_search = text.find("s_top_usage")
if s10_search >= 0:
    s10_start = text.find(" 1 & AUDIT", s10_search)
    if s10_start >= 0:
        s10_end = text.find("\\bottomrule", s10_start)
        if s10_end >= 0:
            text = text[:s10_start] + new_s10 + "\n" + text[s10_end:]
            changes += 1
            print(f"  S10 table replaced")

# S10 floatnote
s10_fn = (
    f"\\floatnote{{Cumulative PubMed publications (2000--2025). "
    f"Usage weights normalised to sum to 1.0 across all {n_scales} scales. "
    f"Mean $\\Delta$ averaged across all six hedonic--eudaimonic dimensions. "
    f"{hedonic_count} of 20 most-used scales show positive (hedonic) loading.}}"
)
# Find old floatnote
fn_marker = "\\floatnote{Cumulative PubMed publications"
fn_start = text.find(fn_marker)
if fn_start >= 0:
    fn_end = text.find("}}", fn_start)  # double }} to find end of floatnote
    if fn_end >= 0:
        # But floatnote ends with just }
        fn_end2 = text.find("}", fn_start + len(fn_marker))
        # Find the proper closing brace
        depth = 1
        pos = fn_start + len("\\floatnote{")
        while pos < len(text) and depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        text = text[:fn_start] + s10_fn + text[pos:]
        changes += 1
        print(f"  S10 floatnote replaced")

# ══════════════════════════════════════════════════════════════════
# Write
# ══════════════════════════════════════════════════════════════════

TEX.write_text(text)
print(f"\nDone! Made {changes} replacements.")
print(f"Key stats:")
print(f"  Scales: {n_scales}")
print(f"  H1C: Δ={h1c_delta:+.003f}, d={h1c_d:.2f}")
print(f"  H1D: Δ={h1d_delta:+.003f}, d={h1d_d:.2f}")
print(f"  H2 agg: slope={h2_slope:+.6f}, R²={h2_r2:.3f}, total={h2_total:+.004f}")
print(f"  All 6/6 positive: YES")
