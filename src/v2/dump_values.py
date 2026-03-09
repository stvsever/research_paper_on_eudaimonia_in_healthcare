"""Dump current R values so we can reconstruct the template."""
import csv, json
from pathlib import Path

V2 = Path(__file__).parent
OUT = V2 / "outputs"

def jload(name):
    with open(OUT / name) as f: return json.load(f)
def cload(name):
    with open(OUT / name) as f: return list(csv.DictReader(f))
def nolz(x, fmt=".3f"):
    return f"{float(x):{fmt}}"[1:]

h1 = jload("h1_results.json")
h2_dims = cload("h2_dimension_trends.csv")
h2_sum = jload("h2_summary.json")
perm = jload("posthoc_permutation.json")
sens = cload("posthoc_sensitivity_n.csv")

scales = [l.strip() for l in open(V2 / "data/scales/clinical_scales_200.txt") if " | " in l]
n = len(scales)

h1c = h1["H1C_ALL_RAW"]
h1d = h1["H1D_ALL_WEIGHTED"]
h1a = h1["H1A_TOPN_RAW"]
h1b = h1["H1B_TOPN_WEIGHTED"]

print(f"n_scales = {n}")
print(f"H1C delta = {float(h1c['overall_mean_delta']):+.3f}")
print(f"H1C d = {float(h1c['overall_cohen_d']):.2f}")
print(f"H1D delta = {float(h1d['overall_mean_delta']):+.3f}")
print(f"H1D d = {float(h1d['overall_cohen_d']):.2f}")
print(f"H1A delta = {float(h1a['overall_mean_delta']):+.3f}")
print(f"H1A d = {float(h1a['overall_cohen_d']):.2f}")
print(f"H1B delta = {float(h1b['overall_mean_delta']):+.3f}")
print(f"H1B d = {float(h1b['overall_cohen_d']):.2f}")
print(f"H2 agg slope = {float(h2_sum['slope']):+.5f}")
print(f"H2 agg slope sci = {float(h2_sum['slope']):+.2e}")
print(f"H2 total = {float(h2_sum['total_change_baselined']):+.3f}")
print(f"H2 R2 = {nolz(float(h2_sum['r_squared']))}")
print(f"H2 p = {float(h2_sum['p_value']):.10f}")
print(f"H2 MK S = {int(float(h2_sum['mk_s']))}")
print(f"H2 MK p = {float(h2_sum['mk_p']):.10f}")
print(f"H2 DW = {float(h2_sum['durbin_watson']):.2f}")
print(f"Perm p = {float(perm['perm_p'])}")
print(f"Perm null_mean = {float(perm['null_mean']):+.4f}")
print(f"Perm null_sd = {float(perm['null_sd']):.3f}")

# Sensitivity range
d_vals = [float(s["cohen_d_raw"]) for s in sens]
print(f"Sens d range = {min(d_vals):.2f}--{max(d_vals):.2f}")
for s in sens:
    print(f"  Top-{s['top_n']}: d={float(s['cohen_d_raw']):.2f}")
