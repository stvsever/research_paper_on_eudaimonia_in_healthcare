import csv, math
from collections import defaultdict
import os

os.chdir(os.path.dirname(__file__) + '/outputs')

rows = list(csv.DictReader(open('h1_cosine_similarities_all.csv')))
by_dim = defaultdict(list)
by_dim_e = defaultdict(list)
for r in rows:
    by_dim[r['dimension']].append(float(r['sim_hedonic']))
    by_dim_e[r['dimension']].append(float(r['sim_eudaimonic']))

dim_order = ['Foundational claim','Evaluative criterion','Time horizon','Adversity','Measurement proxies','Central tension']
print("=== S9 DESCRIPTIVE STATS ===")
for d in dim_order:
    h = by_dim[d]
    e = by_dim_e[d]
    hmean = sum(h)/len(h)
    hsd = math.sqrt(sum((x-hmean)**2 for x in h)/len(h))
    emean = sum(e)/len(e)
    esd = math.sqrt(sum((x-emean)**2 for x in e)/len(e))
    line = "  {}: H={:.3f} sd={:.3f} [{:.2f},{:.2f}]  E={:.3f} sd={:.3f} [{:.2f},{:.2f}]".format(
        d, hmean, hsd, min(h), max(h), emean, esd, min(e), max(e))
    print(line)

print("\n=== S10 MEAN DELTA ===")
scatter = list(csv.DictReader(open('posthoc_usage_delta_scatter.csv')))
scatter_map = {r['scale']: float(r['mean_delta']) for r in scatter}
for s in ['AIMS','ACT','ASSIST','BRIEF','SPIN','BI','ORS','MRS','AIS','RAS']:
    val = scatter_map.get(s, None)
    if val is not None:
        print("  {}: {:+.3f}".format(s, val))
    else:
        print("  {}: NOT FOUND".format(s))
