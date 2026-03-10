"""Explore clinical domain similarity distribution to design the filter threshold."""
import json
import numpy as np
from scipy.spatial.distance import cosine

with open("outputs/embeddings.json") as f:
    data = json.load(f)
with open("outputs/ngram_embeddings.json") as f:
    ngram = json.load(f)

# Clinical domain centroid = mean of all 189 clinical scale embeddings
scale_embs = list(data["scale_embeddings"].values())
centroid = np.mean(scale_embs, axis=0)
centroid = centroid / np.linalg.norm(centroid)

# Scale similarities to centroid
scale_sims = [1 - cosine(e, centroid) for e in scale_embs]
print("=== SCALE similarities to clinical centroid ===")
print(f"  N={len(scale_sims)}, min={min(scale_sims):.4f}, max={max(scale_sims):.4f}")
print(f"  mean={np.mean(scale_sims):.4f}, median={np.median(scale_sims):.4f}")
print(f"  P25={np.percentile(scale_sims,25):.4f}, P10={np.percentile(scale_sims,10):.4f}")

# Word similarities to centroid
word_embs = ngram["word_embeddings"]
word_sims = {w: 1 - cosine(word_embs[w], centroid) for w in word_embs}
sims = list(word_sims.values())
print()
print("=== WORD similarities to clinical centroid ===")
print(f"  N={len(sims)}, min={min(sims):.4f}, max={max(sims):.4f}")
print(f"  mean={np.mean(sims):.4f}, median={np.median(sims):.4f}")
print(f"  P75={np.percentile(sims,75):.4f}, P90={np.percentile(sims,90):.4f}")

# Show example words at different similarity levels
sorted_words = sorted(word_sims.items(), key=lambda x: x[1], reverse=True)
print()
print("Top 20 most clinically relevant words:")
for w, s in sorted_words[:20]:
    print(f"  {s:.4f}  {w}")
print()
print("Words around median:")
mid = len(sorted_words) // 2
for w, s in sorted_words[mid-5:mid+5]:
    print(f"  {s:.4f}  {w}")
print()
print("Bottom 10 least clinically relevant words:")
for w, s in sorted_words[-10:]:
    print(f"  {s:.4f}  {w}")

# Count words at different thresholds
print()
print("=== Threshold analysis ===")
for pct in [50, 60, 70, 75, 80, 90]:
    thresh = np.percentile(sims, pct)
    n_above = sum(1 for s in sims if s >= thresh)
    print(f"  P{pct} threshold={thresh:.4f}: {n_above} words kept")
