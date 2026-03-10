"""Show words around the P75 threshold to validate cutoff."""
import json
import numpy as np
from scipy.spatial.distance import cosine

with open("outputs/embeddings.json") as f:
    data = json.load(f)
with open("outputs/ngram_embeddings.json") as f:
    ngram = json.load(f)

scale_embs = list(data["scale_embeddings"].values())
centroid = np.mean(scale_embs, axis=0)
centroid = centroid / np.linalg.norm(centroid)

word_embs = ngram["word_embeddings"]
word_sims = {w: 1 - cosine(word_embs[w], centroid) for w in word_embs}
sorted_words = sorted(word_sims.items(), key=lambda x: x[1], reverse=True)

# Find P75 index
n = len(sorted_words)
p75_idx = n // 4  # top 25% = first quarter
thresh = sorted_words[p75_idx][1]
print(f"P75 threshold sim = {thresh:.4f}, keeping top {p75_idx} words")
print()
print("Words just ABOVE P75 (kept):")
for w, s in sorted_words[p75_idx-15:p75_idx]:
    print(f"  {s:.4f}  {w}")
print()
print("Words just BELOW P75 (removed):")
for w, s in sorted_words[p75_idx:p75_idx+15]:
    print(f"  {s:.4f}  {w}")

# Also check specific clinical words
clinical_check = ["anxious", "depressed", "therapeutic", "resilient", "traumatic",
                  "mindful", "coping", "emotional", "cognitive", "behavioral",
                  "diagnostic", "psychiatric", "empathetic", "stressful", "distressed",
                  "happy", "sad", "angry", "fearful", "hopeful", "lonely", "joyful",
                  "fulfilled", "meaningful", "pleasant", "suffering", "content",
                  "satisfied", "grateful", "optimistic", "pessimistic", "calm",
                  "relaxed", "tense", "worried", "nervous", "confident", "insecure"]
print()
print("Clinical word check (KEPT if >= {:.4f}):".format(thresh))
for w in clinical_check:
    if w in word_sims:
        status = "KEPT" if word_sims[w] >= thresh else "REMOVED"
        print(f"  {word_sims[w]:.4f}  {w}  [{status}]")
    else:
        print(f"  -----  {w}  [NOT IN VOCAB]")
