"""
Clinical domain filter for the Google Books Ngram replication.

Restricts the ngram vocabulary to words semantically relevant to
Western clinical mental health assessment.  Relevance is defined as
cosine similarity to a **clinical domain centroid** — the normalised
mean of all 189 clinical-scale embeddings used in the primary analysis.

The top-quartile (P75) of words by clinical-domain similarity is
retained, yielding a vocabulary of clinically relevant adjectives and
verbs whose frequency trends better approximate those found in
mental-health literature.

Usage
-----
    from filter_clinical import clinical_vocabulary
    kept_words: set[str] = clinical_vocabulary(
        scale_embeddings,   # dict[str, list[float]]
        word_embeddings,    # dict[str, list[float]]
        percentile=75,
    )
"""

import numpy as np
from scipy.spatial.distance import cosine


def clinical_vocabulary(
    scale_embeddings: dict,
    word_embeddings: dict,
    percentile: int = 75,
) -> set:
    """Return the set of words above the *percentile* threshold of
    cosine-similarity to the clinical-scale centroid.

    Parameters
    ----------
    scale_embeddings : dict
        Mapping scale abbreviation → embedding vector (from
        ``embeddings.json["scale_embeddings"]``).
    word_embeddings : dict
        Mapping word → embedding vector (from
        ``ngram_embeddings.json["word_embeddings"]``).
    percentile : int
        Percentile threshold (0–100).  Words whose similarity to the
        clinical centroid is ≥ the *percentile*-th value are kept.
        Default 75 (upper quartile).

    Returns
    -------
    set[str]
        Words that pass the clinical-domain filter.
    """
    # Clinical domain centroid — normalised mean of all scale embeddings
    centroid = np.mean(list(scale_embeddings.values()), axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Cosine similarity of every word to the centroid
    word_sims = {
        w: 1.0 - cosine(word_embeddings[w], centroid)
        for w in word_embeddings
    }

    threshold = float(np.percentile(list(word_sims.values()), percentile))
    kept = {w for w, s in word_sims.items() if s >= threshold}

    print(f"  Clinical domain filter: threshold={threshold:.4f} "
          f"(P{percentile}), {len(kept)}/{len(word_sims)} words kept")

    return kept
