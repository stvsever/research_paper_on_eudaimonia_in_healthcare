# Semantic Embedding Analysis: Hedonic vs. Eudaimonic Orientation of Major Clinical Scales in Mental Healthcare

**Research Question:** From a semantic embedding perspective, are the most commonly used clinical scales in mental healthcare — those used to define therapy effectiveness and patient progress — semantically closer to hedonic than to eudaimonic conceptions of well-being?

**Method:** OpenAI `text-embedding-3-large` | Cosine similarity | Wilcoxon signed-rank test (one-sided)

**Date:** March 2026

---

## 1. Method

### 1.1 Conceptual Framework

Six dimensions distinguishing hedonic from eudaimonic well-being were extracted from the Eudaimonia Stakeholder Brief and embedded as parallel paired texts:

| # | Dimension | Hedonic pole | Eudaimonic pole |
|---|-----------|-------------|-----------------|
| 1 | **Foundational claim** | Maximising pleasure, minimising pain | Excellent rational activity, virtue, self-realisation |
| 2 | **Evaluative criterion** | Subjective affect, satisfaction, distress absence | Agency quality, character, genuine engagement |
| 3 | **Time horizon** | Momentary, short-horizon assessment | Biographical, sustained across a complete life |
| 4 | **Adversity** | Intrinsically negative; to be avoided | Potentially constitutive of growth and flourishing |
| 5 | **Measurement proxies** | Affect balance, symptom checklists, satisfaction | Purpose, growth, autonomy, mastery, meaning |
| 6 | **Central tension** | Surface pleasure may lack depth (experience machine) | Flourishing may require discomfort and sacrifice |

This yields **12 reference embeddings** (6 dimensions × 2 paradigms).

### 1.2 Clinical Scales

30 of the most commonly used scales in mental healthcare research and clinical practice were selected (see `data/scales/clinical_scales.txt` for full definitions). Each scale's technical description was embedded using the same model.

### 1.3 Analysis

- Cosine similarity was computed between each of the 30 scale embeddings and all 12 dimension reference embeddings (= 360 comparisons).
- For each dimension, a paired Wilcoxon signed-rank test (one-sided: H₁ = hedonic > eudaimonic) tested whether scales are systematically closer to the hedonic pole.
- Cohen's d was computed as a standardised effect size for each dimension.
- An overall aggregate test pooled all 180 paired observations.

---

## 2. Results

### 2.1 Overall Finding

| Metric | Value |
|--------|-------|
| Mean cosine similarity to hedonic dimensions | **0.2920** |
| Mean cosine similarity to eudaimonic dimensions | **0.2157** |
| Mean Δ (Hedonic − Eudaimonic) | **+0.0763** |
| Wilcoxon signed-rank p (one-sided) | **< 1e-10 (\*\*\*)** |
| Cohen's d | **1.22 (large)** |

**The 30 most commonly used clinical scales in mental healthcare are, on aggregate, significantly and substantially closer in semantic space to hedonic well-being descriptions than to eudaimonic well-being descriptions.**

### 2.2 Per-Dimension Breakdown

| Dimension | Mean sim. Hedonic | Mean sim. Eudaimonic | Δ | Wilcoxon p | Cohen's d | Sig. |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Foundational claim | 0.176 | 0.125 | +0.051 | < 0.001 | 1.96 | \*\*\* |
| Evaluative criterion | 0.337 | 0.255 | +0.082 | < 0.001 | 2.31 | \*\*\* |
| Time horizon | 0.346 | 0.216 | +0.130 | < 0.001 | 3.83 | \*\*\* |
| Adversity | 0.237 | 0.136 | +0.101 | < 0.001 | 2.78 | \*\*\* |
| Measurement proxies | 0.493 | 0.377 | +0.116 | < 0.001 | 2.13 | \*\*\* |
| Central tension | 0.162 | 0.184 | −0.022 | 0.999 | −0.77 | n.s. |

Five of six dimensions show **highly significant** hedonic dominance (all p < 0.001 with large effect sizes, d > 1.96). The single exception — **Central tension** — shows a slight eudaimonic lean, which is coherent: the hedonic "central tension" text describes the *inadequacy* of hedonism (Nozick's experience machine), making it semantically dissimilar to symptom-focused scales.

### 2.3 Strongest Hedonic-Leaning Scales (Top 10 by mean Δ)

| Scale | Mean Δ | Interpretation |
|-------|:------:|---------------|
| BAI | +0.113 | Pure somatic/cognitive anxiety symptoms |
| BSI | +0.110 | Somatisation, depression, anxiety distress |
| HAM-A | +0.106 | Clinician-rated anxiety symptom severity |
| MADRS | +0.106 | Clinician-rated depression symptoms |
| DASS-21 | +0.099 | Depression, anxiety, stress symptom triad |
| PHQ-9 | +0.098 | Depressive symptom frequency |
| HAM-D | +0.097 | Clinician-rated depression severity |
| BDI-II | +0.097 | Depressive symptom severity self-report |
| SCL-90-R | +0.097 | Broad psychological symptom distress |
| EPDS | +0.095 | Postnatal depression screening |

### 2.4 Most Balanced or Eudaimonic-Leaning Scales

| Scale | Mean Δ | Interpretation |
|-------|:------:|---------------|
| WHODAS-2.0 | −0.023 | **Only scale leaning eudaimonic** — assesses functioning and capabilities |
| GAF | +0.020 | Global functioning assessment (closest to balanced) |
| SF-36 | +0.033 | Health-related quality of life across functional domains |
| SDS | +0.043 | Functional impairment across life domains |
| EQ-5D | +0.054 | Quality of life including activity and self-care |

---

## 3. Interpretation

### 3.1 The Hedonic Dominance Hypothesis is Supported

The research hypothesis — that commonly used clinical scales are semantically oriented toward hedonic rather than eudaimonic conceptions of well-being — is **strongly supported**. With an overall Cohen's d of 1.22, this is not a marginal finding but a large and consistent effect across all but one theoretical dimension.

### 3.2 What This Means

The clinical mental healthcare toolkit overwhelmingly measures **how patients feel** (symptom presence, distress levels, affect balance, satisfaction) rather than **how patients function, grow, and live** (agency, purpose, character development, meaning-making, capacity for self-directed action).

This is not a design flaw of any individual instrument — each scale is well-validated for its purpose — but it reveals a **systematic orientation** in what the field considers relevant to measure. When therapy effectiveness is evaluated using these instruments, the evaluative criterion is implicitly hedonic: reduction of negative affect and symptoms.

### 3.3 Nuances

1. **The WHODAS-2.0** is the only scale that leans eudaimonic. It assesses what people can *do* (cognition, mobility, self-care, participation) — directly echoing the capabilities approach descended from Aristotle. This confirms the framework's discriminant validity.

2. **Functional scales** (GAF, SF-36, SDS, EQ-5D) cluster closest to the balanced midpoint, suggesting that instruments assessing what people *can do* rather than how they *feel* naturally bridge the hedonic–eudaimonic divide.

3. **Symptom-focused instruments** (BAI, HAM-A, MADRS, PHQ-9, BDI-II) show the strongest hedonic loading. Their descriptions are almost exclusively about subjective distress, negative affect, and somatic symptoms — the direct currency of hedonic measurement.

4. **The Central Tension dimension reversal** is interpretively important: the hedonic "central tension" text discusses the *failure* of pure hedonism (Nozick's experience machine), which is semantically *distant* from symptom-reduction scales. This is a validity check, not a contradiction.

### 3.4 Limitations

- **Method:** Cosine similarity in embedding space is a proxy for semantic relatedness, not conceptual identity. High similarity means the texts occupy nearby regions of meaning, which is meaningful but not equivalent to a content analysis.
- **Scale descriptions vs. scale items:** We embedded scale descriptions (what the scale measures), not individual items. Item-level analysis might reveal more nuance.
- **Embedding model:** Results depend on the training corpus and architecture of `text-embedding-3-large`. Different models may produce somewhat different absolute values, though the *relative* pattern is likely robust.
- **Scale selection:** The 30 scales were selected for frequency of use in clinical research and practice. Alternative selections might shift magnitudes slightly but are unlikely to reverse the overall finding.

---

## 4. Conclusion

The most commonly used clinical scales in mental healthcare are **semantically and significantly oriented toward hedonic well-being** — measuring symptom reduction, affect balance, and subjective distress — rather than eudaimonic well-being (purpose, growth, agency, meaning, character). This hedonic loading is not incidental: it reflects the historical development of clinical psychology around symptom taxonomies and the medical model's emphasis on alleviating suffering.

The finding does not imply that symptom measurement is wrong — relief from suffering is a genuine good. Rather, it indicates that **current measurement practices are insufficient for capturing the full picture of therapeutic effectiveness**. A treatment that successfully reduces PHQ-9 scores while leaving a patient without purpose, agency, or authentic engagement has achieved hedonic relief but not human flourishing.

To evaluate whether mental healthcare truly supports the full spectrum of well-being, the field would need to systematically supplement its hedonic measurement toolkit with validated eudaimonic instruments — a shift from asking only "Do you feel better?" to also asking "Are you living better?"

---

## 5. File Structure

```
research_paper/
├── run_pipeline.py              # Main pipeline runner
├── data/
│   ├── dimensions/
│   │   ├── hedonic.txt          # 6 hedonic dimension texts
│   │   └── eudaimonic.txt       # 6 eudaimonic dimension texts
│   └── scales/
│       └── clinical_scales.txt  # 30 clinical scale definitions
├── src/
│   ├── __init__.py
│   ├── embed.py                 # Embedding generation (OpenAI API)
│   ├── analyze.py               # Cosine similarity + statistical tests
│   └── visualize.py             # Research-grade violin + strip figure
├── outputs/
│   ├── embeddings.json          # Raw embeddings (12 dims + 30 scales)
│   ├── cosine_similarities.csv  # 180 pairwise comparisons
│   ├── dimension_statistics.csv # Per-dimension Wilcoxon + effect sizes
│   └── overall_statistics.json  # Aggregate test results
├── figures/
│   ├── hedonic_eudaimonic_semantic_analysis.png  # 300 DPI
│   └── hedonic_eudaimonic_semantic_analysis.pdf  # Vector
└── REPORT.md                    # This document
```
