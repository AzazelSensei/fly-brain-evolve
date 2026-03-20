# Research Journal: Phase 3 Staged Evolution — Filters Then Connectivity

**Date:** 2026-03-20
**Phase:** Phase 3 — Hierarchical Sparse Coding (Staged Approach)
**Status:** COMPLETE — Marginal improvement over joint, HOG still wins

---

## Hypothesis

Simultaneously evolving filters + connectivity is too hard (co-evolution problem).
Staged approach: evolve filters first (connectivity fixed), then freeze filters
and evolve connectivity. Mirrors biology: optic lobe develops over weeks,
mushroom body learns in seconds.

## Results

### Stage 1: Filter Evolution (connectivity fixed)
```
S1 Gen  0: best=0.5113 mean=0.1774
S1 Gen 10: best=0.5707 mean=0.1859
S1 Gen 20: best=0.5184 mean=0.1759
S1 Gen 35: best=0.5914 mean=0.1849
S1 Gen 39: best=0.5300 mean=0.1811
Best overall: 0.5921 (2590s)
```

### Stage 2: Connectivity Evolution (filters frozen)
```
S2 Gen  0: best=0.5894 mean=0.1760
S2 Gen 10: best=0.6005 mean=0.2046
S2 Gen 20: best=0.5396 mean=0.1904
S2 Gen 39: best=0.5709 mean=0.1839
Best overall: 0.6504 (2520s)
```

### Comparison

| Approach | Best | Mean | Time |
|----------|------|------|------|
| Phase 3 joint | 0.621 | 0.19 | 1878s |
| Phase 3 staged | **0.650** | 0.19 | 5110s |
| Phase 2 HOG+STDP v3 | **0.732** | 0.66 | 2336s |

## Analysis

1. **Staging helps slightly** (0.621 → 0.650) but does not close the gap to HOG (0.732)
2. **Mean fitness stuck at ~0.18** in both stages — population convergence failure
3. **Root cause: noisy fitness evaluation.** On-the-fly Poisson spikes differ each run,
   making fitness estimates unreliable. Evolution can't distinguish better genomes from noise.
4. **HOG is specifically designed for digit recognition** — gradient histograms capture
   exactly the stroke patterns that distinguish handwritten digits. Evolved 5x5 filters
   capture edges but not the histogram aggregation that makes HOG powerful.

## Key Finding for Thesis

At this scale (12 filters, 80 total generations, 300 training images), hand-crafted
features dominate evolved features. This mirrors the ML history: SIFT/HOG dominated
until deep learning could afford orders of magnitude more computation.

Overcoming this requires either:
- 200+ generations with larger population (GPU required)
- Different filter architecture (learned pooling, multi-scale)
- Hybrid: HOG-inspired initialization with fine-tuning evolution

## Decision

Phase 3 documented as complete with negative-but-informative result.
Proceeding to Phase 4 (Attention) using HOG+STDP v3 (0.732) as base.
