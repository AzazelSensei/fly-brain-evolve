# Research Journal: Phase 3 — Hierarchical Sparse Coding

**Date:** 2026-03-20
**Phase:** Phase 3 — Evolved Visual Cortex + Mushroom Body
**Status:** COMPLETE — Hand-crafted HOG still beats evolved filters

---

## Architecture

```
16x16 image → 12 evolved 5x5 filters (stride=2) → 2x2 max-pool → 108 features
    → 108 PN → 500 KC → 10 MBON + 1 APL (619 neurons)
    → Dopamin-STDP ile KC→MBON öğrenme
    → Evrim: filtreler + PN→KC bağlantıları
```

Key innovations:
- On-the-fly Poisson spike generation (eliminates GB memory for spike arrays)
- Per-genome visual features (each genome applies its own filters)
- Gabor-initialized filters (evolution starts from reasonable features)
- Max-pooling reduces 340 → 108 features

## Results

### Gabor+STDP Baseline (fixed Gabor filters, no evolution)
```
3 random brains: best=0.5434 mean=0.2809
```

### Evolved Filters + STDP (40 generations, pop=20)
```
Gen  0: best=0.5887 mean=0.1894
Gen 10: best=0.5409 mean=0.1821
Gen 20: best=0.5298 mean=0.1829
Gen 30: best=0.4795 mean=0.1864
Gen 39: best=0.5097 mean=0.1931
Best overall: 0.6208
```

### Comparison

| Approach | Features | Best Fitness |
|----------|----------|-------------|
| HOG+GA (Phase 1) | 128 hand-crafted | 0.237 |
| HOG+STDP v3 (Phase 2) | 128 hand-crafted | **0.732** |
| Gabor+STDP (Phase 3 baseline) | 108 fixed Gabor | 0.543 |
| **Evolved+STDP (Phase 3)** | **108 evolved** | **0.621** |

## Analysis: Why Phase 3 < Phase 2

1. **HOG features are specifically designed for digit recognition.**
   HOG captures oriented gradient histograms — perfect for distinguishing
   handwritten strokes. Gabor filters capture edges but not histograms.

2. **Filter evolution is too slow.** 12 filters × 25 params = 300 extra parameters.
   With only 40 generations and pop=20, the search space is too large.
   Mean fitness stayed at ~0.19 — population failed to converge.

3. **Co-evolution problem.** Evolving filters + connectivity simultaneously
   means a good filter set might be paired with bad connectivity and get
   eliminated. The search spaces interfere destructively.

4. **Gabor baseline is already reasonable.** Fixed Gabor (0.543) vs evolved (0.621)
   shows only marginal improvement from evolution. The filter search space is
   well-structured (nearby params = similar filters) but the GA can't exploit this.

## Key Finding: Feature Engineering Still Matters

At this scale (12 filters, 40 generations), hand-crafted features dominate.
This mirrors the broader ML history: hand-crafted features (SIFT, HOG, SURF)
dominated until deep learning could afford millions of training examples.

Our evolved filters need either:
- **More generations** (200+) with larger population (50+)
- **Staged evolution:** First evolve ONLY filters on a feature quality metric,
  then freeze and evolve connectivity with STDP
- **More training data:** The current 300 images × 3 epochs is too few for
  joint filter+connectivity optimization

## Biological Implication

This result is biologically consistent. In real insects:
- Optic lobe development takes days/weeks of activity-dependent refinement
- Mushroom body learning (dopamine) operates on seconds/minutes timescale
- The two systems develop on VERY different timescales

Our Phase 3 forces both to evolve on the same timescale (generations),
which is biologically unrealistic. A staged approach (long filter development
→ fast associative learning) would be more faithful.

## Technical Innovation: On-the-fly Spike Generation

The hierarchical_stdp.py kernel generates Poisson spikes inside the
simulation loop instead of pre-encoding:
```
if features[pn] > 0 and np.random.random() < features[pn] * max_rate * dt:
    g_exc[pn] += input_weight
```

This eliminates massive boolean arrays (pop × images × timesteps × features)
and naturally supports per-genome features. Memory: 19MB vs 4.8GB.

## Next Steps

1. **Staged evolution:** Evolve filters separately, then STDP on frozen features
2. **More generations:** 200 gen run to see if evolved filters eventually converge
3. **Proceed to Phase 4 with HOG features:** HOG+STDP at 0.732 is the best base
   for attention/gating experiments
