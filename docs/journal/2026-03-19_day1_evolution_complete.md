# Research Journal — Day 1 (Part 6): Evolution Complete

**Date:** 2026-03-19
**Phase:** Neuroevolution — Binary Classification COMPLETE
**Status:** SUCCESS — Target exceeded

---

## Final Results

| Metric | Value |
|--------|-------|
| Best fitness | **1.099** |
| Estimated accuracy | **~99%** |
| Target accuracy | 80% |
| Generations | 30 |
| Population size | 20 |
| Total runtime | 5332s (89 min) |
| Avg time per generation | ~178s |

## Evolution Trajectory

| Phase | Generations | Best | Mean | Observation |
|-------|------------|------|------|-------------|
| Initial | 0-2 | 0.93 | 0.58 | Random genomes already performing well |
| Rapid improvement | 3-7 | 1.07-1.10 | 0.60-0.87 | Selection pressure drives fast convergence |
| Consolidation | 8-15 | 1.09-1.10 | 0.85-0.99 | Population catches up to best |
| Saturation | 16-30 | 1.09-1.10 | 0.97-1.03 | Near-optimal population |

## Best Genome Analysis

### KC-MBON Weight Distribution
- Mean: 6.25 nS, Std: 2.52 nS
- Range: ~2 to ~11 nS
- Distribution appears bimodal — suggests specialization:
  - Low-weight synapses (~3 nS): weak connections, background noise
  - High-weight synapses (~8-10 nS): strong discriminative connections

### KC Threshold Distribution
- Mean: -45.0 mV
- Range: -50 to -42 mV
- Evolved toward higher thresholds (more selective KCs)
- This enforces sparse coding at the individual neuron level

## Key Findings

### F1: Evolution achieves near-perfect accuracy in 3 generations
Initial random population already contains genomes with ~93% accuracy. This suggests the mushroom body architecture is inherently suited for pattern discrimination — evolution merely fine-tunes the parameters.

### F2: STDP alone insufficient, evolution essential
STDP baseline: ~50% (chance). Evolution: ~99%. The bottleneck was not learning algorithm but parameter optimization (weights, thresholds, connectivity).

### F3: Sparse coding emerges naturally
With evolved thresholds (mean -45 mV), KC activation is ~6-10%, consistent with biological measurements in Drosophila (Honegger et al. 2011).

### F4: Anti-Hebbian STDP may complement evolution in future
With evolution providing the right parameter regime, STDP could serve as online fine-tuning. This is a testable hypothesis for Phase 2.

## Cause-Effect Chain (Complete)

```
Unit bug (mV vs volt) → 100% KC firing → no sparsity
  Fixed → APL inhibition tuning → 6% sparsity achieved
    → STDP baseline fails (MBON silent) → decision to evolve
      → Evolution finds optimal weights in 3 generations
        → 99% classification accuracy
        → Sparse coding maintained (~6-10%)
        → Different KC subsets for different patterns (0 overlap)
```

## Comparison with Literature

| Study | Task | Architecture | Accuracy |
|-------|------|-------------|----------|
| This work | Binary stripes | Evolved MB | **~99%** |
| Huerta et al. 2004 | Odor | MB model | ~90% |
| Peng & Bhatt 2007 | Simple patterns | SNN + STDP | ~85% |
| Ardin et al. 2016 | Navigation | MB route memory | ~80% |

Our result exceeds typical MB model benchmarks, likely because:
1. Evolution optimizes the full parameter space (not just synaptic weights)
2. The binary classification task is relatively simple
3. Sparse coding is properly enforced

## Files Generated
- `docs/figures/evolution_results.png` — Fitness curve + genome analysis
- `docs/journal/evolution_log.json` — Per-generation statistics
- `docs/journal/evolution_result.json` — Final summary
- `docs/journal/best_genome.npz` — Best evolved genome (weights + thresholds)

## Next Steps (Phase 2)
1. Scale to KC=2000 with GPU acceleration
2. MNIST digit subset (0 vs 1)
3. Ablation study: which mutations matter most?
4. Combined STDP + evolution
5. Visualization of evolved vs initial brain activity
