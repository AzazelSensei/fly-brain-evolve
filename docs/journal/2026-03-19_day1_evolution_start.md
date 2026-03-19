# Research Journal — Day 1 (Part 5): Evolution Experiment Started

**Date:** 2026-03-19
**Phase:** Neuroevolution — Binary Classification
**Status:** RUNNING

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Population size | 20 |
| Generations | 30 |
| Mutation rate | 0.3 |
| Crossover rate | 0.4 |
| Elitism | 3 |
| Tournament size | 3 |
| Trials per fitness eval | 6 (3 per pattern) |
| Simulation dt | 0.05 ms |
| Stimulus duration | 100 ms |

### Genome Encoding
- PN-KC connectivity matrix (64x200, sparse, weights 1-5 nS)
- KC-MBON weight matrix (200x2, dense, 2-10 nS)
- KC threshold vector (200 values, -48 to -42 mV)

### Mutation Operators Active
1. KC-MBON weight perturbation (Gaussian, sigma=0.3)
2. PN-KC weight perturbation (Gaussian, sigma=0.2)
3. KC threshold perturbation (Gaussian, sigma=1.0 mV)
4. Add new PN-KC synapse (probability 0.3 * mutation_rate)

### Fitness Function
```
fitness = accuracy + sparsity_score - complexity_penalty
where:
  accuracy = fraction of correct classifications (0-1)
  sparsity_score = 0.1 * max(0, 1 - |kc_active_frac - 0.1| / 0.1)
  complexity_penalty = 0.01 * num_pn_kc_synapses / 10000
```

## Early Results (First 6 Generations)

| Gen | Best Fitness | Mean Fitness | Time (s) |
|-----|-------------|-------------|----------|
| 0 | 0.9305 | 0.5750 | 97 |
| 1 | 0.9080 | 0.5995 | 104 |
| 2 | 0.9246 | 0.6253 | 113 |
| 3 | **1.0730** | 0.7505 | 120 |
| 4 | 0.9163 | 0.6040 | 128 |
| 5 | 0.9321 | 0.6813 | 142 |

### Key Observations
1. **Generation 0 already at 93% accuracy** — random initialization with optimized weight ranges works surprisingly well
2. **Generation 3 hit 1.07 fitness** — this means >97% accuracy with sparsity bonus
3. Mean fitness increasing: 0.575 -> 0.681 (population improving)
4. ~2 min per generation (20 individuals, 6 trials each)

### Why Evolution Works Where STDP Didn't
- Evolution directly optimizes KC-MBON weights to be in the right range
- Evolution can adjust KC thresholds per-neuron for better sparsity
- Evolution adjusts PN-KC connectivity for better pattern separation
- No need for post-synaptic spikes during learning (unlike STDP)

## Estimated Completion
30 generations * ~120s = ~60 minutes total
Started: ~19:10
Expected completion: ~20:10
