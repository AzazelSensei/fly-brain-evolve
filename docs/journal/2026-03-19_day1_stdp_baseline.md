# Research Journal — Day 1 (Part 4): STDP Baseline Results

**Date:** 2026-03-19
**Phase:** Baseline STDP Learning (pre-evolution)
**Status:** STDP alone insufficient, proceeding to evolution

---

## Experiment: STDP-Only Binary Classification

### Setup
- KC threshold: -45 mV, APL inhibition: 200 nS
- KC sparsity: ~6% (verified from previous experiments)
- KC-MBON initial weight: 3.0 nS (V1) then 8.0 nS (V2)
- STDP: Anti-Hebbian (A_pre=0.01, A_post=-0.012)
- 20-30 training epochs, 10 test trials per pattern per epoch

### Results V1 (w_init=3.0): Weights unchanged, accuracy at chance
- Weight change after 20 epochs: 3.000 -> 3.001 (negligible)
- Accuracy: oscillates around 50% (chance level)
- Cause: MBON never fires (0 spikes) -> no post-synaptic events -> STDP inactive

### Results V2 (w_init=8.0): Minimal weight change, MBON still silent
- Weight range after 30 epochs: [7.96, 8.25]
- MBON spike count: 0 in all test trials
- Accuracy: 50% average (pure chance)
- Weights show slight increase (LTP slight dominance), but insufficient magnitude

### Root Cause Analysis

```
KC sparsity ~6% (12/200 active) with ~2 spikes each
-> Total KC spikes reaching each MBON: ~12 * 2 * 8nS = ~192nS cumulative
-> But spread over 100ms with tau_exc=5ms decay
-> Instantaneous g_exc << required to reach -50mV from -70mV
-> MBON stays sub-threshold -> 0 post-synaptic spikes
-> STDP eligibility traces are zero on post side
-> No meaningful weight update
```

### Why This Is Expected (Biological Context)
In real Drosophila, STDP at KC-MBON synapses is modulated by dopaminergic neurons (DANs). This is a **3-factor learning rule**:
1. Pre-synaptic KC spike
2. Post-synaptic MBON activity (or lack thereof)
3. **Dopamine signal** (reward/punishment)

Without the dopamine modulation, pure STDP is too slow for the sparse coding regime. The mushroom body evolved to work with modulatory signals, not raw Hebbian/anti-Hebbian learning.

### Decision: Proceed to Neuroevolution
**D7: Use evolution to optimize connectivity and weights instead of relying on STDP alone.**

Rationale: Evolution can simultaneously optimize:
- PN-KC connectivity structure (which PNs connect to which KCs)
- PN-KC weights (excitatory drive strength)
- KC thresholds (sparsity control per-neuron)
- KC-MBON weights (output mapping)

STDP will be kept as a fine-tuning mechanism during fitness evaluation, but the primary optimization comes from evolutionary search.

## Figures
- `baseline_stdp_results.png` — V1 results
- `stdp_v2_results.png` — V2 results with higher initial weights
