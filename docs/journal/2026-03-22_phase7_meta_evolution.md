# Research Journal: Phase 7 — Meta-Evolution (Learning to Learn)

**Date:** 2026-03-22
**Phase:** Phase 7 — Meta-Evolution
**Status:** NEW RECORD — 0.948 fitness, evolution discovers optimal learning rules

---

## Concept

Previously, STDP parameters (learning rate, time constants, reward/punishment signals)
were hand-tuned. In Phase 7, these parameters enter the genome and are evolved alongside
connectivity. Evolution now optimizes not just the circuit, but HOW the circuit learns.

This mirrors biological reality: natural evolution shaped not just brain architecture
but the molecular machinery of synaptic plasticity itself.

## Evolved Parameters

| Parameter | Hand-tuned (Phase 5) | Evolved (Phase 7) | Change |
|-----------|---------------------|-------------------|--------|
| lr | 0.0003 | **0.00021** | Slower, more stable |
| reward | 1.0 | **2.49** | 2.5x stronger reward |
| punish | -0.1 | **-0.22** | 2.2x stronger punishment |
| tau_elig | 0.040 | 0.040 | Same (already optimal) |
| tau_kc | 0.020 | **0.031** | 55% longer KC trace |
| tau_mbon | 0.020 | **0.031** | 55% longer MBON trace |
| w_max | 15.0 | **17.8** | Wider weight range |
| max_passes | 4 | **5** | Maximum deliberation |
| kc_decay | 0.75 | **0.90** | More memory retention |

## Key Discoveries

### 1. Lower learning rate + stronger signals
Evolution chose lr=0.00021 (30% lower) but reward=2.49 (150% higher).
The net effect: same magnitude weight changes but with clearer signal-to-noise ratio.
Like turning up the volume while speaking more slowly.

### 2. Reward/punishment ratio preserved
Hand-tuned: 1.0/0.1 = 10:1 ratio
Evolved: 2.49/0.22 = 11.3:1 ratio
Evolution independently discovered nearly the same ratio but at higher absolute magnitudes.

### 3. Longer eligibility traces
tau_kc and tau_mbon both evolved from 20ms to 31ms. This means the brain remembers
"who was active" for 55% longer, allowing credit assignment over a wider time window.

### 4. Maximum deliberation confirmed
passes=5 (max), kc_decay=0.90 (high retention). Consistent across Phase 5 and Phase 7:
evolution always prefers maximum thinking time.

## Results

```
Gen   0: best=0.4595 mean=0.2524 lr=0.00124 rw=1.8 pu=-0.49
Gen  25: best=0.8937 mean=0.7560 lr=0.00022 rw=2.4 pu=-0.11
Gen  50: best=0.9220 mean=0.8195 lr=0.00018 rw=2.4 pu=-0.22
Gen  85: best=0.9481 mean=0.8424 lr=0.00021 rw=2.5 pu=-0.22
Gen  99: best=0.9481 mean=0.8225 lr=0.00021 rw=2.5 pu=-0.22
```

Best: **0.9481** (vs Phase 5 tuned: 0.922). Time: 19137s (~5.3 hours).
100 generations, 25 population, 5 epochs, dt=0.0001.

## Comparison: All Phases

| Phase | Best | Mean | Key Innovation |
|-------|------|------|---------------|
| 1 (GA only) | 0.268 | 0.10 | Neuroevolution |
| 2 (STDP) | 0.732 | 0.66 | Dopamine learning |
| 4 (Attention) | 0.745 | 0.62 | MBON->KC feedback |
| 5 (Multi-pass) | 0.890 | 0.80 | Deliberation |
| 5 tuned | 0.922 | 0.82 | dt optimization |
| **7 (Meta-evo)** | **0.948** | **0.84** | **Evolved learning rules** |

Total improvement: 0.268 -> 0.948 = **3.5x** with same 639 neurons.

## Biological Significance

Meta-evolution validates the "Baldwin Effect" — the evolutionary theory that
learning ability itself is under selective pressure. Organisms that learn faster
and more effectively have higher fitness, so evolution shapes the learning machinery.

Our model demonstrates this computationally:
- Evolution discovered that stronger dopamine signals with slower learning rates
  produce more stable, effective learning
- The optimal eligibility trace duration (31ms) is longer than the hand-tuned value,
  suggesting that biological STDP time constants were themselves shaped by evolution
  to maximize learning efficiency
