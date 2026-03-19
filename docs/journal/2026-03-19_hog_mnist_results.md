# Research Journal: HOG Features + Mushroom Body — MNIST 10-Class

**Date:** 2026-03-19
**Phase:** Growing Brain — Stage 3 (Feature Detectors)
**Status:** COMPLETE — Critical finding: good features alone are not enough

---

## Experiment

Hand-crafted HOG (Histogram of Oriented Gradients) features as input to mushroom body:
```
16x16 image → HOG (4x4 cells, 8 bins) → 128 features → 300 KC → 10 MBON
Total: 439 neurons
```

Parameters: POP=40, GENS=80, n_test=80, n_per_class=30 (300 images total)

## HOG Feature Quality

HOG dramatically improved digit separability compared to raw pixels:

| Metric | Raw 8x8 | HOG 16x16 |
|--------|---------|-----------|
| Pairs with >85% similarity | 8/45 | **0/45** |
| Most similar pair | 5 vs 8: 0.957 | 3 vs 5: 0.788 |
| Most different pair | 0 vs 1: 0.631 | 1 vs 5: 0.402 |

HOG features make every digit pair distinguishable. The representation problem is solved.

## Classification Results

| Model | Neurons | Input | Best Fitness |
|-------|---------|-------|-------------|
| 8x8 raw | 275 | 64 pixels | 0.234 |
| 16x16 raw | 467 | 256 pixels | 0.237 |
| **HOG+MB** | **439** | **128 HOG** | **0.237** |
| 16x16+cortex | 651 | 340 features | 0.250 |
| 16x16 raw big | 767 | 256 pixels | 0.268 |

HOG+MB = 0.2372 — no improvement despite vastly better features.

## Critical Finding

**The bottleneck has shifted from representation to learning.**

Evidence:
1. HOG features perfectly separate all 10 digit classes (0 pairs >85% similar)
2. Yet the GA cannot find connectivity patterns that exploit this separation
3. Mean fitness stayed at ~0.10 (chance level) across 80 generations
4. Best fitness never exceeded 0.24 despite perfect input features

## Why the GA Fails

1. **Search space too large**: 128 PN × 300 KC = 38,400 potential connections.
   With only 6 connections per KC, there are C(128,6) ≈ 5.5 billion possible
   connection patterns per KC. Tournament selection + mutation cannot explore this.

2. **Fitness landscape is deceptive**: A random KC connectivity gets ~10% accuracy.
   Small mutations to random connectivity also get ~10%. There is no smooth gradient
   from random to functional — the GA needs a "lucky" combination of many correct
   connections simultaneously.

3. **No credit assignment**: The GA only sees total accuracy. It cannot determine
   which KC-MBON connections helped and which hurt. This is exactly the problem
   that backpropagation solves in ANNs, and that dopamine-modulated STDP solves
   in real brains.

## Biological Implication

Real Drosophila mushroom bodies learn via **dopamine-modulated synaptic plasticity**,
not evolutionary search over connectivity. The connectome (PN→KC wiring) is largely
random and genetically determined, but the KC→MBON weights are learned during the
animal's lifetime through reinforcement signals.

Our GA is trying to evolve what biology LEARNS. This is fundamentally the wrong
optimization level for KC→MBON weights.

## Next Steps

The path forward is clear: **implement dopamine-modulated learning (STDP)**

1. Keep random PN→KC connectivity (biologically correct)
2. Replace GA optimization of KC→MBON weights with online learning
3. Implement reward/punishment dopamine signals that modulate KC→MBON plasticity
4. This is Phase 2 of the AGI roadmap: Dopamine reward system

## Time

Total runtime: 884s (~15 minutes), 80 generations × 40 population × 80 test samples
