# Research Journal: Visual Cortex V1 — First Attempt

**Date:** 2026-03-19
**Phase:** Growing Brain — Stage 3 (Feature Detectors)
**Status:** No improvement yet, but evolved filters show promise

---

## Experiment
Added 8 evolved 3x3 Gabor-like filters between raw pixels and mushroom body:
```
16x16 image → 8 filters (stride=2) → 340 features → 300 KC → 10 MBON
Total: 651 neurons
```

## Results Comparison

| Model | Neurons | Input | Best Fitness |
|-------|---------|-------|-------------|
| 8x8 raw | 275 | 64 pixels | 0.234 |
| 16x16 raw | 467 | 256 pixels | 0.237 |
| 16x16 raw big | 767 | 256 pixels | 0.268 |
| **16x16 + cortex** | **651** | **340 features** | **0.250** |

Visual cortex did NOT improve over raw 16x16 with larger brain (0.250 vs 0.268).

## Why It Didn't Work (Yet)

1. **Filter evolution too slow**: 8 filters × 9 parameters = 72 parameters to evolve.
   With only 60 generations and mutation rate 0.3, filters barely change from random init.

2. **Stride too large**: stride=2 on 16x16 → 7x7 output per filter. Loses spatial detail.

3. **Fixed filter bank**: All genomes use the same Gabor initialization.
   Need more diverse starting points.

4. **Search space too large**: Evolving filters + connectivity + weights simultaneously
   is harder than evolving each separately.

## Positive Signal
The evolved 3x3 filters (visible in figure) show oriented edge-like structures.
Evolution IS discovering useful visual features, but needs more time and better search strategy.

## Next Steps
Two approaches to try:
A. **Pre-evolved filters**: First evolve ONLY the filters on a simple edge detection task,
   then freeze them and evolve the mushroom body
B. **Classical CV features**: Use hand-crafted HOG/edge features (skip filter evolution),
   focus on whether the MB can classify with good features
