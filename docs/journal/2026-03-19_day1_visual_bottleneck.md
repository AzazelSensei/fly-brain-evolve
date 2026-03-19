# Research Journal: Visual Bottleneck Discovery

**Date:** 2026-03-19
**Finding:** CRITICAL — Input resolution is the bottleneck, not brain size

---

## Experiment: KC Scaling Test (8x8 input, 10 classes)

| Brain Size | KC Count | Best Accuracy | Mean | Time |
|-----------|---------|--------------|------|------|
| 275 neurons | 200 KC | ~20% | ~10% | 231s |
| 475 neurons | 400 KC | ~22% | ~10% | 379s |
| 1075 neurons | 1000 KC | ~21% | ~10% | 1708s |

**Result: Increasing KC count 5x produces ZERO improvement.**

## Root Cause: 8x8 Resolution Destroys Digit Information

Cosine similarity between average digit images at 8x8:
- 5 vs 8: 0.957 (nearly identical)
- 3 vs 5: 0.951
- 7 vs 9: 0.942
- Only 0 vs 1 is clearly separable (0.631)

Most digit pairs are >90% similar at 8x8. No amount of KC sparse coding can separate signals that are already mixed at the input level.

## Biological Parallel
Real flies have ~60,000 neurons in their optic lobe BEFORE information reaches the mushroom body. The visual preprocessing is far more complex than the "thinking" part. We skipped this entirely.

## Next Step: Increase visual resolution to 16x16 (256 PNs)
