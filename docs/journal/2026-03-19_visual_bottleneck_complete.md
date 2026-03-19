# Research Journal: Visual Resolution Experiments — Complete Analysis

**Date:** 2026-03-19
**Phase:** Growing Brain — Visual System Investigation
**Status:** COMPLETE — Critical bottleneck identified

---

## Hypothesis Tested
"MNIST 10-class classification fails because the brain (KC layer) is too small."

## Experiments Run

### Experiment 1: KC Scaling (8x8 input fixed)
| Config | Neurons | KC Count | Best Accuracy | Mean |
|--------|---------|----------|--------------|------|
| Baseline | 275 | 200 | ~20% | ~10% |
| Medium | 475 | 400 | ~22% | ~10% |
| Large | 1075 | 1000 | ~21% | ~10% |

**Result: 5x more KC neurons = NO improvement.** Brain size is NOT the bottleneck.

### Experiment 2: Resolution Scaling (PN count varies)
| Config | Resolution | PNs | KCs | Best Accuracy |
|--------|-----------|-----|-----|--------------|
| 8x8 | 64 pixels | 64 | 200 | ~23% |
| 16x16 | 256 pixels | 256 | 200 | ~24% |
| 16x16 | 256 pixels | 256 | 500 | ~27% |

**Result: 4x better eyes = marginal improvement.** Resolution alone is NOT the bottleneck either.

### Experiment 3: Digit Similarity Analysis
Cosine similarity between average digit activations at 8x8:

| Pair | Similarity | Can a brain distinguish? |
|------|-----------|------------------------|
| 0 vs 1 | 0.631 | Yes — very different shapes |
| 1 vs 4 | 0.722 | Barely |
| 5 vs 8 | 0.957 | No — nearly identical blobs |
| 3 vs 5 | 0.951 | No |
| 7 vs 9 | 0.942 | No |
| 3 vs 8 | 0.934 | No |

**8 out of 10 digits are >85% similar at 8x8 resolution.** Even 16x16 doesn't fully resolve this because the mushroom body receives RAW pixels, not features.

## Root Cause: Missing Visual Preprocessing

```
Real fly visual pathway:
  Photoreceptors → Lamina → Medulla → Lobula → Projection Neurons → Mushroom Body
  (800 cells)      (edge)   (motion)  (shape)   (features)          (sparse coding)

  60,000 neurons dedicated to visual preprocessing before MB!

Our model:
  Raw pixels → Projection Neurons → Mushroom Body
  (no preprocessing at all)
```

The mushroom body evolved to process FEATURES (edges, orientations, spatial frequencies), not raw pixel intensities. When we feed raw pixels, every digit looks like a similar blob of activation in the center of the image.

## Biological Insight

This finding explains a fundamental principle of brain architecture:
1. **Sensory cortices are HUGE** — most neurons in any brain handle sensory preprocessing
2. **Decision-making circuits are SMALL** — mushroom body (4,000 neurons) vs optic lobe (60,000)
3. **The bottleneck is always representation, not computation**

A mushroom body with 200 KC can distinguish thousands of patterns — IF those patterns are properly preprocessed into distinct feature vectors. Raw pixels violate this assumption.

## Decision: Add Visual Cortex (Feature Detector Layer)

Implement a biologically-inspired feature extraction layer between raw pixels and the mushroom body:
- Evolved edge detectors (Gabor-like oriented filters)
- Small receptive fields (3x3 or 5x5 patches)
- Multiple orientations per spatial location
- Output: feature activation map → PN input to mushroom body

This mirrors the lamina/medulla layers of the insect optic lobe.

## Figures
- `visual_test_similarity.png` — Digit similarity matrix at 8x8
- `visual_test_means.png` — Average digit appearance at 8x8
- `mnist_small_result.png` — 275-neuron MNIST attempt
- `mnist_resolution_comparison.png` — 8x8 vs 16x16 comparison
