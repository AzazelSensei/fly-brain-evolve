# Research Journal — Phase 2: Dopamine Learning Critical Findings

**Date:** 2026-03-19
**Phase:** Neuromodulated Learning — Three-Stage Hybrid
**Status:** Significant scientific finding

---

## Key Finding: Dopamine Learning is Destructive in This Regime

### Stage 2 Results (Dopamine Parameter Sweep)

The best dopamine configuration was lr=0.01 with only 2 epochs (fitness=1.094).
Every other configuration REDUCED fitness below the pre-evolved baseline:

| LR | Epochs | Fitness | W_change | Verdict |
|----|--------|---------|----------|---------|
| 0.01 | 2 | **1.094** | 0.94 | Nearly preserved |
| 0.01 | 5 | 0.494 | 1.82 | Destroyed |
| 0.02 | 2 | 0.644 | 1.53 | Destroyed |
| All others | * | ~0.5 | >2.0 | Destroyed |

**Pattern:** Even minimal dopamine learning (lr=0.01, 2 epochs) causes ~0.94 nS average weight change. Anything more than this destroys the evolved solution.

### Stage 3 Results (Combined Evolution + Dopamine)
When dopamine learning was applied inside the evolution loop:
- Gen 50 (just started): best=1.099 (carried from Stage 1)
- Gen 55: dropped to 0.797
- Gen 75: stabilized around 0.56-0.79

**The dopamine learning fights evolution.** Evolution optimizes weights for classification; dopamine learning shifts them based on individual trial outcomes. The two forces pull in different directions.

## Why This Happens — Deep Analysis

### The Sparse Coding Bottleneck
With only ~12-18 KCs active per pattern and ~200 KC→MBON synapses per MBON:
- Only ~6-9% of synapses receive any pre-synaptic activity
- These few active synapses accumulate ALL the eligibility trace
- A single dopamine signal modifies them disproportionately
- After 2+ epochs, these critical synapses are shifted enough to change the MBON output
- The carefully evolved weight balance is disrupted

### Biological Insight
This actually mirrors a real biological constraint:
1. In real Drosophila, dopamine learning is VERY slow (many trials needed)
2. The mushroom body uses anti-Hebbian STDP (depression-dominant), not potentiation
3. Learning in flies is about REDUCING specific MBON responses, not increasing them
4. The fly brain's architecture was shaped by ~600 million years of evolution before any individual fly learns anything

**Our finding supports the hypothesis that the mushroom body's parameter regime was set by evolution, and online learning operates as a very gentle fine-tuner within a narrow range.**

## Implications for AGI Roadmap

### What This Means
1. **Evolution sets the stage, learning fine-tunes** — this is the correct hierarchy
2. **Learning rate must be tiny** relative to evolved weight magnitudes
3. **The 3-factor rule needs a more sophisticated dopamine signal** — not just +1/-1 but graded, pattern-specific

### Revised Approach for Phase 2
Instead of dopamine modifying KC→MBON weights directly, implement:
1. **Dopamine as a gating signal** — controls WHICH synapses are eligible for modification
2. **Compartmentalized learning** — different MBON compartments learn independently
3. **Depression-dominant rule** — align with biology (reduce, don't increase)
4. **Homeostatic constraint** — total weight per MBON stays roughly constant

## Decision
**D8: Proceed to Phase 3 (multi-class + hierarchical) using pure evolution.**
Dopamine learning will be revisited with a more sophisticated model in Phase 4 when we have attention/gating mechanisms that can properly regulate it.

The core evolution engine works perfectly (~99% accuracy in <80 seconds). Scaling to more complex tasks is the higher priority.
