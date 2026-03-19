# Research Journal — Phase 2 (Part 1): Dopamine Learning First Attempt

**Date:** 2026-03-19
**Phase:** Neuromodulated Learning
**Status:** FAILED — learning rate too high, destabilizes weights

---

## Experiment
- Evolution + Dopamine-modulated STDP hybrid
- 3 training epochs per fitness evaluation
- learning_rate=0.5, tau_eligibility=50ms
- Dopamine: +1 (correct), -1 (wrong), 0 (no output)

## Result
- Gen 0: best=0.97 (inherited from random init)
- Gen 5-49: oscillates around 0.5-0.6 (near chance)
- Dopamine learning overwrites evolved weights destructively

## Root Cause
Learning rate 0.5 with 3 training epochs causes catastrophic weight changes.
Eligibility trace accumulates too much, and dopamine multiplier amplifies it.

**Fix needed:**
1. Lower learning rate: 0.5 → 0.01-0.05
2. Shorter eligibility trace: 50ms → 20ms
3. Fewer training epochs per eval: 3 → 1
4. Or: dopamine only modulates a FRACTION of the weight change
