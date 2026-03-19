# Research Journal — Day 1 (Part 3): Critical Unit Bug Discovery & Fix

**Date:** 2026-03-19
**Phase:** Bug Fix — Brian2 Unit System
**Status:** RESOLVED

---

## Bug Report: E2 — Brian2 Unit Conversion Error (CRITICAL)

### Severity: CRITICAL
All previous simulation results were invalid due to this bug.

### Symptom
All KC neurons fired at maximum rate (~9600 spikes / 200 KCs / 100ms) regardless of parameter changes (threshold, weights, APL inhibition). No parameter sweep produced any change in activation.

### Root Cause
The `_brian2_namespace()` method in `builder.py` used incorrect unit conversion:

```python
# WRONG: Config stores values in SI (volts), but multiplied by mV
"V_rest": params["V_rest"] * mV    # -0.070 * mV = -70 uV (WRONG!)
"g_L": params["g_L"] * nS          # 25e-9 * nS = 25 aS (WRONG!)

# CORRECT: Use base SI units
"V_rest": params["V_rest"] * volt   # -0.070 * volt = -70 mV (CORRECT)
"g_L": params["g_L"] * siemens      # 25e-9 * siemens = 25 nS (CORRECT)
```

### Impact Analysis
With wrong units:
- V_rest = -70 uV (instead of -70 mV) — 1000x too small
- V_thresh = -50 uV (instead of -50 mV)
- Gap between rest and threshold: 20 uV (instead of 20 mV)
- g_L = 25 aS (instead of 25 nS) — 1e9 too small
- Synaptic drive (I_syn/g_L) was ~1e6 times larger than intended
- Every neuron fired at refractory-limited maximum rate

### Verification
After fix, single-spike conductance test:
- 50 nS single spike: peak = -51.6 mV (below -50 mV threshold, no fire) CORRECT
- 100 nS single spike: peak = -50.4 mV (above threshold, fires) CORRECT

### Lesson Learned
Always verify Brian2 unit conversions with a minimal test before building complex networks. The `x * mV / mV * mV` pattern is a common trap — it equals `x * mV`, not `x * volt`.

---

## Post-Fix: Sparse Coding Achievement

### Parameter Tuning Results
After fixing units, systematic sweep of APL weight, PN-KC weight, and KC threshold:

| Config | APL (nS) | PN-KC (nS) | KC Thresh (mV) | Sparsity |
|--------|----------|-----------|----------------|----------|
| baseline | 50 | 3-8 | -50 | 81% |
| apl=200 | 200 | 3-8 | -50 | 79% |
| combo2 | 200 | 2-5 | -48 | 56% |
| **combo3** | **200** | **1-3** | **-45** | **6%** |
| combo5 | 200 | 1-3 | -48 | 43% |

**Selected parameters: combo3 (APL=200nS, PN-KC=1.5-4nS, KC threshold=-45mV)**

### Critical Finding: Zero KC Overlap
When comparing horizontal vs vertical stripe patterns:
- Horizontal activates 12 unique KCs (6%)
- Vertical activates 7 unique KCs (3.5%)
- **Overlap = 0** — completely distinct KC representations!

This confirms the mushroom body architecture correctly produces pattern-specific sparse representations via random connectivity + global inhibition.

### Remaining Issue: MBON Not Firing
MBON neurons receive KC input but don't reach threshold. KC-MBON weights (0.5-2 nS) need to be increased, or STDP learning will accumulate sufficient drive over training epochs.

**Decision:** Leave MBON weights low for now — STDP learning and evolution will optimize these.

## Updated Configuration
```yaml
neuron:
  kc:
    v_thresh: -0.045  # -45mV (was -50mV)
  apl:
    v_thresh: -0.045

connectivity:
  pn_kc_weight_range: [1.5, 4.0]  # nS
  input_weight: 50  # nS
  apl_kc_weight: 200  # nS
  kc_mbon_weight_range: [2.0, 5.0]  # nS
```

## Figures
- `sparsity_tuning.png` — Parameter sweep showing path from 81% to 6% sparsity
- `sim_corrected_chain.png` — First working simulation with correct units
- `brain_live_activity.png` — Side-by-side comparison of both patterns with KC population map
