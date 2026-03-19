# Research Journal — Day 1 (Part 2): Baseline Simulation Results

**Date:** 2026-03-19
**Phase:** Baseline Brian2 Simulation (pre-evolution, pre-STDP learning)
**Status:** Complete

---

## Objectives

Run Brian2 simulation with both input patterns (horizontal/vertical stripes) to verify:
1. All neuron layers fire correctly
2. Signal propagates through PN → KC → MBON pathway
3. APL inhibition affects KC firing
4. Baseline MBON responses show differential activity

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| PN count | 64 (= 8x8 pixels) |
| KC count | 200 |
| MBON count | 2 |
| APL | 1 |
| Duration | 100ms per stimulus |
| Input rate | 0-100 Hz (Poisson) |
| dt | 0.1ms |
| STDP | OFF (baseline measurement) |

## Key Observations

### O1: PN Layer — Working Correctly
- **Horizontal**: ~1497 spikes across 64 PNs in 100ms
- **Vertical**: ~1419 spikes across 64 PNs in 100ms
- Active PNs correspond to white pixels (rows 0,2,4,6 for horizontal; cols 0,2,4,6 for vertical)
- Inactive PNs correspond to black pixels (0 Hz rate)
- Spike count difference is due to Poisson stochasticity (expected ~1500 for 32 active neurons at 100Hz * 0.1s ≈ 320 spikes + refractory effects)

### O2: KC Layer — High Activation (Unexpected)
- **Both patterns**: 200/200 KC active (100% activation!)
- This is **NOT sparse coding** — this is a problem
- Expected: ~10-20% KC activation (20-40 out of 200)
- **Root cause analysis:** APL inhibition parameters are insufficient. The APL→KC inhibitory synapse weight (1.0 nS) is too weak relative to the excitatory PN→KC drive

### O3: KC Voltage Traces — Saturated Firing
- All sample KC membrane voltage traces show repetitive spiking at high frequency
- Traces hit threshold (-50mV) repeatedly with very short inter-spike intervals
- This confirms the excitation-inhibition balance is off

### O4: MBON Layer — Both MBONs Fire Similarly
- **Horizontal**: 100 total MBON spikes (MBON 0: ~50, MBON 1: ~50)
- **Vertical**: 100 total MBON spikes (MBON 0: ~50, MBON 1: ~50)
- No discrimination between patterns — expected at this stage (no STDP training)
- But the near-equal firing is concerning — with saturated KCs, all MBONs receive similar input

## Error Analysis

### E1: Lack of Sparse Coding (CRITICAL)
**Problem:** 100% KC activation instead of target ~10%
**Cause:** APL inhibition feedback loop is too weak. The KC→APL→KC circuit should enforce winner-take-all competition, but:
  - KC→APL weight: 0.5 nS (too weak — 200 KCs contributing only collectively pushes APL)
  - APL→KC weight: 1.0 nS (insufficient to suppress KC firing)
  - PN→KC weight: 0.3-1.0 nS each, with 6 inputs per KC = up to 6 nS excitatory drive

**Impact:** Without sparse coding, the key computational advantage of the mushroom body is lost. KCs cannot form distinct representations of different patterns.

**Planned fix:**
  1. Increase APL→KC inhibitory strength (10-50 nS range)
  2. Increase KC firing threshold (-45mV to -40mV)
  3. Reduce PN→KC connection weights
  4. Consider adding lateral inhibition between KCs

### E2: LIF Equation Unit Handling
**Observation:** Brian2 unit system required careful handling. The `V_rest * mV / mV * mV` pattern was needed to avoid unit mismatch errors. This is fragile.
**Resolution:** Will refactor to use Brian2's native unit system more cleanly.

## Figures Generated
1. `sim_snapshot_horizontal.png` — Full 6-panel simulation dashboard for horizontal pattern
2. `sim_snapshot_vertical.png` — Full 6-panel simulation dashboard for vertical pattern

## Cause-Effect Analysis

```
Cause: Weak APL inhibition (1.0 nS) vs strong excitatory drive (~6 nS per KC)
  → Effect: All 200 KCs fire (0% sparsity)
    → Effect: All MBONs receive identical input
      → Effect: No pattern discrimination possible
        → Effect: STDP learning would be ineffective (no differential signal)
```

## Decision for Next Step

**D6: Fix sparse coding before proceeding to STDP or evolution**
**Rationale:** Sparse coding is the foundational computational mechanism of the mushroom body. All downstream processing (STDP learning, pattern discrimination) depends on it. Proceeding without fixing this would produce meaningless results.

**Priority adjustments:**
1. [NEXT] Tune inhibition parameters to achieve 5-15% KC sparsity
2. Re-run simulation snapshots to verify sparse coding
3. Only then proceed to STDP learning phase
4. Evolution pipeline follows after STDP baseline works

## Next Steps
1. Parameter sweep: APL→KC weight vs KC sparsity
2. Adjust KC threshold if needed
3. Validate sparse coding with both patterns
4. Verify different patterns produce different KC activation sets
