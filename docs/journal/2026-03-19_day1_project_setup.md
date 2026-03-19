# Research Journal — Day 1: Project Setup & Baseline Architecture

**Date:** 2026-03-19
**Phase:** Infrastructure & Connectome Construction
**Status:** Complete

---

## Objectives

1. Establish project structure with reproducible build system
2. Implement synthetic mushroom body connectome based on Drosophila literature
3. Validate Brian2 spiking neural network simulation
4. Generate baseline visualizations of network architecture

## Decisions & Rationale

### D1: Synthetic Connectome over FlyWire API
**Decision:** Generate synthetic connectome rather than pulling from FlyWire CAVE API.
**Rationale:** FlyWire API requires authentication and data processing pipeline. Synthetic generation using published parameters (Aso et al. 2014) gives us full control over network topology while maintaining biological plausibility. Real connectome integration is planned for Phase 2.

### D2: Reduced KC Count for Development (200 vs 2000)
**Decision:** Start with 200 Kenyon Cells instead of biological 2000.
**Rationale:** Brian2 simulation time scales quadratically with neuron count. 200 KC is sufficient to validate sparse coding, STDP learning, and evolution pipeline. Will scale to 2000 after pipeline validation.

### D3: 2 MBON Output Neurons for Binary Classification
**Decision:** Simplified from biological ~21 MBON types to 2.
**Rationale:** Binary classification (horizontal vs vertical stripes) requires only 2 output classes. This minimal setup isolates whether the mushroom body architecture itself can support learned discrimination, before adding complexity.

### D4: Anti-Hebbian STDP at KC→MBON
**Decision:** Use anti-Hebbian STDP (A_post = -0.0105, slight LTD bias) at KC→MBON synapses.
**Rationale:** Biological evidence shows KC→MBON synapses undergo depression during learning in Drosophila (Hige et al. 2015). The learned odor representation is encoded as *reduced* MBON response, not enhanced.

### D5: Rate Coding for Sensory Input
**Decision:** Poisson rate coding — pixel intensity maps to firing rate (0-100 Hz).
**Rationale:** Simplest encoding that preserves spatial information. Each pixel drives one Projection Neuron. 8x8 image = 64 PNs (subset of 150 PN pool). Alternative temporal coding schemes can be explored later.

## Implementation Details

### Connectome Parameters
| Parameter | Value | Biological Reference |
|-----------|-------|---------------------|
| PN count | 150 | ~150-200 in adult Drosophila |
| KC count | 200 (dev) / 2000 (prod) | ~2000 in γ, α'β', αβ lobes |
| MBON count | 2 | Simplified from ~21 types |
| APL count | 1 | 1 per hemisphere |
| KC input convergence | 6 PNs per KC | 5-7 (Caron et al. 2013) |
| PN→KC density | ~4.0% | ~3-5% estimated |
| KC→MBON | All-to-all | Dense connectivity observed |

### Neuron Model: Leaky Integrate-and-Fire (LIF)
- Conductance-based synapses (excitatory + inhibitory)
- KC threshold: -50mV (high, enforces sparse firing)
- PN threshold: -55mV (lower, more responsive to input)
- APL threshold: -45mV (lowest, fires easily for global inhibition)

### STDP Parameters
- Pre-post window: 20ms (tau_pre = tau_post = 20ms)
- LTP magnitude: A_pre = 0.01
- LTD magnitude: A_post = -0.0105 (5% LTD bias → net depression)
- Weight bounds: [0.0, 1.0]

## Validation Results

### Test Suite: 19/19 PASSED
- Connectome shape and sparsity: PASS
- KC input count (exactly 6 per KC): PASS
- KC→MBON all-to-all connectivity: PASS
- APL connections: PASS
- Seed reproducibility: PASS
- Different seeds produce different connectomes: PASS
- Brian2 network construction: PASS
- Neuron model equations and parameters: PASS
- Genome creation and sparsity: PASS
- Mutation preserves shape: PASS
- Crossover produces valid offspring: PASS
- Rate encoding: PASS
- Poisson spike generation: PASS
- **Integration test — PN neurons fire with input: PASS**

### Key Observation from Integration Test
PN neurons successfully fire when driven by Poisson spike input (50ms simulation, 100Hz max rate). This confirms the conductance-based LIF model parameters are in a biologically reasonable regime.

## Figures Generated
1. `mushroom_body_architecture.png` — 4-panel connectome overview
2. `input_patterns.png` — Binary classification stimuli with rate encoding
3. `spike_raster_comparison.png` — Poisson spike trains for both patterns

## Errors & Corrections

### E1: Brian2 "unused object" warnings
**Error:** Brian2 warns that NeuronGroup/Synapses objects were never added to a Network.
**Cause:** Unit test calls `builder.build()` to check component creation, but doesn't construct a Network object.
**Resolution:** Expected behavior in test context. The `build_network()` method properly adds all components. No code change needed — warnings are informational only.

## Next Steps
1. Run baseline Brian2 simulation with STDP learning (no evolution)
2. Verify KC sparse coding emerges from APL inhibition
3. Verify MBON differential response to patterns A vs B after STDP training
4. If baseline works → proceed to neuroevolution pipeline

## Dependencies & Versions
- Python 3.14.0
- Brian2 2.10.1
- NumPy 2.4.3
- SciPy 1.17.1
- Matplotlib 3.10.8
