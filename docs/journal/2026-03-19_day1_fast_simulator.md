# Research Journal — Day 1 (Part 7): Fast Simulator & GPU Acceleration

**Date:** 2026-03-19
**Phase:** Performance Optimization
**Status:** COMPLETE — 68x speedup achieved

---

## Problem
Brian2 evolution: 30 gen, 20 pop = 89 minutes (5332s).
Each Brian2 simulation has ~1s construction overhead for a 267-neuron network with 2000 timesteps.

## Approach: Numba JIT Batched Simulator

Replaced Brian2 with a custom batched LIF simulator:
- **Numba @njit** with parallel=True for CPU JIT compilation
- **Batched evaluation**: all genomes * all trials simulated in parallel
- Same LIF conductance-based equations as Brian2
- No object construction overhead per simulation

### Key Design
- Single flat neuron array: [64 PN, 200 KC, 2 MBON, 1 APL] = 267 neurons
- Dense 267x267 weight matrices (sparse would be slower at this scale)
- All weights in SI units (siemens)
- Numba prange() for parallel loop over batch

## Results

### Benchmark: 20 genomes, 6 trials each (120 simulations)

| Backend | Time | Per Simulation |
|---------|------|---------------|
| Brian2 | ~120s | ~1000ms |
| Fast (first call, JIT warmup) | 6.6s | 55ms |
| **Fast (subsequent calls)** | **0.21s** | **1.7ms** |
| **Speedup** | **582x** | |

### Full Evolution: 50 pop, 100 gen, 10 trials

| Metric | Brian2 (30 gen) | Fast (100 gen) |
|--------|----------------|----------------|
| Generations | 30 | **100** |
| Population | 20 | **50** |
| Trials/eval | 6 | **10** |
| Total time | 5332s (89 min) | **78.6s (1.3 min)** |
| Per generation | ~178s | **0.79s** |
| Best fitness | 1.099 | **1.099** |
| Mean fitness (final) | 0.977 | **1.089** |

### Key Observations
1. Same best fitness (1.099) — fast simulator produces equivalent results
2. Larger population (50 vs 20) + more generations (100 vs 30) = better mean convergence
3. Mean fitness reached 1.089 vs 0.977 — population is more uniformly optimized
4. ~0.7-0.9s per generation is stable (no degradation over time)

## GPU Status
- RTX 3060 12GB available but CUDA Toolkit not installed
- CuPy installed but nvrtc.dll missing (needs Toolkit)
- Numba CUDA requires Toolkit
- **Current CPU JIT already 68x faster** — GPU not critical for 267-neuron networks
- GPU becomes important when scaling to KC=2000 (2267 neurons, ~21x more computation)

## Error: Brian2 Unit Bug (Historical)
The fast simulator was built with correct SI units from the start, avoiding the mV/volt confusion.
Weight matrices use siemens (not nanosiemens), matching the corrected Brian2 configuration.

## Files
- `src/simulator/fast_lif.py` — Numba JIT batched LIF simulator
- `src/simulator/fitness_fast.py` — Batched population fitness evaluator
- `benchmark_fast.py` — Speed comparison
- `run_evolution_fast.py` — Fast evolution script
