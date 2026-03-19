# Research Journal: GPU vs CPU Benchmark — Spiking Network Simulator

**Date:** 2026-03-20
**Phase:** Infrastructure — GPU Migration Planning
**Status:** COMPLETE — Critical finding: naive GPU is 1500x SLOWER for small networks

---

## Benchmark Setup

- **GPU:** NVIDIA RTX 3060 12GB, CUDA 12.9, CuPy 14.0.1
- **CPU:** Numba JIT (existing simulator)
- **Test:** Single simulation trial (2000 timesteps, 100ms at dt=0.05ms)

## Results: Naive CuPy Implementation

| Neurons | KC Count | CPU (Numba) | GPU (CuPy) | Winner |
|---------|----------|-------------|-----------|--------|
| 439 | 300 | ~2ms | 2998ms | CPU (1500x) |
| 639 | 500 | ~3ms | 3475ms | CPU (1158x) |
| 1139 | 1000 | ~8ms | 3438ms | CPU (430x) |
| 2139 | 2000 | ~20ms | 3440ms | CPU (172x) |
| 5139 | 5000 | ~80ms | 3265ms | CPU (41x) |

**GPU is slower at ALL tested sizes.** Even at 5000 neurons, CPU wins by 41x.

## Root Cause Analysis

Python-level timestep loop: `for t in range(2000)` launches GPU kernels 2000 times.
Each launch has ~1.5ms overhead: kernel dispatch + synchronization + host-device coordination.
2000 × 1.5ms = **3000ms overhead alone** — before any actual computation.

The actual GPU computation per timestep is fast (<0.1ms even at 5K neurons).
But the launch overhead dominates completely.

## When GPU Would Win

For GPU to overtake CPU, the per-timestep computation must exceed the launch overhead:
- At 5K neurons: GPU compute ~0.1ms, launch overhead ~1.5ms → overhead dominates
- At ~50K neurons: GPU compute ~1.5ms, launch overhead ~1.5ms → breakeven
- At 100K+ neurons: GPU compute >>1.5ms → GPU finally wins

**Alternative:** Fuse the entire 2000-timestep loop into a SINGLE CUDA kernel.
This eliminates 1999 kernel launches. Requires writing the full simulation in CUDA C.
Estimated development time: 1-2 weeks. Justified only for Phase 5+ (10K+ neurons).

## Proper GPU Strategy (for Phase 4+)

Based on the GPU planner agent's analysis:

### 1. Sparse Weight Representation (critical)
Current: Dense N×N matrices (N=5000 → 200MB per genome → 4GB for pop=20)
Better: Structured sparse (PN→KC: 6 connections per KC → 120KB per genome → 2.4MB for pop=20)
**4000x memory reduction.** Makes 15K neurons trivial on 12GB GPU.

### 2. Fused Kernel Architecture
Instead of per-timestep kernel launches:
- Kernel 1 (fused): Input + Synaptic current + Voltage update + Spike detection
- Kernel 2: Structured spike propagation (PN→KC, KC→MBON, KC↔APL separately)
- Kernel 3: STDP trace + eligibility update
- Kernel 4: Conductance decay (CuPy element-wise, no custom kernel)

### 3. Estimated GPU Speedup with Proper Implementation

| Neurons | CPU (Numba) | GPU (fused kernels) | Speedup |
|---------|-------------|-------------------|---------|
| 639 | 47s/gen | ~5s/gen | ~10x |
| 5,000 | ~48 min/gen | ~30-60s/gen | ~50-100x |
| 10,000 | ~3.2 hr/gen | ~2-4 min/gen | ~50-100x |
| 15,000 | ~7.2 hr/gen | ~5-10 min/gen | ~50-90x |

At 10K neurons with 50 generations:
- CPU: ~160 hours (infeasible)
- GPU (fused): ~2.5 hours (feasible overnight)

## Thesis Value

This benchmark provides valuable data points:
1. **Naive GPU ≠ fast GPU** — kernel launch overhead dominates for small networks
2. **CPU JIT (Numba) is optimal for <5K neurons** — our entire Phase 1-3 regime
3. **GPU migration requires architectural changes** — not a drop-in replacement
4. **Sparse representation is more important than GPU** — 4000x memory reduction

## Decision

- **Phase 1-3:** Continue with CPU Numba (optimal)
- **Phase 4 (5K+ neurons):** Implement fused CUDA kernels + sparse weights
- **Phase 5+ (10K+):** GPU mandatory, sparse + fused + batch parallelism

GPU simulator code saved at `src/simulator/gpu_simulator.py` as foundation.
