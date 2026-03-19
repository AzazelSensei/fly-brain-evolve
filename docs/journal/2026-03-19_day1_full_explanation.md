# Research Journal — Day 1: Complete Project Explanation (Non-Technical Summary)

**Date:** 2026-03-19
**Purpose:** Full project explanation for documentation and paper introduction draft

---

## What Did We Do? The Big Picture

We simulated a fruit fly's brain circuit on a computer, then "evolved" that brain so it could learn to recognize visual patterns.

The circuit we modeled is the **Mushroom Body** — a real structure in the *Drosophila melanogaster* (fruit fly) brain. In nature, this structure helps flies learn to associate smells with rewards or punishments. We repurposed it for a visual task: distinguishing between horizontal and vertical stripe patterns.

---

## The Architecture: How the Simulated Brain Works

The mushroom body has four main components, each modeled as groups of spiking neurons:

```
Eyes (Input)  →  Projection Neurons (PN)  →  Kenyon Cells (KC)  →  Output Neurons (MBON)
"What do I see?"   "Relay the signal"        "Create sparse code"    "Make a decision"
(64 pixels)        (64 neurons)              (200 neurons)            (2 neurons)
```

Additionally, there is an **APL neuron** — a single global inhibitor that prevents too many Kenyon Cells from firing at once. This enforces "sparse coding," where only ~6-12% of KCs respond to any given input. This is biologically accurate: real fly brains use sparse coding to create unique, non-overlapping representations for different stimuli.

### How a Single Neuron Works

Each neuron follows the **Leaky Integrate-and-Fire (LIF)** model:
- The neuron has a membrane voltage that starts at rest (-70 mV)
- When other neurons send signals, the voltage rises
- If the voltage crosses a threshold (e.g., -45 mV for KCs), the neuron "fires" — it produces a spike
- After firing, the voltage resets to -70 mV and the neuron is briefly inactive (refractory period: 2 ms)

This is governed by the differential equation:
```
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m
```
Where `I_syn` is the total synaptic current from connected neurons, `g_L` is the leak conductance, and `tau_m` is the membrane time constant.

### How Neurons Communicate

Neurons are connected by **synapses** with specific weights. When neuron A fires, it increases the excitatory conductance (`g_exc`) of neuron B by an amount equal to the synaptic weight. The effect decays exponentially with time constant `tau_exc = 5 ms`.

For inhibitory connections (APL → KC), the same mechanism applies but through `g_inh`, which pulls the voltage down instead of up.

---

## The Task: Binary Pattern Classification

We presented the brain with 8×8 pixel images:

```
Pattern A (Label = 0):          Pattern B (Label = 1):
■ ■ ■ ■ ■ ■ ■ ■               ■ □ ■ □ ■ □ ■ □
□ □ □ □ □ □ □ □               ■ □ ■ □ ■ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■               ■ □ ■ □ ■ □ ■ □
□ □ □ □ □ □ □ □               ■ □ ■ □ ■ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■               ■ □ ■ □ ■ □ ■ □
□ □ □ □ □ □ □ □               ■ □ ■ □ ■ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■               ■ □ ■ □ ■ □ ■ □
□ □ □ □ □ □ □ □               ■ □ ■ □ ■ □ ■ □
(Horizontal stripes)            (Vertical stripes)
```

Each pixel maps to one Projection Neuron. White pixels cause their PN to fire at 100 Hz (Poisson-distributed spikes), black pixels remain silent (0 Hz). The stimulus is presented for 100 ms.

The brain's answer is determined by which MBON fires more:
- If MBON 0 > MBON 1 → predict "Horizontal" (class 0)
- If MBON 1 > MBON 0 → predict "Vertical" (class 1)

---

## What Is "Evolution" in This Context?

We used **neuroevolution** — applying Darwin's natural selection to artificial neural networks. This is different from backpropagation (used in deep learning). Here's how it works:

### Step 1: Create a Random Population
We created 50 "brains" (genomes), each with:
- Random PN→KC connectivity (which input neurons connect to which coding neurons)
- Random KC→MBON weights (how strongly coding neurons drive output neurons)
- Random KC thresholds (how easily each coding neuron fires)

### Step 2: Evaluate Fitness
Each brain is tested: show it both patterns multiple times, count correct classifications.

**Fitness = accuracy + sparsity_bonus - complexity_penalty**

- **accuracy** (0 to 1): fraction of correct answers. 1.0 = 100% correct, 0.5 = coin flip
- **sparsity_bonus** (0 to 0.1): reward for maintaining ~10% KC activation (biologically realistic sparse coding)
- **complexity_penalty** (~0.01): small penalty for having too many synapses (encourages efficient solutions)

So a perfect brain with good sparsity scores about 1.0 + 0.1 - 0.01 = **1.09**.

### Step 3: Selection, Crossover, Mutation

**Tournament selection**: pick 3 random individuals, the best one becomes a parent.

**Crossover**: two parents create a child. Each Kenyon Cell (and all its connections) comes from either parent A or parent B with 50% probability. This is "KC-based uniform crossover."

**Mutation operators** (applied with 30% probability each):
1. **Weight perturbation**: small Gaussian noise added to KC→MBON weights
2. **PN-KC weight perturbation**: adjust input connection strengths
3. **Threshold mutation**: change how easily specific KCs fire
4. **Add synapse**: create a new PN→KC connection

**Elitism**: the 5 best brains are copied directly to the next generation (never lost).

### Step 4: Repeat for 100 Generations

Each generation: evaluate → select → crossover → mutate → new generation.

---

## Reading the Graphs

### Graph 1: "Fitness Over Generations" (top-left)

```
Y-axis: Fitness score (0.0 to ~1.1)
X-axis: Generation number (0 to 100)

Red line (Best):    The single best individual in each generation
Blue line (Mean):   Average fitness across all 50 individuals
Light blue band:    Standard deviation (spread of the population)
Gray dashed line:   0.5 = chance level (random guessing)
Green dashed line:  0.8 = our target (80% accuracy)
```

**How to read it:**
- Generation 0: Mean is ~0.6 (most brains are bad), Best is ~1.0 (one lucky brain)
- Generation 10: Mean crosses 1.0 (most brains are now excellent)
- Generation 50+: Mean stabilizes at ~1.09 (entire population is near-perfect)
- The blue band narrows over time = population converges (less diversity)

**Interpretation:** Evolution found excellent solutions within 10 generations. The remaining 90 generations refined the population so that nearly every individual performs optimally.

### Graph 2: "Time Per Generation" (top-right)

```
Y-axis: Seconds per generation (0.6 to 1.0)
X-axis: Generation number
```

Each generation takes ~0.7-0.9 seconds. This is after our 68x speedup from replacing Brian2 with a Numba JIT simulator. The original Brian2 version took ~178 seconds per generation.

### Graph 3: "Best KC-MBON Weights" (bottom-left)

```
Y-axis: Number of synapses at this weight
X-axis: Synaptic weight in nanoSiemens (nS)
```

This histogram shows the distribution of KC→MBON synaptic weights in the best evolved brain. Evolution shaped this distribution:
- Some synapses are weak (~2-4 nS): these KCs don't strongly influence the output
- Many synapses are strong (~6-10 nS): these are the "important" connections
- The bimodal distribution suggests the brain learned which KCs matter for classification

### Graph 4: "Best KC Thresholds" (bottom-right)

```
Y-axis: Number of KC neurons at this threshold
X-axis: Firing threshold in millivolts (mV)
```

Each KC neuron has its own threshold (how much input it needs to fire). Evolution tuned these:
- Mean: -45.0 mV (range: -50 to -40 mV)
- Higher threshold (closer to -40) = more selective = fires only for very strong input
- Lower threshold (closer to -50) = less selective = fires more easily

The bell-shaped distribution centered at -45 mV means most KCs are moderately selective, with some very selective (high threshold) and some sensitive (low threshold) neurons.

---

## The Evolved Brain in Action

When the evolved brain receives input:

### Horizontal stripes presented:
1. 32 out of 64 PNs fire (corresponding to white pixel rows)
2. These PNs activate specific KCs through sparse random connections
3. Only ~18 out of 200 KCs fire (9% — sparse coding maintained!)
4. These 18 KCs drive MBON 0 more strongly: MBON 0 produces 6 spikes, MBON 1 produces 3
5. argmax([6, 3]) = 0 → **"Horizontal!" ✓**

### Vertical stripes presented:
1. Different 32 PNs fire (corresponding to white pixel columns)
2. These activate a **different set** of KCs (only 3 overlap with the horizontal set!)
3. ~19 out of 200 KCs fire (9.5%)
4. These 19 KCs drive MBON 1 more strongly: MBON 0 produces 1 spike, MBON 1 produces 3
5. argmax([1, 3]) = 1 → **"Vertical!" ✓**

**Classification accuracy: ~99% over hundreds of trials.**

---

## Why This Matters

### Scientific Significance
1. **Biological plausibility**: Our model uses the same architecture as the real fly mushroom body — PN→KC sparse coding → MBON readout with APL inhibition
2. **Sparse coding works**: With only 6-12% of KCs active, patterns get non-overlapping representations — exactly as observed in real flies (Honegger et al., 2011)
3. **Evolution over STDP**: Pure spike-timing dependent plasticity (biological learning rule) failed because MBON neurons never fired. Evolution optimized the weight landscape to make the circuit functional. This suggests that in real flies, the circuit topology and parameter ranges were shaped by millions of years of evolution, with STDP providing fine-tuning within a pre-evolved parameter regime.

### Technical Contributions
1. **Custom Numba JIT simulator**: 582x faster than Brian2 for small networks, enabling rapid experimentation
2. **Systematic parameter tuning**: Documented the path from 100% KC activation (broken sparse coding) to 6% (biologically realistic)
3. **Unit bug discovery**: The Brian2 unit system (mV vs volt) error was a non-obvious trap that produced plausible-looking but completely wrong results

---

## Complete Error Log and Corrections

| # | Error | When Found | Root Cause | Fix | Impact |
|---|-------|-----------|------------|-----|--------|
| E1 | 100% KC activation | Baseline simulation | Brian2 `*mV` instead of `*volt` | Changed all namespace conversions | All prior simulations invalid |
| E2 | APL inhibition ineffective | APL sweep | Unit bug made thresholds ~0.02 mV apart | Fixed by E1 resolution | Masked by E1 |
| E3 | STDP weights unchanged | STDP baseline | MBON never fires → no post-synaptic trace | Proceeded to evolution | Expected limitation |
| E4 | SpikeGenerator dt collision | Fast simulation tests | Poisson spikes generated at continuous times, not dt-aligned | Added offset and dt-aligned generation | Brian2 refused to run |
| E5 | Brian2 too slow (89 min) | Evolution run | Per-simulation object construction overhead | Built Numba JIT batched simulator | 68x speedup |

---

## File Inventory (Day 1)

### Code
| File | Purpose |
|------|---------|
| `src/connectome/loader.py` | Generate synthetic mushroom body connectome |
| `src/connectome/builder.py` | Build Brian2 spiking neural network |
| `src/neurons/models.py` | LIF neuron equations and parameters |
| `src/neurons/plasticity.py` | STDP learning rule |
| `src/encoding/spike_encoder.py` | Image → Poisson spike train conversion |
| `src/evolution/genome.py` | Genome data structure |
| `src/evolution/mutations.py` | Mutation operators |
| `src/evolution/crossover.py` | Crossover operator |
| `src/evolution/fitness.py` | Brian2-based fitness evaluation |
| `src/evolution/population.py` | Evolution loop |
| `src/simulator/fast_lif.py` | Numba JIT batched LIF simulator (582x faster) |
| `src/simulator/fitness_fast.py` | Fast batched population fitness evaluator |
| `src/visualization/*.py` | All visualization scripts |

### Figures (docs/figures/)
| Figure | Description |
|--------|-------------|
| `mushroom_body_architecture.png` | 4-panel connectome overview |
| `brain_detail.png` | Dark-themed neural circuit with statistics |
| `simulation_pipeline.png` | Full evolution pipeline diagram |
| `input_patterns.png` | Input stimuli and rate encoding |
| `spike_raster_comparison.png` | Poisson spike trains for both patterns |
| `sim_corrected_chain.png` | First working simulation after unit fix |
| `brain_live_activity.png` | Pre-evolution brain activity (sparse coding demo) |
| `sparsity_tuning.png` | Parameter sweep for sparsity optimization |
| `sparsity_sweep.png` | APL inhibition strength sweep |
| `threshold_sweep.png` | KC threshold sweep (showed unit bug) |
| `baseline_stdp_results.png` | STDP learning attempt (failed) |
| `stdp_v2_results.png` | STDP with higher weights (still failed) |
| `evolution_results.png` | Brian2 evolution results (30 gen, 89 min) |
| `evolved_brain_live.png` | Evolved brain activity comparison |
| `evolution_fast_results.png` | Fast evolution results (100 gen, 79 sec) |

### Journal Entries (docs/journal/)
| Entry | Topic |
|-------|-------|
| `2026-03-19_day1_project_setup.md` | Architecture decisions and validation |
| `2026-03-19_day1_baseline_simulation.md` | First simulation results and KC sparsity problem |
| `2026-03-19_day1_unit_bug_fix.md` | Critical Brian2 unit conversion bug |
| `2026-03-19_day1_stdp_baseline.md` | STDP learning failure analysis |
| `2026-03-19_day1_evolution_start.md` | Evolution experiment launch |
| `2026-03-19_day1_evolution_complete.md` | Evolution results and findings |
| `2026-03-19_day1_fast_simulator.md` | Numba JIT speedup achievement |
| `2026-03-19_day1_full_explanation.md` | This document |
