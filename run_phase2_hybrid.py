import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import json
import time
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

from src.encoding.spike_encoder import make_horizontal_stripes, make_vertical_stripes
from src.simulator.fitness_fast import evaluate_population
from src.simulator.fast_lif import (
    simulate_batch, build_neuron_params, build_weight_matrix,
    build_threshold_vector, NUM_NEURONS,
)
from src.simulator.dopamine_lif import train_with_dopamine_batch

patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
labels = [0, 1]
dt_val = config["simulation"]["dt"]
num_steps = int(0.1 / dt_val)
refr_steps = int(0.002 / dt_val)


class EvoGenome:
    def __init__(self, pn_kc, kc_mbon, kc_thresh):
        self.pn_kc = pn_kc
        self.kc_mbon = kc_mbon
        self.kc_thresh = kc_thresh

    @staticmethod
    def random(rng):
        pn_kc = np.zeros((64, 200))
        for kc in range(200):
            chosen = rng.choice(64, size=6, replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=6)
        kc_mbon = rng.uniform(2.0, 10.0, size=(200, 2))
        kc_thresh = rng.uniform(-48, -42, size=200)
        return EvoGenome(pn_kc, kc_mbon, kc_thresh)

    def copy(self):
        return EvoGenome(self.pn_kc.copy(), self.kc_mbon.copy(), self.kc_thresh.copy())


def mutate(g, rate, rng):
    c = g.copy()
    if rng.random() < rate:
        c.kc_mbon = np.clip(c.kc_mbon + rng.normal(0, 0.3, c.kc_mbon.shape), 0, 15)
    if rng.random() < rate:
        mask = c.pn_kc > 0
        c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.2, c.pn_kc.shape) * mask, 0, 8)
        c.pn_kc[~mask] = 0
    if rng.random() < rate * 0.5:
        idx = rng.choice(200, size=20, replace=False)
        c.kc_thresh[idx] += rng.normal(0, 1, size=20)
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    return c


def crossover(a, b, rng):
    mask = rng.random(200) < 0.5
    return EvoGenome(
        np.where(mask[np.newaxis, :], a.pn_kc, b.pn_kc).copy(),
        np.where(mask[:, np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(mask, a.kc_thresh, b.kc_thresh).copy(),
    )


def evolve_step(population, fitnesses, rng, elitism=5, tournament_k=3, mutation_rate=0.3, crossover_rate=0.4):
    sorted_idx = np.argsort(fitnesses)[::-1]
    new_pop = [population[i].copy() for i in sorted_idx[:elitism]]
    while len(new_pop) < len(population):
        ti = rng.choice(len(population), size=tournament_k, replace=False)
        pa = population[ti[np.argmax(fitnesses[ti])]]
        if rng.random() < crossover_rate:
            ti2 = rng.choice(len(population), size=tournament_k, replace=False)
            pb = population[ti2[np.argmax(fitnesses[ti2])]]
            child = crossover(pa, pb, rng)
        else:
            child = pa.copy()
        new_pop.append(mutate(child, mutation_rate, rng))
    return new_pop


def dopamine_train(genome, n_epochs, lr, rng):
    tau_m, V_rest, V_reset, g_L = build_neuron_params()
    W_exc, W_inh = build_weight_matrix(genome)
    V_thresh = build_threshold_vector(genome)
    kc_mbon_w = genome.kc_mbon.copy()

    for epoch in range(n_epochs):
        for p_idx in rng.permutation(len(patterns)):
            pat = patterns[p_idx].flatten()
            rates = np.clip(pat, 0, 1) * 100.0
            spikes = np.zeros((1, num_steps, 64), dtype=np.bool_)
            for pn in range(min(len(rates), 64)):
                spikes[0, :, pn] = rng.random(num_steps) < (rates[pn] * dt_val)

            mc, ks, new_w = train_with_dopamine_batch(
                W_exc.reshape(1, NUM_NEURONS, NUM_NEURONS),
                W_inh.reshape(1, NUM_NEURONS, NUM_NEURONS),
                kc_mbon_w.reshape(1, 200, 2),
                spikes,
                V_thresh.reshape(1, NUM_NEURONS),
                np.array([labels[p_idx]], dtype=np.int32),
                V_rest, V_reset, g_L, 0.0, -0.080,
                tau_m, 0.005, 0.010,
                dt_val, num_steps, refr_steps,
                0.020, lr,
            )
            kc_mbon_w = new_w[0]
    return kc_mbon_w


rng = np.random.default_rng(42)
POP_SIZE = 50
history = []
t_start = time.time()

population = [EvoGenome.random(np.random.default_rng(rng.integers(1e9))) for _ in range(POP_SIZE)]

print("=" * 70)
print("STAGE 1: Pure Evolution (structure optimization)")
print("=" * 70)
_ = evaluate_population(population[:2], patterns, labels, config, n_trials=6, seed=0)

for gen in range(50):
    t0 = time.time()
    fitnesses = evaluate_population(population, patterns, labels, config, n_trials=10, seed=rng.integers(1e9))
    best_idx = np.argmax(fitnesses)
    h = {"stage": 1, "gen": gen, "best": round(float(fitnesses[best_idx]), 4),
         "mean": round(float(fitnesses.mean()), 4), "std": round(float(fitnesses.std()), 4),
         "time_s": round(time.time()-t0, 2)}
    history.append(h)
    if gen % 10 == 0:
        print(f"  Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f}")
    population = evolve_step(population, fitnesses, rng)

print(f"\nStage 1 complete: best={history[-1]['best']:.4f} mean={history[-1]['mean']:.4f}")

print("\n" + "=" * 70)
print("STAGE 2: Dopamine Fine-Tuning (online learning)")
print("=" * 70)

best_5 = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)[:5]

lr_schedule = [0.01, 0.02, 0.05, 0.1]
epoch_schedule = [2, 5, 10, 20]

dopamine_results = []
for lr in lr_schedule:
    for n_ep in epoch_schedule:
        test_genome = population[best_5[0]].copy()
        pre_w = test_genome.kc_mbon.copy()
        new_w = dopamine_train(test_genome, n_ep, lr, np.random.default_rng(42))

        test_genome.kc_mbon = new_w
        fit = evaluate_population([test_genome], patterns, labels, config, n_trials=20, seed=42)
        w_change = np.abs(new_w - pre_w).mean()
        dopamine_results.append({
            "lr": lr, "epochs": n_ep, "fitness": round(float(fit[0]), 4),
            "w_change": round(float(w_change), 4),
        })

print("\nDopamine learning rate sweep:")
print(f"{'LR':>6} {'Epochs':>6} {'Fitness':>8} {'W_change':>10}")
for r in dopamine_results:
    marker = " <-- " if r["fitness"] > 1.05 else ""
    print(f"{r['lr']:6.3f} {r['epochs']:6d} {r['fitness']:8.4f} {r['w_change']:10.4f}{marker}")

best_dopamine = max(dopamine_results, key=lambda r: r["fitness"])
print(f"\nBest dopamine config: lr={best_dopamine['lr']}, epochs={best_dopamine['epochs']}, fitness={best_dopamine['fitness']}")

print("\n" + "=" * 70)
print("STAGE 3: Combined Evolution + Dopamine")
print("=" * 70)

best_lr = best_dopamine["lr"]
best_epochs = best_dopamine["epochs"]

for gen in range(50, 80):
    t0 = time.time()

    for genome in population:
        new_w = dopamine_train(genome, best_epochs, best_lr, np.random.default_rng(rng.integers(1e9)))
        genome.kc_mbon = new_w

    fitnesses = evaluate_population(population, patterns, labels, config, n_trials=10, seed=rng.integers(1e9))
    best_idx = np.argmax(fitnesses)
    h = {"stage": 3, "gen": gen, "best": round(float(fitnesses[best_idx]), 4),
         "mean": round(float(fitnesses.mean()), 4), "std": round(float(fitnesses.std()), 4),
         "time_s": round(time.time()-t0, 2)}
    history.append(h)
    if gen % 5 == 0:
        print(f"  Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} time={h['time_s']:.1f}s")
    population = evolve_step(population, fitnesses, rng)

total_time = time.time() - t_start
print(f"\nTotal time: {total_time:.0f}s")

with open("docs/journal/phase2_hybrid_log.json", "w") as f:
    json.dump(history, f, indent=2)
with open("docs/journal/phase2_dopamine_sweep.json", "w") as f:
    json.dump(dopamine_results, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Phase 2: Three-Stage Hybrid (Evolution → Dopamine Tuning → Combined)",
             fontsize=14, fontweight="bold")

s1 = [h for h in history if h["stage"] == 1]
s3 = [h for h in history if h["stage"] == 3]

axes[0, 0].plot([h["gen"] for h in s1], [h["best"] for h in s1], "o-", color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0, 0].plot([h["gen"] for h in s1], [h["mean"] for h in s1], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0, 0].axhline(y=0.8, color="green", ls="--", alpha=0.5)
axes[0, 0].set_title("Stage 1: Pure Evolution"); axes[0, 0].set_ylabel("Fitness")
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

lrs = sorted(set(r["lr"] for r in dopamine_results))
for lr in lrs:
    subset = [r for r in dopamine_results if r["lr"] == lr]
    axes[0, 1].plot([r["epochs"] for r in subset], [r["fitness"] for r in subset],
                   "o-", label=f"lr={lr}", markersize=6)
axes[0, 1].axhline(y=1.05, color="green", ls="--", alpha=0.5)
axes[0, 1].set_xlabel("Training Epochs"); axes[0, 1].set_ylabel("Fitness")
axes[0, 1].set_title("Stage 2: Dopamine LR Sweep"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

if s3:
    axes[1, 0].plot([h["gen"] for h in s3], [h["best"] for h in s3], "o-", color="#E74C3C", lw=2, label="Best", markersize=3)
    axes[1, 0].plot([h["gen"] for h in s3], [h["mean"] for h in s3], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[1, 0].axhline(y=0.8, color="green", ls="--", alpha=0.5)
axes[1, 0].set_title("Stage 3: Evolution + Dopamine"); axes[1, 0].set_xlabel("Generation")
axes[1, 0].set_ylabel("Fitness"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

all_gens = [h["gen"] for h in history]
all_best = [h["best"] for h in history]
all_mean = [h["mean"] for h in history]
axes[1, 1].plot(all_gens, all_best, "-", color="#E74C3C", lw=2, label="Best")
axes[1, 1].plot(all_gens, all_mean, "-", color="#3498DB", lw=1.5, label="Mean")
axes[1, 1].axvline(x=50, color="gray", ls="--", alpha=0.5)
axes[1, 1].text(25, 0.3, "Stage 1\n(Evolution)", ha="center", fontsize=9, color="gray")
axes[1, 1].text(65, 0.3, "Stage 3\n(Evo+Dopa)", ha="center", fontsize=9, color="gray")
axes[1, 1].set_xlabel("Generation"); axes[1, 1].set_ylabel("Fitness")
axes[1, 1].set_title("Full Timeline"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/phase2_hybrid_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
