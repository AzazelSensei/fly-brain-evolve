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

patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
labels = [0, 1]


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

    @property
    def num_synapses(self):
        return int(np.count_nonzero(self.pn_kc))


def mutate(g, rate, rng):
    c = g.copy()
    if rng.random() < rate:
        noise = rng.normal(0, 0.3, size=c.kc_mbon.shape)
        c.kc_mbon = np.clip(c.kc_mbon + noise, 0.0, 15.0)
    if rng.random() < rate:
        mask = c.pn_kc > 0
        noise = rng.normal(0, 0.2, size=c.pn_kc.shape) * mask
        c.pn_kc = np.clip(c.pn_kc + noise, 0.0, 8.0)
        c.pn_kc[~mask] = 0
    if rng.random() < rate * 0.5:
        idx = rng.choice(200, size=20, replace=False)
        c.kc_thresh[idx] += rng.normal(0, 1, size=20)
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    if rng.random() < rate * 0.3:
        kc_idx = rng.integers(200)
        zero_pns = np.where(c.pn_kc[:, kc_idx] == 0)[0]
        if len(zero_pns) > 0:
            c.pn_kc[rng.choice(zero_pns), kc_idx] = rng.uniform(1, 4)
    return c


def crossover(a, b, rng):
    mask = rng.random(200) < 0.5
    return EvoGenome(
        np.where(mask[np.newaxis, :], a.pn_kc, b.pn_kc).copy(),
        np.where(mask[:, np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(mask, a.kc_thresh, b.kc_thresh).copy(),
    )


POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.4
ELITISM = 5
TOURNAMENT_K = 3
N_TRIALS = 10

rng = np.random.default_rng(42)
population = [EvoGenome.random(np.random.default_rng(rng.integers(1e9))) for _ in range(POP_SIZE)]

history = []
best_ever_fitness = -1
best_ever_genome = None

t_total_start = time.time()

fitnesses_warmup = evaluate_population(population[:2], patterns, labels, config, n_trials=6, seed=0)

for gen in range(GENERATIONS):
    t0 = time.time()

    fitnesses = evaluate_population(population, patterns, labels, config, n_trials=N_TRIALS, seed=rng.integers(1e9))

    best_idx = np.argmax(fitnesses)
    if fitnesses[best_idx] > best_ever_fitness:
        best_ever_fitness = fitnesses[best_idx]
        best_ever_genome = population[best_idx].copy()

    gen_time = time.time() - t0
    h = {
        "gen": gen, "best": round(float(fitnesses[best_idx]), 4),
        "mean": round(float(fitnesses.mean()), 4), "std": round(float(fitnesses.std()), 4),
        "worst": round(float(fitnesses.min()), 4), "time_s": round(gen_time, 2),
    }
    history.append(h)

    if gen % 10 == 0 or gen == GENERATIONS - 1:
        elapsed = time.time() - t_total_start
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} time={gen_time:.2f}s total={elapsed:.0f}s")

    sorted_idx = np.argsort(fitnesses)[::-1]
    new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]

    while len(new_pop) < POP_SIZE:
        ti = rng.choice(len(population), size=TOURNAMENT_K, replace=False)
        parent_a = population[ti[np.argmax(fitnesses[ti])]]
        if rng.random() < CROSSOVER_RATE:
            ti2 = rng.choice(len(population), size=TOURNAMENT_K, replace=False)
            parent_b = population[ti2[np.argmax(fitnesses[ti2])]]
            child = crossover(parent_a, parent_b, rng)
        else:
            child = parent_a.copy()
        child = mutate(child, MUTATION_RATE, rng)
        new_pop.append(child)

    population = new_pop

total_time = time.time() - t_total_start

final_fit = evaluate_population(population, patterns, labels, config, n_trials=20, seed=0)
best_final_idx = np.argmax(final_fit)
best_final = population[best_final_idx]

result = {
    "best_fitness": round(float(final_fit[best_final_idx]), 4),
    "best_ever_fitness": round(best_ever_fitness, 4),
    "generations": GENERATIONS,
    "population_size": POP_SIZE,
    "trials_per_eval": N_TRIALS,
    "total_time_s": round(total_time, 1),
    "avg_gen_time_s": round(total_time / GENERATIONS, 2),
    "speedup_vs_brian2": round(5332 / max(total_time, 1), 1),
}

with open("docs/journal/evolution_fast_result.json", "w") as f:
    json.dump(result, f, indent=2)
with open("docs/journal/evolution_fast_log.json", "w") as f:
    json.dump(history, f, indent=2)
np.savez("docs/journal/best_genome_fast.npz",
         pn_kc=best_final.pn_kc, kc_mbon=best_final.kc_mbon, kc_thresh=best_final.kc_thresh)

print(f"\nFinal: best={result['best_fitness']:.4f} in {result['total_time_s']}s ({result['speedup_vs_brian2']}x faster than Brian2)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Fast Evolution: {GENERATIONS} gen, {POP_SIZE} pop, {total_time:.0f}s total", fontsize=14, fontweight="bold")

gens = [h["gen"] for h in history]
axes[0,0].plot(gens, [h["best"] for h in history], "-", color="#FF6B6B", lw=2, label="Best")
axes[0,0].plot(gens, [h["mean"] for h in history], "-", color="#45B7D1", lw=1.5, label="Mean")
mn = np.array([h["mean"] for h in history])
sd = np.array([h["std"] for h in history])
axes[0,0].fill_between(gens, mn-sd, mn+sd, alpha=0.2, color="#45B7D1")
axes[0,0].axhline(y=0.5, color="gray", ls="--", alpha=0.5)
axes[0,0].axhline(y=0.8, color="green", ls="--", alpha=0.5, label="Target")
axes[0,0].set_xlabel("Generation"); axes[0,0].set_ylabel("Fitness")
axes[0,0].set_title("Fitness Over Generations"); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(gens, [h["time_s"] for h in history], ".-", color="#F39C12", lw=1)
axes[0,1].set_xlabel("Generation"); axes[0,1].set_ylabel("Time (s)")
axes[0,1].set_title("Time Per Generation"); axes[0,1].grid(True, alpha=0.3)

w = best_final.kc_mbon.flatten()
axes[1,0].hist(w, bins=30, color="#45B7D1", alpha=0.7, edgecolor="white")
axes[1,0].set_xlabel("Weight (nS)"); axes[1,0].set_ylabel("Count")
axes[1,0].set_title(f"Best KC-MBON Weights (mean={w.mean():.1f})"); axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(best_final.kc_thresh, bins=20, color="#FF6B6B", alpha=0.7, edgecolor="white")
axes[1,1].set_xlabel("Threshold (mV)"); axes[1,1].set_ylabel("Count")
axes[1,1].set_title(f"Best KC Thresholds (mean={best_final.kc_thresh.mean():.1f})"); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/evolution_fast_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
