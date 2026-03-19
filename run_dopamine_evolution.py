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

from src.encoding.spike_encoder import make_horizontal_stripes, make_vertical_stripes, image_to_rates
from src.simulator.fast_lif import (
    simulate_batch, build_neuron_params, build_weight_matrix,
    build_threshold_vector, NUM_NEURONS,
)
from src.simulator.dopamine_lif import (
    train_with_dopamine_batch, build_neuron_params as build_np_dop,
)

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


def evaluate_with_dopamine(population, training_epochs=5, test_trials=5, seed=42):
    rng = np.random.default_rng(seed)
    tau_m, V_rest, V_reset, g_L = build_np_dop()

    results = []
    for genome in population:
        W_exc, W_inh = build_weight_matrix(genome)
        V_thresh = build_threshold_vector(genome)
        kc_mbon_w = genome.kc_mbon.copy()

        for epoch in range(training_epochs):
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
                    V_rest, V_reset, g_L,
                    0.0, -0.080,
                    tau_m, 0.005, 0.010,
                    dt_val, num_steps, refr_steps,
                    0.020, 0.05,
                )
                kc_mbon_w = new_w[0]

        correct = 0
        total = 0
        kc_actives = []
        for p_idx in range(len(patterns)):
            for trial in range(test_trials):
                pat = patterns[p_idx].flatten()
                rates = np.clip(pat, 0, 1) * 100.0
                spikes = np.zeros((1, num_steps, 64), dtype=np.bool_)
                for pn in range(min(len(rates), 64)):
                    spikes[0, :, pn] = rng.random(num_steps) < (rates[pn] * dt_val)

                mc, ks = simulate_batch(
                    W_exc.reshape(1, NUM_NEURONS, NUM_NEURONS),
                    W_inh.reshape(1, NUM_NEURONS, NUM_NEURONS),
                    spikes,
                    V_thresh.reshape(1, NUM_NEURONS),
                    V_rest, V_reset, g_L,
                    0.0, -0.080,
                    tau_m, 0.005, 0.010,
                    dt_val, num_steps, refr_steps,
                )

                kc_active = np.sum(ks[0])
                kc_actives.append(kc_active)
                pred = np.argmax(mc[0]) if mc[0].sum() > 0 else rng.integers(2)
                if pred == labels[p_idx]:
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)
        sparsity = np.mean(kc_actives) / 200
        sparsity_score = max(0, 1.0 - abs(sparsity - 0.1) / 0.1) * 0.1
        complexity = np.count_nonzero(genome.pn_kc) / 10000 * 0.01

        genome.kc_mbon = kc_mbon_w
        results.append(accuracy + sparsity_score - complexity)

    return np.array(results)


POP_SIZE = 30
GENERATIONS = 50
rng = np.random.default_rng(42)
population = [EvoGenome.random(np.random.default_rng(rng.integers(1e9))) for _ in range(POP_SIZE)]

history = []
t_start = time.time()

_ = evaluate_with_dopamine(population[:1], training_epochs=1, test_trials=1, seed=0)

for gen in range(GENERATIONS):
    t0 = time.time()
    fitnesses = evaluate_with_dopamine(population, training_epochs=5, test_trials=5, seed=rng.integers(1e9))

    best_idx = np.argmax(fitnesses)
    h = {"gen": gen, "best": round(float(fitnesses[best_idx]), 4),
         "mean": round(float(fitnesses.mean()), 4), "std": round(float(fitnesses.std()), 4),
         "time_s": round(time.time()-t0, 2)}
    history.append(h)

    if gen % 5 == 0 or gen == GENERATIONS-1:
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} time={h['time_s']:.1f}s")

    sorted_idx = np.argsort(fitnesses)[::-1]
    new_pop = [population[i].copy() for i in sorted_idx[:3]]
    while len(new_pop) < POP_SIZE:
        ti = rng.choice(len(population), size=3, replace=False)
        pa = population[ti[np.argmax(fitnesses[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(len(population), size=3, replace=False)
            pb = population[ti2[np.argmax(fitnesses[ti2])]]
            child = crossover(pa, pb, rng)
        else:
            child = pa.copy()
        child = mutate(child, 0.3, rng)
        new_pop.append(child)
    population = new_pop

total_time = time.time() - t_start
result = {"best": round(float(max(h["best"] for h in history)), 4),
          "generations": GENERATIONS, "total_time_s": round(total_time, 1)}

with open("docs/journal/dopamine_evolution_log.json", "w") as f:
    json.dump(history, f, indent=2)
with open("docs/journal/dopamine_evolution_result.json", "w") as f:
    json.dump(result, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 2: Evolution + Dopamine-Modulated STDP", fontsize=14, fontweight="bold")

gens = [h["gen"] for h in history]
axes[0].plot(gens, [h["best"] for h in history], "o-", color="#E74C3C", lw=2, label="Best", markersize=4)
axes[0].plot(gens, [h["mean"] for h in history], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=3)
mn = np.array([h["mean"] for h in history])
sd = np.array([h["std"] for h in history])
axes[0].fill_between(gens, mn-sd, mn+sd, alpha=0.2, color="#3498DB")
axes[0].axhline(y=0.5, color="gray", ls="--", alpha=0.5)
axes[0].axhline(y=0.8, color="green", ls="--", alpha=0.5, label="Target")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].set_title("Fitness: Evolution + Dopamine Learning"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(gens, [h["time_s"] for h in history], ".-", color="#F39C12")
axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Time (s)")
axes[1].set_title("Time Per Generation"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/dopamine_evolution_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nDone: {result['best']:.4f} in {result['total_time_s']:.0f}s")
