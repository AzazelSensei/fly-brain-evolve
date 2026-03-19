import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import time
import json
import numpy as np
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

from src.encoding.spike_encoder import make_horizontal_stripes, make_vertical_stripes

patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
labels = [0, 1]


class SimpleGenome:
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
        return SimpleGenome(pn_kc, kc_mbon, kc_thresh)


rng = np.random.default_rng(42)
pop_size = 20
population = [SimpleGenome.random(np.random.default_rng(rng.integers(1e9))) for _ in range(pop_size)]

from src.simulator.fitness_fast import evaluate_population

t0 = time.time()
fitnesses = evaluate_population(population, patterns, labels, config, n_trials=6, seed=42)
t_first = time.time() - t0

t0 = time.time()
fitnesses2 = evaluate_population(population, patterns, labels, config, n_trials=6, seed=99)
t_second = time.time() - t0

t0 = time.time()
for _ in range(5):
    evaluate_population(population, patterns, labels, config, n_trials=6, seed=rng.integers(1e6))
t_avg = (time.time() - t0) / 5

result = {
    "first_call_s": round(t_first, 2),
    "second_call_s": round(t_second, 2),
    "avg_call_s": round(t_avg, 2),
    "pop_size": pop_size,
    "trials_per_genome": 6,
    "total_sims": pop_size * 6,
    "per_sim_ms": round(t_avg / (pop_size * 6) * 1000, 1),
    "best_fitness": round(float(fitnesses.max()), 4),
    "mean_fitness": round(float(fitnesses.mean()), 4),
    "brian2_baseline_s": 120.0,
    "speedup_vs_brian2": round(120.0 / max(t_avg, 0.01), 1),
}

with open("docs/journal/benchmark_fast_results.json", "w") as f:
    json.dump(result, f, indent=2)

for k, v in result.items():
    print(f"  {k}: {v}")
