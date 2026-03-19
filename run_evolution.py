import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import json
import time
import numpy as np
from brian2 import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

prefs.codegen.target = "numpy"

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)
from src.neurons.models import get_neuron_params

dt_val = config["simulation"]["dt"]
patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
labels = [0, 1]

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

def make_ns(p):
    return {"tau_m":p["tau_m"]*second,"V_rest":p["V_rest"]*volt,"V_thresh":p["V_thresh"]*volt,
            "V_reset":p["V_reset"]*volt,"tau_exc":p["tau_exc"]*second,"tau_inh":p["tau_inh"]*second,
            "g_L":p["g_L"]*siemens,"E_exc":p["E_exc"]*volt,"E_inh":p["E_inh"]*volt}


class EvoGenome:
    def __init__(self, pn_kc, kc_mbon, kc_thresh, seed=42):
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
        nonz_pns = np.where(c.pn_kc[:, kc_idx] > 0)[0]
        if len(zero_pns) > 0 and len(nonz_pns) > 0:
            add_pn = rng.choice(zero_pns)
            c.pn_kc[add_pn, kc_idx] = rng.uniform(1, 4)

    return c


def crossover(a, b, rng):
    mask = rng.random(200) < 0.5
    pn_kc = np.where(mask[np.newaxis, :], a.pn_kc, b.pn_kc)
    kc_mbon = np.where(mask[:, np.newaxis], a.kc_mbon, b.kc_mbon)
    kc_thresh = np.where(mask, a.kc_thresh, b.kc_thresh)
    return EvoGenome(pn_kc.copy(), kc_mbon.copy(), kc_thresh.copy())


def evaluate(genome, rng, n_trials=6):
    correct = 0
    total = 0
    kc_actives = []

    for pi in range(len(patterns)):
        for tr in range(n_trials // len(patterns)):
            start_scope()
            ns_pn = make_ns(get_neuron_params("pn", config))
            ns_mb = make_ns(get_neuron_params("mbon", config))
            ns_ap = make_ns(get_neuron_params("apl", config))

            pn = NeuronGroup(64, eqs, threshold="v>V_thresh", reset="v=V_reset",
                             refractory=2*ms, method="euler", namespace=ns_pn, name="pn")
            pn.v = -70*mV

            kc_ns = make_ns(get_neuron_params("kc", config))
            kc_ns["V_thresh"] = float(genome.kc_thresh.mean()) * mV
            kc_g = NeuronGroup(200, eqs, threshold="v>V_thresh", reset="v=V_reset",
                               refractory=2*ms, method="euler", namespace=kc_ns, name="kc")
            kc_g.v = -70*mV

            mb = NeuronGroup(2, eqs, threshold="v>V_thresh", reset="v=V_reset",
                             refractory=2*ms, method="euler", namespace=ns_mb, name="mb")
            mb.v = -70*mV

            ap = NeuronGroup(1, eqs, threshold="v>V_thresh", reset="v=V_reset",
                             refractory=2*ms, method="euler", namespace=ns_ap, name="ap")
            ap.v = -70*mV

            pidx, kidx = np.nonzero(genome.pn_kc)
            pk = Synapses(pn, kc_g, "ws:1", on_pre="g_exc_post+=ws*nS", name="pk")
            pk.connect(i=pidx.astype(int), j=kidx.astype(int))
            pk.ws = genome.pn_kc[pidx, kidx]

            km_syn = Synapses(kc_g, mb, "w:1", on_pre="g_exc_post+=w*nS", name="km")
            km_syn.connect()
            km_syn.w = genome.kc_mbon.flatten()

            ka = Synapses(kc_g, ap, on_pre="g_exc_post+=2*nS", name="ka")
            ka.connect()
            ak = Synapses(ap, kc_g, on_pre="g_inh_post+=200*nS", name="ak")
            ak.connect()

            rates = image_to_rates(patterns[pi], max_rate=100.0)
            fr = np.zeros(64); fr[:len(rates)] = rates
            idx, t = generate_poisson_spike_indices_and_times(fr, 0.1, dt=dt_val, seed=rng.integers(1e6))
            inp = SpikeGeneratorGroup(64, idx, t*second, name="inp")
            isyn = Synapses(inp, pn, on_pre="g_exc_post+=50*nS", name="is")
            isyn.connect("i==j")

            kmon = SpikeMonitor(kc_g, name="kmon")
            mmon = SpikeMonitor(mb, name="mmon")

            defaultclock.dt = dt_val * second
            net = Network(pn, kc_g, mb, ap, pk, km_syn, ka, ak, inp, isyn, kmon, mmon)
            net.run(0.1 * second)

            c = np.array(mmon.count)
            kc_a = len(set(np.array(kmon.i))) if kmon.num_spikes > 0 else 0
            kc_actives.append(kc_a)

            pred = int(np.argmax(c)) if c.sum() > 0 else rng.integers(2)
            if pred == labels[pi]:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    mean_kc_active = np.mean(kc_actives) if kc_actives else 0
    sparsity = mean_kc_active / 200
    sparsity_score = max(0, 1.0 - abs(sparsity - 0.1) / 0.1) * 0.1
    complexity = np.count_nonzero(genome.pn_kc) / 10000 * 0.01

    return accuracy + sparsity_score - complexity


POP_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.4
ELITISM = 3

rng = np.random.default_rng(42)
population = [EvoGenome.random(np.random.default_rng(rng.integers(1e9))) for _ in range(POP_SIZE)]

history = []
best_ever_fitness = -1
best_ever_genome = None

t_start = time.time()

for gen in range(GENERATIONS):
    gen_start = time.time()

    fitnesses = []
    for genome in population:
        f = evaluate(genome, np.random.default_rng(rng.integers(1e9)))
        fitnesses.append(f)
    fitnesses = np.array(fitnesses)

    best_idx = np.argmax(fitnesses)
    gen_time = time.time() - gen_start

    if fitnesses[best_idx] > best_ever_fitness:
        best_ever_fitness = fitnesses[best_idx]
        best_ever_genome = population[best_idx].copy()

    h = {
        "gen": gen, "best": round(float(fitnesses[best_idx]), 4),
        "mean": round(float(fitnesses.mean()), 4), "std": round(float(fitnesses.std()), 4),
        "worst": round(float(fitnesses.min()), 4), "time_s": round(gen_time, 1),
    }
    history.append(h)

    elapsed = time.time() - t_start
    with open("docs/journal/evolution_log.json", "w") as f:
        json.dump(history, f, indent=2)

    sorted_idx = np.argsort(fitnesses)[::-1]
    new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]

    while len(new_pop) < POP_SIZE:
        ti = rng.choice(len(population), size=3, replace=False)
        parent_a = population[ti[np.argmax(fitnesses[ti])]]

        if rng.random() < CROSSOVER_RATE:
            ti2 = rng.choice(len(population), size=3, replace=False)
            parent_b = population[ti2[np.argmax(fitnesses[ti2])]]
            child = crossover(parent_a, parent_b, rng)
        else:
            child = parent_a.copy()

        child = mutate(child, MUTATION_RATE, rng)
        new_pop.append(child)

    population = new_pop

final_fit = [evaluate(g, np.random.default_rng(rng.integers(1e9)), n_trials=10) for g in population]
best_final = population[np.argmax(final_fit)]

result = {
    "best_fitness": round(float(max(final_fit)), 4),
    "best_ever_fitness": round(best_ever_fitness, 4),
    "generations": GENERATIONS,
    "population_size": POP_SIZE,
    "total_time_s": round(time.time() - t_start, 1),
}
with open("docs/journal/evolution_result.json", "w") as f:
    json.dump(result, f, indent=2)

np.savez("docs/journal/best_genome.npz",
         pn_kc=best_final.pn_kc, kc_mbon=best_final.kc_mbon, kc_thresh=best_final.kc_thresh)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Neuroevolution Results - Binary Classification", fontsize=14, fontweight="bold")

gens = [h["gen"] for h in history]
axes[0].plot(gens, [h["best"] for h in history], "o-", color="#FF6B6B", lw=2, label="Best")
axes[0].plot(gens, [h["mean"] for h in history], "s-", color="#45B7D1", lw=1.5, label="Mean")
mn = np.array([h["mean"] for h in history])
sd = np.array([h["std"] for h in history])
axes[0].fill_between(gens, mn-sd, mn+sd, alpha=0.2, color="#45B7D1")
axes[0].axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
axes[0].axhline(y=0.8, color="green", ls="--", alpha=0.5, label="Target")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].set_title("Fitness Over Generations"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

w = best_final.kc_mbon.flatten()
axes[1].hist(w, bins=30, color="#45B7D1", alpha=0.7, edgecolor="white")
axes[1].set_xlabel("Weight (nS)"); axes[1].set_ylabel("Count")
axes[1].set_title(f"Best Genome KC-MBON Weights\nmean={w.mean():.2f}, std={w.std():.2f}")
axes[1].grid(True, alpha=0.3)

axes[2].hist(best_final.kc_thresh, bins=20, color="#FF6B6B", alpha=0.7, edgecolor="white")
axes[2].set_xlabel("Threshold (mV)"); axes[2].set_ylabel("Count")
axes[2].set_title(f"Best Genome KC Thresholds\nmean={best_final.kc_thresh.mean():.1f}")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/evolution_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
