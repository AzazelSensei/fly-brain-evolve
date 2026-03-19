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
from PIL import Image
from numba import njit, prange

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

dt_val = config["simulation"]["dt"]
num_steps = int(0.1 / dt_val)
refr_steps = int(0.002 / dt_val)

data = np.load("data/mnist_cache.npz", allow_pickle=True)
all_images = data["data"]
all_labels = data["target"]


def prepare_mnist_8x8(n_train=200, n_test=50, seed=42):
    rng = np.random.default_rng(seed)
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for digit in range(10):
        mask = all_labels == str(digit)
        digit_imgs = all_images[mask]
        chosen = rng.choice(len(digit_imgs), size=n_train + n_test, replace=False)

        for i in chosen[:n_train]:
            img28 = digit_imgs[i].reshape(28, 28)
            img8 = np.array(Image.fromarray(img28.astype(np.uint8)).resize((8, 8), Image.BILINEAR))
            train_imgs.append(img8.flatten() / 255.0)
            train_labels.append(digit)

        for i in chosen[n_train:n_train + n_test]:
            img28 = digit_imgs[i].reshape(28, 28)
            img8 = np.array(Image.fromarray(img28.astype(np.uint8)).resize((8, 8), Image.BILINEAR))
            test_imgs.append(img8.flatten() / 255.0)
            test_labels.append(digit)

    return np.array(train_imgs), np.array(train_labels), np.array(test_imgs), np.array(test_labels)


from src.simulator.growing_brain import BrainConfig, GrowingGenome, simulate_growing_batch


def evaluate_mnist(population, test_imgs, test_labels, brain_config, seed=42):
    rng = np.random.default_rng(seed)
    n = brain_config.num_neurons
    tau_m, V_rest, V_reset, g_L = brain_config.build_params()

    n_test = min(len(test_imgs), 100)
    test_indices = rng.choice(len(test_imgs), size=n_test, replace=False)

    batch_size = len(population) * n_test
    batch_W_exc = np.zeros((batch_size, n, n))
    batch_W_inh = np.zeros((batch_size, n, n))
    batch_V_thresh = np.zeros((batch_size, n))
    batch_input = np.zeros((batch_size, num_steps, brain_config.num_pn), dtype=np.bool_)
    batch_labels = np.zeros(batch_size, dtype=np.int32)

    idx = 0
    for g_idx, genome in enumerate(population):
        W_exc, W_inh = genome.build_weight_matrices()
        V_thresh = genome.build_threshold_vector()

        for t_idx in test_indices:
            batch_W_exc[idx] = W_exc
            batch_W_inh[idx] = W_inh
            batch_V_thresh[idx] = V_thresh

            rates = test_imgs[t_idx] * 100.0
            for pn in range(min(len(rates), brain_config.num_pn)):
                batch_input[idx, :, pn] = rng.random(num_steps) < (rates[pn] * dt_val)
            batch_labels[idx] = test_labels[t_idx]
            idx += 1

    all_mbon, all_kc = simulate_growing_batch(
        batch_W_exc, batch_W_inh, batch_input, batch_V_thresh,
        V_rest, V_reset, g_L, 0.0, -0.080,
        tau_m, 0.005, 0.010,
        dt_val, num_steps, refr_steps,
        n, brain_config.mbon_start, brain_config.mbon_end,
        brain_config.kc_start, brain_config.kc_end,
        50e-9, brain_config.num_pn,
    )

    fitnesses = np.zeros(len(population))
    for g_idx in range(len(population)):
        start = g_idx * n_test
        end = start + n_test
        correct = 0
        for t in range(start, end):
            counts = all_mbon[t]
            if counts.sum() > 0:
                pred = np.argmax(counts)
            else:
                pred = rng.integers(brain_config.num_mbon)
            if pred == batch_labels[t]:
                correct += 1
        accuracy = correct / n_test
        kc_sparsity = np.mean([np.sum(all_kc[t]) for t in range(start, end)]) / brain_config.num_kc
        sparsity_score = max(0, 1.0 - abs(kc_sparsity - 0.1) / 0.1) * 0.1
        complexity = np.count_nonzero(population[g_idx].pn_kc) / 10000 * 0.01
        fitnesses[g_idx] = accuracy + sparsity_score - complexity

    return fitnesses


def mutate(g, rate, rng):
    c = g.copy()
    if rng.random() < rate:
        c.kc_mbon = np.clip(c.kc_mbon + rng.normal(0, 0.5, c.kc_mbon.shape), 0, 15)
    if rng.random() < rate:
        mask = c.pn_kc > 0
        c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
        c.pn_kc[~mask] = 0
    if rng.random() < rate * 0.5:
        idx = rng.choice(c.kc_thresh.shape[0], size=min(20, c.kc_thresh.shape[0]), replace=False)
        c.kc_thresh[idx] += rng.normal(0, 1, size=len(idx))
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    return c


def crossover(a, b, rng):
    mask = rng.random(a.config.num_kc) < 0.5
    return GrowingGenome(
        a.config,
        np.where(mask[np.newaxis, :], a.pn_kc, b.pn_kc).copy(),
        np.where(mask[:, np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(mask, a.kc_thresh, b.kc_thresh).copy(),
    )


def evolve(brain_config, test_imgs, test_labels, pop_size=30, generations=50, seed=42):
    rng = np.random.default_rng(seed)
    population = [GrowingGenome.random(brain_config, np.random.default_rng(rng.integers(1e9)))
                  for _ in range(pop_size)]

    history = []

    _ = evaluate_mnist(population[:2], test_imgs[:10], test_labels[:10], brain_config, seed=0)

    for gen in range(generations):
        t0 = time.time()
        fitnesses = evaluate_mnist(population, test_imgs, test_labels, brain_config, seed=rng.integers(1e9))

        best_idx = np.argmax(fitnesses)
        h = {"gen": gen, "best": round(float(fitnesses[best_idx]), 4),
             "mean": round(float(fitnesses.mean()), 4), "time_s": round(time.time()-t0, 2)}
        history.append(h)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"  Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} time={h['time_s']:.1f}s")

        sorted_idx = np.argsort(fitnesses)[::-1]
        new_pop = [population[i].copy() for i in sorted_idx[:3]]
        while len(new_pop) < pop_size:
            ti = rng.choice(len(population), size=3, replace=False)
            pa = population[ti[np.argmax(fitnesses[ti])]]
            if rng.random() < 0.4:
                ti2 = rng.choice(len(population), size=3, replace=False)
                pb = population[ti2[np.argmax(fitnesses[ti2])]]
                child = crossover(pa, pb, rng)
            else:
                child = pa.copy()
            new_pop.append(mutate(child, 0.3, rng))
        population = new_pop

    return population, history


train_imgs, train_labels, test_imgs, test_labels = prepare_mnist_8x8(n_train=100, n_test=50)
print(f"MNIST 8x8: {len(train_imgs)} train, {len(test_imgs)} test, 10 classes")

configs_to_test = [
    ("267 neuron (200 KC, 10 MBON)", BrainConfig(num_pn=64, num_kc=200, num_mbon=10, num_apl=1)),
    ("475 neuron (400 KC, 10 MBON)", BrainConfig(num_pn=64, num_kc=400, num_mbon=10, num_apl=1)),
    ("1075 neuron (1000 KC, 10 MBON)", BrainConfig(num_pn=64, num_kc=1000, num_mbon=10, num_apl=1)),
]

all_results = {}
for name, brain_cfg in configs_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    pop, hist = evolve(brain_cfg, test_imgs, test_labels, pop_size=30, generations=50, seed=42)
    elapsed = time.time() - t0
    all_results[name] = {"history": hist, "time_s": round(elapsed, 1)}
    print(f"Done in {elapsed:.0f}s, best={hist[-1]['best']:.4f}")

with open("docs/journal/mnist_challenge_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MNIST 10-Class Challenge: Can a Fly Brain Recognize Handwritten Digits?",
             fontsize=13, fontweight="bold")

colors = ["#E74C3C", "#3498DB", "#2ECC71"]
for i, (name, res) in enumerate(all_results.items()):
    hist = res["history"]
    gens = [h["gen"] for h in hist]
    axes[0].plot(gens, [h["best"] for h in hist], "-", color=colors[i], lw=2, label=name)
    axes[1].plot(gens, [h["mean"] for h in hist], "-", color=colors[i], lw=1.5, label=name)

axes[0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance (10%)")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness (accuracy + bonus)")
axes[0].set_title("Best Fitness"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].axhline(y=0.1, color="gray", ls="--", alpha=0.5)
axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Mean Fitness")
axes[1].set_title("Population Mean"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/mnist_challenge.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nFigure saved to docs/figures/mnist_challenge.png")
