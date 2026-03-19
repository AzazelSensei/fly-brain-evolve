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

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

dt_val = config["simulation"]["dt"]
num_steps = int(0.1 / dt_val)
refr_steps = int(0.002 / dt_val)

data = np.load("data/mnist_cache.npz", allow_pickle=True)
all_images = data["data"]
all_labels = data["target"]

from src.simulator.growing_brain import BrainConfig, GrowingGenome, simulate_growing_batch


def prepare_data(n_per_class=30, seed=42):
    rng = np.random.default_rng(seed)
    imgs, labs = [], []
    for d in range(10):
        mask = all_labels == str(d)
        digit_imgs = all_images[mask]
        chosen = rng.choice(len(digit_imgs), size=n_per_class, replace=False)
        for i in chosen:
            img8 = np.array(Image.fromarray(digit_imgs[i].reshape(28,28).astype(np.uint8)).resize((8,8), Image.BILINEAR))
            imgs.append(img8.flatten() / 255.0)
            labs.append(d)
    return np.array(imgs), np.array(labs)


def evaluate_batch(population, imgs, labs, brain_cfg, n_test=50, seed=42):
    rng = np.random.default_rng(seed)
    n = brain_cfg.num_neurons
    tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()
    test_idx = rng.choice(len(imgs), size=n_test, replace=False)

    batch_size = len(population) * n_test
    bW = np.zeros((batch_size, n, n))
    bI = np.zeros((batch_size, n, n))
    bT = np.zeros((batch_size, n))
    bS = np.zeros((batch_size, num_steps, brain_cfg.num_pn), dtype=np.bool_)
    bL = np.zeros(batch_size, dtype=np.int32)

    idx = 0
    for gi, g in enumerate(population):
        We, Wi = g.build_weight_matrices()
        Vt = g.build_threshold_vector()
        for ti in test_idx:
            bW[idx] = We; bI[idx] = Wi; bT[idx] = Vt
            rates = imgs[ti] * 100.0
            for pn in range(min(len(rates), brain_cfg.num_pn)):
                bS[idx, :, pn] = rng.random(num_steps) < (rates[pn] * dt_val)
            bL[idx] = labs[ti]
            idx += 1

    mc, kc = simulate_growing_batch(
        bW, bI, bS, bT, V_rest, V_reset, g_L, 0.0, -0.080,
        tau_m, 0.005, 0.010, dt_val, num_steps, refr_steps,
        n, brain_cfg.mbon_start, brain_cfg.mbon_end,
        brain_cfg.kc_start, brain_cfg.kc_end, 50e-9, brain_cfg.num_pn,
    )

    fitnesses = np.zeros(len(population))
    for gi in range(len(population)):
        s, e = gi*n_test, (gi+1)*n_test
        ok = sum(1 for t in range(s,e) if (np.argmax(mc[t]) if mc[t].sum()>0 else rng.integers(10)) == bL[t])
        acc = ok / n_test
        sp = np.mean([np.sum(kc[t]) for t in range(s,e)]) / brain_cfg.num_kc
        fitnesses[gi] = acc + max(0, 1-abs(sp-0.1)/0.1)*0.1 - np.count_nonzero(population[gi].pn_kc)/10000*0.01
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
        idx = rng.choice(c.kc_thresh.shape[0], size=20, replace=False)
        c.kc_thresh[idx] += rng.normal(0, 1, size=20)
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    return c


def crossover(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    return GrowingGenome(a.config,
        np.where(m[np.newaxis,:], a.pn_kc, b.pn_kc).copy(),
        np.where(m[:,np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(m, a.kc_thresh, b.kc_thresh).copy())


imgs, labs = prepare_data(n_per_class=30)
print(f"Data: {len(imgs)} images, 10 classes, 8x8 pixels")

brain_cfg = BrainConfig(num_pn=64, num_kc=200, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons (64 PN, 200 KC, 10 MBON, 1 APL)")

rng = np.random.default_rng(42)
POP = 30
GENS = 60
pop = [GrowingGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]

print(f"\nJIT warmup...")
_ = evaluate_batch(pop[:2], imgs[:20], labs[:20], brain_cfg, n_test=10, seed=0)
print("Done. Starting evolution...\n")

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = evaluate_batch(pop, imgs, labs, brain_cfg, n_test=60, seed=rng.integers(1e9))
    bi = np.argmax(fit)
    h = {"gen":gen, "best":round(float(fit[bi]),4), "mean":round(float(fit.mean()),4), "time":round(time.time()-t0,2)}
    history.append(h)

    if gen % 5 == 0 or gen == GENS-1:
        elapsed = time.time() - t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s) [{elapsed:.0f}s total]")

    si = np.argsort(fit)[::-1]
    new = [pop[i].copy() for i in si[:3]]
    while len(new) < POP:
        ti = rng.choice(POP, size=3, replace=False)
        pa = pop[ti[np.argmax(fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(POP, size=3, replace=False)
            ch = crossover(pa, pop[ti2[np.argmax(fit[ti2])]], rng)
        else:
            ch = pa.copy()
        new.append(mutate(ch, 0.3, rng))
    pop = new

total = time.time() - t0_total
best_acc = max(h["best"] for h in history)
chance = 0.10

print(f"\n{'='*50}")
print(f"MNIST 10-CLASS RESULT")
print(f"Brain: {brain_cfg.num_neurons} neurons")
print(f"Best fitness: {best_acc:.4f}")
print(f"Chance level: {chance:.2f} (10%)")
print(f"Total time: {total:.0f}s")
print(f"{'='*50}")

with open("docs/journal/mnist_small_results.json", "w") as f:
    json.dump({"history":history, "best":best_acc, "neurons":brain_cfg.num_neurons, "time_s":round(total,1)}, f, indent=2)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(f"MNIST 10-Class: {brain_cfg.num_neurons} Neuron Fly Brain", fontsize=14, fontweight="bold")
gens = [h["gen"] for h in history]
ax.plot(gens, [h["best"] for h in history], "o-", color="#E74C3C", lw=2, label="Best", markersize=4)
ax.plot(gens, [h["mean"] for h in history], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=3)
ax.axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance (10%)")
ax.axhline(y=0.5, color="green", ls="--", alpha=0.5, label="50% accuracy")
ax.set_xlabel("Generation", fontsize=12)
ax.set_ylabel("Fitness (accuracy + sparsity bonus)", fontsize=12)
ax.set_ylim(0, max(best_acc+0.1, 0.6))
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("docs/figures/mnist_small_result.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
