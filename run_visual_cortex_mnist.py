import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import json, time
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
all_images, all_labels = data["data"], data["target"]

from src.simulator.growing_brain import BrainConfig, GrowingGenome, simulate_growing_batch
from src.simulator.visual_cortex import VisualCortex, EvolvedVisualCortex

rng = np.random.default_rng(42)


def prepare(size, n_per_class=30):
    imgs, labs = [], []
    for d in range(10):
        mask = all_labels == str(d)
        di = all_images[mask]
        ch = rng.choice(len(di), size=n_per_class, replace=False)
        for i in ch:
            img = np.array(Image.fromarray(di[i].reshape(28, 28).astype(np.uint8)).resize((size, size), Image.BILINEAR))
            imgs.append(img.flatten() / 255.0)
            labs.append(d)
    return np.array(imgs), np.array(labs)


class VisualBrainGenome:
    def __init__(self, cortex_filters, pn_kc, kc_mbon, kc_thresh):
        self.cortex_filters = [f.copy() for f in cortex_filters]
        self.pn_kc = pn_kc
        self.kc_mbon = kc_mbon
        self.kc_thresh = kc_thresh

    @staticmethod
    def random(num_pn, num_kc, num_mbon, filter_size=3, num_filters=8, rng_seed=42):
        r = np.random.default_rng(rng_seed)
        filters = []
        for _ in range(num_filters):
            f = r.normal(0, 0.5, (filter_size, filter_size))
            f -= f.mean()
            norm = np.linalg.norm(f)
            if norm > 0:
                f /= norm
            filters.append(f)

        pn_kc = np.zeros((num_pn, num_kc))
        for kc in range(num_kc):
            chosen = r.choice(num_pn, size=min(6, num_pn), replace=False)
            pn_kc[chosen, kc] = r.uniform(1.0, 5.0, size=len(chosen))

        kc_mbon = r.uniform(2.0, 10.0, size=(num_kc, num_mbon))
        kc_thresh = r.uniform(-48, -42, size=num_kc)
        return VisualBrainGenome(filters, pn_kc, kc_mbon, kc_thresh)

    def copy(self):
        return VisualBrainGenome(
            self.cortex_filters, self.pn_kc.copy(),
            self.kc_mbon.copy(), self.kc_thresh.copy())

    def to_growing_genome(self, brain_cfg):
        return GrowingGenome(brain_cfg, self.pn_kc, self.kc_mbon, self.kc_thresh)


def mutate(g, rate, r):
    c = g.copy()
    if r.random() < rate:
        fi = r.integers(len(c.cortex_filters))
        c.cortex_filters[fi] = c.cortex_filters[fi] + r.normal(0, 0.1, c.cortex_filters[fi].shape)
        c.cortex_filters[fi] -= c.cortex_filters[fi].mean()
        norm = np.linalg.norm(c.cortex_filters[fi])
        if norm > 0:
            c.cortex_filters[fi] /= norm
    if r.random() < rate:
        c.kc_mbon = np.clip(c.kc_mbon + r.normal(0, 0.5, c.kc_mbon.shape), 0, 15)
    if r.random() < rate:
        mask = c.pn_kc > 0
        c.pn_kc = np.clip(c.pn_kc + r.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
        c.pn_kc[~mask] = 0
    if r.random() < rate * 0.5:
        idx = r.choice(len(c.kc_thresh), size=20, replace=False)
        c.kc_thresh[idx] += r.normal(0, 1, size=20)
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    if r.random() < rate * 0.2:
        fi = r.integers(len(c.cortex_filters))
        size = c.cortex_filters[fi].shape[0]
        new_f = r.normal(0, 0.5, (size, size))
        new_f -= new_f.mean()
        norm = np.linalg.norm(new_f)
        if norm > 0:
            new_f /= norm
        c.cortex_filters[fi] = new_f
    return c


def crossover(a, b, r):
    n_f = len(a.cortex_filters)
    f_mask = r.random(n_f) < 0.5
    filters = [a.cortex_filters[i].copy() if f_mask[i] else b.cortex_filters[i].copy() for i in range(n_f)]
    kc_mask = r.random(a.pn_kc.shape[1]) < 0.5
    return VisualBrainGenome(
        filters,
        np.where(kc_mask[np.newaxis, :], a.pn_kc, b.pn_kc).copy(),
        np.where(kc_mask[:, np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(kc_mask, a.kc_thresh, b.kc_thresh).copy())


def evaluate(pop, imgs, labs, img_size, brain_cfg, n_test=60, seed=42):
    r = np.random.default_rng(seed)
    n = brain_cfg.num_neurons
    tau_m, Vr, Vrs, gL = brain_cfg.build_params()
    ti = r.choice(len(imgs), size=n_test, replace=False)

    bs = len(pop) * n_test
    bW = np.zeros((bs, n, n)); bI = np.zeros((bs, n, n))
    bT = np.zeros((bs, n)); bS = np.zeros((bs, num_steps, brain_cfg.num_pn), dtype=np.bool_)
    bL = np.zeros(bs, dtype=np.int32)

    idx = 0
    for gi, genome in enumerate(pop):
        cortex = VisualCortex(img_size, stride=2, evolved_filters=genome.cortex_filters)
        gg = genome.to_growing_genome(brain_cfg)
        We, Wi = gg.build_weight_matrices()
        Vt = gg.build_threshold_vector()

        for t in ti:
            features = cortex.process(imgs[t])
            bW[idx] = We; bI[idx] = Wi; bT[idx] = Vt
            rates = features * 100.0
            for pn in range(min(len(rates), brain_cfg.num_pn)):
                bS[idx, :, pn] = r.random(num_steps) < (rates[pn] * dt_val)
            bL[idx] = labs[t]; idx += 1

    mc, kc = simulate_growing_batch(
        bW, bI, bS, bT, Vr, Vrs, gL, 0.0, -0.080,
        tau_m, 0.005, 0.010, dt_val, num_steps, refr_steps,
        n, brain_cfg.mbon_start, brain_cfg.mbon_end,
        brain_cfg.kc_start, brain_cfg.kc_end, 50e-9, brain_cfg.num_pn)

    fit = np.zeros(len(pop))
    for gi in range(len(pop)):
        s, e = gi * n_test, (gi + 1) * n_test
        ok = sum(1 for t in range(s, e)
                 if (np.argmax(mc[t]) if mc[t].sum() > 0 else r.integers(10)) == bL[t])
        acc = ok / n_test
        sp = np.mean([np.sum(kc[t]) for t in range(s, e)]) / brain_cfg.num_kc
        fit[gi] = acc + max(0, 1 - abs(sp - 0.1) / 0.1) * 0.1 - np.count_nonzero(pop[gi].pn_kc) / 50000 * 0.01
    return fit


imgs16, labs16 = prepare(16, n_per_class=30)
print(f"Data: {len(imgs16)} images, 10 classes, 16x16")

cortex_test = VisualCortex(16, stride=2)
num_features = cortex_test.output_size
print(f"Gabor bank: {cortex_test.num_filters} filters, output={num_features} features")

brain_cfg = BrainConfig(num_pn=num_features, num_kc=300, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons ({num_features} PN, 300 KC, 10 MBON)")

POP = 30
GENS = 60

pop = [VisualBrainGenome.random(num_features, 300, 10, filter_size=3, num_filters=8,
                                 rng_seed=rng.integers(1e9)) for _ in range(POP)]

print("\nJIT warmup...")
_ = evaluate(pop[:2], imgs16[:20], labs16[:20], 16, brain_cfg, n_test=10, seed=0)
print("Done.\n")

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = evaluate(pop, imgs16, labs16, 16, brain_cfg, n_test=60, seed=rng.integers(1e9))
    bi = np.argmax(fit)
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2)}
    history.append(h)

    if gen % 5 == 0 or gen == GENS - 1:
        el = time.time() - t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s) [{el:.0f}s]")

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
best = max(h["best"] for h in history)

print(f"\n{'='*50}")
print(f"VISUAL CORTEX + MUSHROOM BODY RESULT")
print(f"Neurons: {brain_cfg.num_neurons}")
print(f"Best fitness: {best:.4f}")
print(f"Time: {total:.0f}s")
print(f"{'='*50}")

with open("docs/journal/visual_cortex_results.json", "w") as f:
    json.dump({"history": history, "best": best, "neurons": brain_cfg.num_neurons,
               "time_s": round(total, 1), "num_features": num_features}, f, indent=2)

best_genome = pop[np.argmax(fit)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Visual Cortex + Mushroom Body: MNIST 10-Class",
             fontsize=14, fontweight="bold")

gens = [h["gen"] for h in history]
axes[0, 0].plot(gens, [h["best"] for h in history], "o-", color="#E74C3C", lw=2, label="Best", markersize=4)
axes[0, 0].plot(gens, [h["mean"] for h in history], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=3)
axes[0, 0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance 10%")
axes[0, 0].axhline(y=0.5, color="green", ls="--", alpha=0.5, label="50%")
axes[0, 0].set_xlabel("Generation"); axes[0, 0].set_ylabel("Fitness")
axes[0, 0].set_title("Fitness Over Generations"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

n_filters = len(best_genome.cortex_filters)
rows = 2; cols = (n_filters + 1) // 2
for i in range(n_filters):
    ax = axes[0, 1].inset_axes([
        (i % cols) / cols, 1 - (i // cols + 1) / rows,
        0.9 / cols, 0.9 / rows
    ])
    ax.imshow(best_genome.cortex_filters[i], cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 1].axis("off")
axes[0, 1].set_title("Evolved Visual Filters (3x3)", fontsize=11)

w = best_genome.kc_mbon.flatten()
axes[1, 0].hist(w, bins=30, color="#45B7D1", alpha=0.7, edgecolor="white")
axes[1, 0].set_xlabel("Weight (nS)"); axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title(f"KC-MBON Weights (mean={w.mean():.1f})"); axes[1, 0].grid(True, alpha=0.3)

comparison = {
    "8x8 raw (275n)": 0.234,
    "16x16 raw (467n)": 0.237,
    "16x16 raw (767n)": 0.268,
    f"16x16+cortex ({brain_cfg.num_neurons}n)": best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors = ["#E74C3C", "#E74C3C", "#E74C3C", "#2ECC71"]
axes[1, 1].barh(names, vals, color=colors, alpha=0.7, edgecolor="black")
axes[1, 1].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1, 1].set_xlabel("Best Fitness")
axes[1, 1].set_title("Comparison: Raw Pixels vs Visual Cortex")
for i, v in enumerate(vals):
    axes[1, 1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
fig.savefig("docs/figures/visual_cortex_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
