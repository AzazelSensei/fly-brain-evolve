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

rng = np.random.default_rng(42)


def compute_hog_simple(image_flat, img_size=16, cell_size=4, n_bins=8):
    img = image_flat.reshape(img_size, img_size)

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) % np.pi

    n_cells_y = img_size // cell_size
    n_cells_x = img_size // cell_size
    histograms = np.zeros((n_cells_y, n_cells_x, n_bins))

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_mag = magnitude[y0:y1, x0:x1]
            cell_ang = angle[y0:y1, x0:x1]

            for py in range(cell_size):
                for px in range(cell_size):
                    bin_idx = int(cell_ang[py, px] / np.pi * n_bins) % n_bins
                    histograms[cy, cx, bin_idx] += cell_mag[py, px]

    features = histograms.flatten()
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def prepare_hog(n_per_class=30, img_size=16):
    imgs, feats, labs = [], [], []
    for d in range(10):
        mask = all_labels == str(d)
        di = all_images[mask]
        ch = rng.choice(len(di), size=n_per_class, replace=False)
        for i in ch:
            img = np.array(Image.fromarray(di[i].reshape(28, 28).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR))
            img_norm = img.flatten() / 255.0
            hog = compute_hog_simple(img_norm, img_size)
            imgs.append(img_norm)
            feats.append(hog)
            labs.append(d)
    return np.array(imgs), np.array(feats), np.array(labs)


imgs, feats, labs = prepare_hog(n_per_class=30, img_size=16)
n_features = feats.shape[1]
print(f"HOG features: {n_features} per image (16x16, 4x4 cells, 8 orientation bins)")

fig_hog, axes_hog = plt.subplots(2, 5, figsize=(15, 6))
fig_hog.suptitle("HOG Features Per Digit — What the Brain Would See", fontsize=14, fontweight="bold")
for d in range(10):
    mask = labs == d
    mean_feat = feats[mask].mean(axis=0)
    r, c = d // 5, d % 5
    axes_hog[r, c].bar(range(n_features), mean_feat, color="#45B7D1", alpha=0.7, width=1)
    axes_hog[r, c].set_title(f"Digit {d}", fontsize=10)
    axes_hog[r, c].set_ylim(0, mean_feat.max() * 1.3)
    axes_hog[r, c].set_xticks([])
plt.tight_layout()
fig_hog.savefig("docs/figures/hog_features_per_digit.png", dpi=200, bbox_inches="tight")
plt.close(fig_hog)
print("HOG visualization saved")

hog_sim = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        a = feats[labs == i].mean(axis=0)
        b = feats[labs == j].mean(axis=0)
        hog_sim[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("\nHOG digit similarity (most confusable):")
for i in range(10):
    for j in range(i+1, 10):
        if hog_sim[i, j] > 0.85:
            print(f"  {i} vs {j}: {hog_sim[i,j]:.3f}")

print("\nHOG most distinguishable:")
for i in range(10):
    for j in range(i+1, 10):
        if hog_sim[i, j] < 0.70:
            print(f"  {i} vs {j}: {hog_sim[i,j]:.3f}")


def evaluate(pop, features, labs, cfg, n_test=60, seed=42):
    r = np.random.default_rng(seed)
    n = cfg.num_neurons
    tau_m, Vr, Vrs, gL = cfg.build_params()
    ti = r.choice(len(features), size=n_test, replace=False)

    bs = len(pop) * n_test
    bW = np.zeros((bs, n, n)); bI = np.zeros((bs, n, n))
    bT = np.zeros((bs, n)); bS = np.zeros((bs, num_steps, cfg.num_pn), dtype=np.bool_)
    bL = np.zeros(bs, dtype=np.int32)

    idx = 0
    for g in pop:
        We, Wi = g.build_weight_matrices()
        Vt = g.build_threshold_vector()
        for t in ti:
            bW[idx] = We; bI[idx] = Wi; bT[idx] = Vt
            rates = features[t] * 100.0
            for pn in range(min(len(rates), cfg.num_pn)):
                bS[idx, :, pn] = r.random(num_steps) < (rates[pn] * dt_val)
            bL[idx] = labs[t]; idx += 1

    mc, kc = simulate_growing_batch(
        bW, bI, bS, bT, Vr, Vrs, gL, 0.0, -0.080,
        tau_m, 0.005, 0.010, dt_val, num_steps, refr_steps,
        n, cfg.mbon_start, cfg.mbon_end, cfg.kc_start, cfg.kc_end, 50e-9, cfg.num_pn)

    fit = np.zeros(len(pop))
    for gi in range(len(pop)):
        s, e = gi * n_test, (gi + 1) * n_test
        ok = sum(1 for t in range(s, e) if (np.argmax(mc[t]) if mc[t].sum() > 0 else r.integers(10)) == bL[t])
        acc = ok / n_test
        sp = np.mean([np.sum(kc[t]) for t in range(s, e)]) / cfg.num_kc
        fit[gi] = acc + max(0, 1 - abs(sp - 0.1) / 0.1) * 0.1 - np.count_nonzero(pop[gi].pn_kc) / 50000 * 0.01
    return fit


def mutate(g, r):
    c = g.copy()
    c.kc_mbon = np.clip(c.kc_mbon + r.normal(0, 0.5, c.kc_mbon.shape), 0, 15)
    mask = c.pn_kc > 0
    c.pn_kc = np.clip(c.pn_kc + r.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
    c.pn_kc[~mask] = 0
    idx = r.choice(len(c.kc_thresh), size=20, replace=False)
    c.kc_thresh[idx] += r.normal(0, 1, size=20)
    c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    return c


def crossover(a, b, r):
    m = r.random(a.config.num_kc) < 0.5
    return GrowingGenome(a.config,
        np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy(),
        np.where(m[:, np.newaxis], a.kc_mbon, b.kc_mbon).copy(),
        np.where(m, a.kc_thresh, b.kc_thresh).copy())


brain_cfg = BrainConfig(num_pn=n_features, num_kc=300, num_mbon=10, num_apl=1)
print(f"\nBrain: {brain_cfg.num_neurons} neurons ({n_features} PN, 300 KC, 10 MBON)")

POP = 40
GENS = 80
pop = [GrowingGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]

print("JIT warmup...")
_ = evaluate(pop[:2], feats[:20], labs[:20], brain_cfg, n_test=10, seed=0)
print("Done.\n")

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = evaluate(pop, feats, labs, brain_cfg, n_test=80, seed=rng.integers(1e9))
    bi = np.argmax(fit)
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2)}
    history.append(h)
    if gen % 10 == 0 or gen == GENS - 1:
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s)")

    si = np.argsort(fit)[::-1]
    new = [pop[i].copy() for i in si[:4]]
    while len(new) < POP:
        ti = rng.choice(POP, size=3, replace=False)
        pa = pop[ti[np.argmax(fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(POP, size=3, replace=False)
            ch = crossover(pa, pop[ti2[np.argmax(fit[ti2])]], rng)
        else:
            ch = pa.copy()
        new.append(mutate(ch, rng))
    pop = new

total = time.time() - t0_total
best = max(h["best"] for h in history)

print(f"\n{'='*50}")
print(f"HOG + MUSHROOM BODY RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"{'='*50}")

with open("docs/journal/hog_mnist_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total, 1),
               "neurons": brain_cfg.num_neurons, "n_features": n_features}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("HOG Features + Mushroom Body: MNIST 10-Class", fontsize=14, fontweight="bold")

axes[0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-",
             color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-",
             color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance")
axes[0].axhline(y=0.5, color="green", ls="--", alpha=0.5, label="50%")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

comparison = {
    "8x8 raw": 0.234, "16x16 raw": 0.237, "16x16 raw big": 0.268,
    "16x16+cortex": 0.250, "HOG+MB": best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors_bar = ["#E74C3C"]*4 + ["#2ECC71"]
axes[1].barh(names, vals, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1].set_xlabel("Best Fitness"); axes[1].set_title("All Approaches Compared")
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/hog_mnist_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
