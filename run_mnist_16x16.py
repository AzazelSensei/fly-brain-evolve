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


def evaluate(pop, imgs, labs, cfg, n_test=60, seed=42):
    r = np.random.default_rng(seed)
    n = cfg.num_neurons
    tau_m, Vr, Vrs, gL = cfg.build_params()
    ti = r.choice(len(imgs), size=n_test, replace=False)
    bs = len(pop) * n_test
    bW = np.zeros((bs, n, n)); bI = np.zeros((bs, n, n))
    bT = np.zeros((bs, n)); bS = np.zeros((bs, num_steps, cfg.num_pn), dtype=np.bool_)
    bL = np.zeros(bs, dtype=np.int32)
    idx = 0
    for g in pop:
        We, Wi = g.build_weight_matrices()
        Vt = g.build_threshold_vector()
        for t in ti:
            bW[idx]=We; bI[idx]=Wi; bT[idx]=Vt
            rates = imgs[t] * 100.0
            for pn in range(min(len(rates), cfg.num_pn)):
                bS[idx,:,pn] = r.random(num_steps) < (rates[pn]*dt_val)
            bL[idx] = labs[t]; idx += 1

    mc, kc = simulate_growing_batch(
        bW, bI, bS, bT, Vr, Vrs, gL, 0.0, -0.080,
        tau_m, 0.005, 0.010, dt_val, num_steps, refr_steps,
        n, cfg.mbon_start, cfg.mbon_end, cfg.kc_start, cfg.kc_end, 50e-9, cfg.num_pn)

    fit = np.zeros(len(pop))
    for gi in range(len(pop)):
        s, e = gi*n_test, (gi+1)*n_test
        ok = sum(1 for t in range(s,e) if (np.argmax(mc[t]) if mc[t].sum()>0 else r.integers(10))==bL[t])
        acc = ok/n_test
        sp = np.mean([np.sum(kc[t]) for t in range(s,e)])/cfg.num_kc
        fit[gi] = acc + max(0,1-abs(sp-0.1)/0.1)*0.1 - np.count_nonzero(pop[gi].pn_kc)/50000*0.01
    return fit


def mutate(g, rate, r):
    c = g.copy()
    if r.random() < rate:
        c.kc_mbon = np.clip(c.kc_mbon + r.normal(0, 0.5, c.kc_mbon.shape), 0, 15)
    if r.random() < rate:
        mask = c.pn_kc > 0
        c.pn_kc = np.clip(c.pn_kc + r.normal(0, 0.3, c.pn_kc.shape)*mask, 0, 8)
        c.pn_kc[~mask] = 0
    if r.random() < rate*0.5:
        idx = r.choice(c.kc_thresh.shape[0], size=20, replace=False)
        c.kc_thresh[idx] += r.normal(0, 1, size=20)
        c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    return c


def crossover(a, b, r):
    m = r.random(a.config.num_kc) < 0.5
    return GrowingGenome(a.config,
        np.where(m[np.newaxis,:],a.pn_kc,b.pn_kc).copy(),
        np.where(m[:,np.newaxis],a.kc_mbon,b.kc_mbon).copy(),
        np.where(m,a.kc_thresh,b.kc_thresh).copy())


def run_experiment(name, brain_cfg, imgs, labs, pop_size=30, gens=50):
    print(f"\n{'='*60}")
    print(f"{name}: {brain_cfg.num_neurons} neurons ({brain_cfg.num_pn} PN, {brain_cfg.num_kc} KC, {brain_cfg.num_mbon} MBON)")
    print(f"{'='*60}")

    pop = [GrowingGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(pop_size)]
    _ = evaluate(pop[:2], imgs[:20], labs[:20], brain_cfg, n_test=10, seed=0)

    history = []
    t0 = time.time()
    for gen in range(gens):
        gt = time.time()
        fit = evaluate(pop, imgs, labs, brain_cfg, n_test=60, seed=rng.integers(1e9))
        bi = np.argmax(fit)
        h = {"gen":gen, "best":round(float(fit[bi]),4), "mean":round(float(fit.mean()),4),
             "time":round(time.time()-gt,2)}
        history.append(h)
        if gen % 10 == 0 or gen == gens-1:
            print(f"  Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s)")
        si = np.argsort(fit)[::-1]
        new = [pop[i].copy() for i in si[:3]]
        while len(new) < pop_size:
            ti = rng.choice(pop_size, size=3, replace=False)
            pa = pop[ti[np.argmax(fit[ti])]]
            if rng.random() < 0.4:
                ti2 = rng.choice(pop_size, size=3, replace=False)
                ch = crossover(pa, pop[ti2[np.argmax(fit[ti2])]], rng)
            else:
                ch = pa.copy()
            new.append(mutate(ch, 0.3, rng))
        pop = new

    total = time.time() - t0
    best = max(h["best"] for h in history)
    print(f"  Result: best={best:.4f} in {total:.0f}s")
    return {"history": history, "best": best, "time_s": round(total, 1), "neurons": brain_cfg.num_neurons}


results = {}

imgs8, labs8 = prepare(8, n_per_class=30)
results["8x8 (64 PN, 200 KC)"] = run_experiment(
    "8x8 baseline", BrainConfig(64, 200, 10, 1), imgs8, labs8)

imgs16, labs16 = prepare(16, n_per_class=30)
results["16x16 (256 PN, 200 KC)"] = run_experiment(
    "16x16 better eyes", BrainConfig(256, 200, 10, 1), imgs16, labs16)

results["16x16 (256 PN, 500 KC)"] = run_experiment(
    "16x16 bigger brain", BrainConfig(256, 500, 10, 1), imgs16, labs16)

with open("docs/journal/mnist_resolution_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Visual Resolution on MNIST Classification", fontsize=14, fontweight="bold")

colors = ["#E74C3C", "#3498DB", "#2ECC71"]
for i, (name, res) in enumerate(results.items()):
    h = res["history"]
    axes[0].plot([x["gen"] for x in h], [x["best"] for x in h], "-", color=colors[i], lw=2, label=name)
    axes[1].plot([x["gen"] for x in h], [x["mean"] for x in h], "-", color=colors[i], lw=1.5, label=name)

for ax in axes:
    ax.axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance 10%")
    ax.set_xlabel("Generation"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
axes[0].set_ylabel("Best Fitness"); axes[0].set_title("Best Individual")
axes[1].set_ylabel("Mean Fitness"); axes[1].set_title("Population Average")

plt.tight_layout()
fig.savefig("docs/figures/mnist_resolution_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nFigure saved")
