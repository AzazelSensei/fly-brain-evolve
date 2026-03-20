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

from src.simulator.growing_brain import BrainConfig
from src.simulator.dopamine_stdp import train_and_evaluate_attention

rng = np.random.default_rng(42)

LR = 0.0003
TAU_ELIG = 0.040
TAU_KC = 0.020
TAU_MBON = 0.020
W_MIN = 0.0
W_MAX = 15.0
REWARD = 1.0
PUNISH = -0.1
W_INIT = 5.0
MAX_RATE = 500.0
INPUT_WEIGHT = 100e-9
NUM_KC = 500
EPOCHS = 3
POP = 20
GENS = 50


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


def prepare_data(n_train_per_class=30, n_test_per_class=10, img_size=16):
    train_feats, train_labs = [], []
    test_feats, test_labs = [], []
    for d in range(10):
        mask = all_labels == str(d)
        di = all_images[mask]
        ch = rng.choice(len(di), size=n_train_per_class + n_test_per_class, replace=False)
        for i, idx in enumerate(ch):
            img = np.array(Image.fromarray(
                di[idx].reshape(28, 28).astype(np.uint8)
            ).resize((img_size, img_size), Image.BILINEAR))
            hog = compute_hog_simple(img.flatten() / 255.0, img_size)
            if i < n_train_per_class:
                train_feats.append(hog)
                train_labs.append(d)
            else:
                test_feats.append(hog)
                test_labs.append(d)
    return (np.array(train_feats), np.array(train_labs, dtype=np.int32),
            np.array(test_feats), np.array(test_labs, dtype=np.int32))


def encode_spikes(features, num_steps, dt, max_rate, seed):
    r = np.random.default_rng(seed)
    n_imgs, n_feat = features.shape
    spikes = np.zeros((n_imgs, num_steps, n_feat), dtype=np.bool_)
    thresholds = features * max_rate * dt
    for i in range(n_imgs):
        for pn in range(n_feat):
            if thresholds[i, pn] > 0:
                spikes[i, :, pn] = r.random(num_steps) < thresholds[i, pn]
    return spikes


class AttentionGenome:
    def __init__(self, config, pn_kc, kc_thresh, kc_apl_w=2.0, apl_kc_w=200.0,
                 w_init=5.0, fb_strength=0.5, fb_inhibition=0.1):
        self.config = config
        self.pn_kc = pn_kc
        self.kc_thresh = kc_thresh
        self.kc_apl_w = kc_apl_w
        self.apl_kc_w = apl_kc_w
        self.w_init = w_init
        self.fb_strength = fb_strength
        self.fb_inhibition = fb_inhibition

    @staticmethod
    def random(config, rng, kc_pn_k=6):
        pn_kc = np.zeros((config.num_pn, config.num_kc))
        for kc in range(config.num_kc):
            chosen = rng.choice(config.num_pn, size=min(kc_pn_k, config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))
        kc_thresh = rng.uniform(-48, -42, size=config.num_kc)
        fb_str = rng.uniform(0.1, 1.0)
        fb_inh = rng.uniform(0.01, 0.3)
        return AttentionGenome(config, pn_kc, kc_thresh, fb_strength=fb_str, fb_inhibition=fb_inh)

    def copy(self):
        return AttentionGenome(self.config, self.pn_kc.copy(), self.kc_thresh.copy(),
                               self.kc_apl_w, self.apl_kc_w, self.w_init,
                               self.fb_strength, self.fb_inhibition)

    def build_weight_matrices(self):
        n = self.config.num_neurons
        W_exc = np.zeros((n, n))
        W_inh = np.zeros((n, n))
        cfg = self.config
        pi, ki = np.nonzero(self.pn_kc)
        for idx in range(len(pi)):
            W_exc[cfg.pn_start + pi[idx], cfg.kc_start + ki[idx]] = self.pn_kc[pi[idx], ki[idx]] * 1e-9
        for kc_i in range(cfg.num_kc):
            W_exc[cfg.kc_start + kc_i, cfg.apl_start] = self.kc_apl_w * 1e-9
        for kc_i in range(cfg.num_kc):
            W_inh[cfg.apl_start, cfg.kc_start + kc_i] = self.apl_kc_w * 1e-9
        return W_exc, W_inh

    def build_threshold_vector(self):
        n = self.config.num_neurons
        cfg = self.config
        V_thresh = np.full(n, -0.055)
        V_thresh[cfg.kc_start:cfg.kc_end] = self.kc_thresh * 1e-3
        V_thresh[cfg.mbon_start:cfg.mbon_end] = -0.050
        V_thresh[cfg.apl_start:cfg.apl_end] = -0.045
        return V_thresh

    def init_kc_mbon(self):
        return np.full((self.config.num_kc, self.config.num_mbon), self.w_init)


def mutate(g, rng):
    c = g.copy()
    mask = c.pn_kc > 0
    c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
    c.pn_kc[~mask] = 0
    idx = rng.choice(len(c.kc_thresh), size=20, replace=False)
    c.kc_thresh[idx] += rng.normal(0, 1, size=20)
    c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    c.w_init = float(np.clip(c.w_init + rng.normal(0, 0.3), 1.0, 10.0))
    c.fb_strength = float(np.clip(c.fb_strength + rng.normal(0, 0.15), 0.0, 5.0))
    c.fb_inhibition = float(np.clip(c.fb_inhibition + rng.normal(0, 0.05), 0.0, 2.0))
    return c


def crossover(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(m, a.kc_thresh, b.kc_thresh).copy()
    w_init = (a.w_init + b.w_init) / 2
    fb_str = a.fb_strength if rng.random() < 0.5 else b.fb_strength
    fb_inh = a.fb_inhibition if rng.random() < 0.5 else b.fb_inhibition
    return AttentionGenome(a.config, pn_kc, kc_thresh, w_init=w_init,
                           fb_strength=fb_str, fb_inhibition=fb_inh)


def run_evaluation(pop, brain_cfg, train_spikes, train_labels, test_spikes, test_labels,
                   tau_m, V_rest, V_reset, g_L):
    ps = len(pop)
    n = brain_cfg.num_neurons
    pop_W_exc = np.zeros((ps, n, n))
    pop_W_inh = np.zeros((ps, n, n))
    pop_V_thresh = np.zeros((ps, n))
    pop_kc_mbon = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    pop_fb_str = np.zeros(ps)
    pop_fb_inh = np.zeros(ps)

    for gi, g in enumerate(pop):
        We, Wi = g.build_weight_matrices()
        pop_W_exc[gi] = We
        pop_W_inh[gi] = Wi
        pop_V_thresh[gi] = g.build_threshold_vector()
        pop_kc_mbon[gi] = g.init_kc_mbon()
        pop_fb_str[gi] = g.fb_strength
        pop_fb_inh[gi] = g.fb_inhibition

    return train_and_evaluate_attention(
        pop_W_exc, pop_W_inh, pop_kc_mbon, pop_V_thresh,
        train_spikes, train_labels,
        test_spikes, test_labels,
        tau_m, V_rest, V_reset, g_L,
        0.0, -0.080, 0.005, 0.010,
        dt_val, num_steps, refr_steps,
        brain_cfg.num_neurons, brain_cfg.num_pn, brain_cfg.num_mbon, brain_cfg.num_kc,
        brain_cfg.kc_start, brain_cfg.kc_end, brain_cfg.mbon_start, brain_cfg.mbon_end,
        INPUT_WEIGHT,
        TAU_KC, TAU_MBON, TAU_ELIG,
        LR, W_MIN, W_MAX,
        REWARD, PUNISH,
        pop_fb_str, pop_fb_inh,
    )


train_feats, train_labs, test_feats, test_labs = prepare_data(30, 10)
n_features = train_feats.shape[1]
print(f"HOG features: {n_features}, Train: {len(train_labs)}, Test: {len(test_labs)}")

brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons ({n_features} PN, {NUM_KC} KC, 10 MBON)")
print(f"Phase 4: MBON->KC attention feedback (0 extra neurons)")
print(f"STDP: lr={LR}, epochs={EPOCHS}")

tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()

print("\nEncoding spike trains...")
test_spikes = encode_spikes(test_feats, num_steps, dt_val, MAX_RATE, rng.integers(1e9))
all_train_spikes = []
all_train_labels = []
for epoch in range(EPOCHS):
    epoch_spikes = encode_spikes(train_feats, num_steps, dt_val, MAX_RATE, rng.integers(1e9))
    perm = rng.permutation(len(train_labs))
    all_train_spikes.append(epoch_spikes[perm])
    all_train_labels.append(train_labs[perm])
train_spikes = np.concatenate(all_train_spikes, axis=0)
train_labels_full = np.concatenate(all_train_labels, axis=0)
print(f"Training: {len(train_labels_full)} presentations ({EPOCHS} epochs)")

print("\n--- JIT Warmup ---")
warmup_pop = [AttentionGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(2)]
t0 = time.time()
_ = run_evaluation(warmup_pop, brain_cfg, train_spikes[:100], train_labels_full[:100],
                   test_spikes[:20], test_labs[:20], tau_m, V_rest, V_reset, g_L)
print(f"Done ({time.time()-t0:.1f}s)")

print(f"\n--- Phase 4: Attention + STDP + Evolution ---")
print(f"POP={POP}, GENS={GENS}")
pop = [AttentionGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = run_evaluation(pop, brain_cfg, train_spikes, train_labels_full,
                         test_spikes, test_labs, tau_m, V_rest, V_reset, g_L)
    bi = np.argmax(fit)
    best_g = pop[bi]
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2),
         "fb_str": round(best_g.fb_strength, 3), "fb_inh": round(best_g.fb_inhibition, 3)}
    history.append(h)

    if gen % 5 == 0 or gen == GENS - 1:
        el = time.time() - t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} "
              f"fb={h['fb_str']:.2f}/{h['fb_inh']:.2f} ({h['time']:.1f}s) [{el:.0f}s]")

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

print(f"\n{'='*60}")
print(f"PHASE 4: ATTENTION + STDP + EVOLUTION RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"Phase 2 best (no attention): 0.732")
print(f"Best feedback params: str={pop[np.argmax(fit)].fb_strength:.3f} inh={pop[np.argmax(fit)].fb_inhibition:.3f}")
print(f"{'='*60}")

with open("docs/journal/phase4_attention_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total, 1),
               "neurons": brain_cfg.num_neurons, "n_features": n_features,
               "params": {"lr": LR, "epochs": EPOCHS, "num_kc": NUM_KC,
                          "pop": POP, "gens": GENS}}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 4: MBON→KC Attention Feedback + STDP", fontsize=14, fontweight="bold")

axes[0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-",
             color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-",
             color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance")
axes[0].axhline(y=0.732, color="green", ls="--", alpha=0.5, label="Phase 2 (0.732)")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Fitness Over Generations")

comparison = {
    "GA only": 0.268,
    "STDP+Evo v1": 0.278,
    "STDP+Evo v2": 0.598,
    "STDP+Evo v3": 0.732,
    "Phase 3 staged": 0.650,
    "Phase 4 attention": best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors_bar = ["#E74C3C"] * 5 + ["#2ECC71"]
axes[1].barh(names, vals, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1].set_xlabel("Best Fitness"); axes[1].set_title("All Approaches")
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/phase4_attention_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
