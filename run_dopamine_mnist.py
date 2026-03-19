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
from src.simulator.dopamine_stdp import train_and_evaluate

rng = np.random.default_rng(42)

stdp_cfg = config.get("dopamine_stdp", {})
LR = 0.0003
TAU_ELIG = stdp_cfg.get("tau_eligibility", 0.040)
TAU_KC = stdp_cfg.get("tau_kc_trace", 0.020)
TAU_MBON = stdp_cfg.get("tau_mbon_trace", 0.020)
W_MIN = stdp_cfg.get("w_min", 0.0)
W_MAX = stdp_cfg.get("w_max", 15.0)
REWARD = 1.0
PUNISH = -0.1
W_INIT = stdp_cfg.get("w_init_mean", 5.0)
MAX_RATE = 500.0
INPUT_WEIGHT = 100e-9


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


class DopamineGenome:
    def __init__(self, config, pn_kc, kc_thresh, kc_apl_w=2.0, apl_kc_w=200.0, w_init=5.0):
        self.config = config
        self.pn_kc = pn_kc
        self.kc_thresh = kc_thresh
        self.kc_apl_w = kc_apl_w
        self.apl_kc_w = apl_kc_w
        self.w_init = w_init

    @staticmethod
    def random(config, rng, kc_pn_k=6):
        pn_kc = np.zeros((config.num_pn, config.num_kc))
        for kc in range(config.num_kc):
            chosen = rng.choice(config.num_pn, size=min(kc_pn_k, config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))
        kc_thresh = rng.uniform(-48, -42, size=config.num_kc)
        return DopamineGenome(config, pn_kc, kc_thresh)

    def copy(self):
        return DopamineGenome(self.config, self.pn_kc.copy(), self.kc_thresh.copy(),
                             self.kc_apl_w, self.apl_kc_w, self.w_init)

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
    c.w_init = float(np.clip(c.w_init + rng.normal(0, 0.5), 1.0, 10.0))
    return c


def crossover(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(m, a.kc_thresh, b.kc_thresh).copy()
    w_init = (a.w_init + b.w_init) / 2
    return DopamineGenome(a.config, pn_kc, kc_thresh, w_init=w_init)


def build_population_arrays(pop, brain_cfg):
    n = brain_cfg.num_neurons
    ps = len(pop)
    pop_W_exc = np.zeros((ps, n, n))
    pop_W_inh = np.zeros((ps, n, n))
    pop_V_thresh = np.zeros((ps, n))
    pop_kc_mbon = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    for gi, g in enumerate(pop):
        We, Wi = g.build_weight_matrices()
        pop_W_exc[gi] = We
        pop_W_inh[gi] = Wi
        pop_V_thresh[gi] = g.build_threshold_vector()
        pop_kc_mbon[gi] = g.init_kc_mbon()
    return pop_W_exc, pop_W_inh, pop_V_thresh, pop_kc_mbon


def run_evaluation(pop, brain_cfg, train_spikes, train_labels, test_spikes, test_labels, tau_m, V_rest, V_reset, g_L):
    pop_W_exc, pop_W_inh, pop_V_thresh, pop_kc_mbon = build_population_arrays(pop, brain_cfg)
    return train_and_evaluate(
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
    )


train_feats, train_labs, test_feats, test_labs = prepare_data(30, 10)
n_features = train_feats.shape[1]
print(f"HOG features: {n_features}, Train: {len(train_labs)}, Test: {len(test_labs)}")

NUM_KC = 500
brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons ({n_features} PN, {NUM_KC} KC, 10 MBON)")

print(f"STDP params: lr={LR}, reward={REWARD}, punish={PUNISH}")
print(f"Input: max_rate={MAX_RATE}Hz, input_weight={INPUT_WEIGHT*1e9:.0f}nS")

EPOCHS = 3

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
print(f"Training: {len(train_labels_full)} presentations ({EPOCHS} epochs x {len(train_labs)} images)")
print(f"Testing: {len(test_labs)} images")
print(f"Spike arrays: train={train_spikes.shape}, test={test_spikes.shape}")

tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()

print("\n--- Phase 1: Pure STDP Test (no evolution) ---")
pure_pop = [DopamineGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(3)]

print("JIT warmup + pure STDP test...")
t0 = time.time()
pure_fit = run_evaluation(pure_pop, brain_cfg, train_spikes, train_labels_full, test_spikes, test_labs, tau_m, V_rest, V_reset, g_L)
print(f"Pure STDP ({len(pure_pop)} random brains): best={pure_fit.max():.4f} mean={pure_fit.mean():.4f} ({time.time()-t0:.1f}s)")
for i, f in enumerate(pure_fit):
    print(f"  Brain {i}: fitness={f:.4f}")

print("\n--- Phase 2: Evolution + STDP ---")
POP = 25
GENS = 50
pop = [DopamineGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = run_evaluation(pop, brain_cfg, train_spikes, train_labels_full, test_spikes, test_labs, tau_m, V_rest, V_reset, g_L)
    bi = np.argmax(fit)
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2)}
    history.append(h)

    if gen % 5 == 0 or gen == GENS - 1:
        el = time.time() - t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s) [{el:.0f}s]")

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
print(f"DOPAMINE STDP + EVOLUTION RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"Pure STDP baseline: {pure_fit.max():.4f}")
print(f"{'='*50}")

with open("docs/journal/dopamine_stdp_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total, 1),
               "pure_stdp_best": round(float(pure_fit.max()), 4),
               "pure_stdp_mean": round(float(pure_fit.mean()), 4),
               "neurons": brain_cfg.num_neurons, "n_features": n_features,
               "params": {"lr": LR, "tau_elig": TAU_ELIG, "reward": REWARD,
                          "punishment": PUNISH, "w_init": W_INIT,
                          "max_rate": MAX_RATE, "input_weight_nS": INPUT_WEIGHT*1e9}}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dopamine-STDP + Evolution: MNIST 10-Class", fontsize=14, fontweight="bold")

axes[0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-",
             color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-",
             color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance")
axes[0].axhline(y=pure_fit.max(), color="purple", ls="--", alpha=0.5,
                label=f"Pure STDP ({pure_fit.max():.2f})")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

comparison = {
    "8x8 raw GA": 0.234,
    "16x16 raw GA": 0.237,
    "HOG+GA": 0.237,
    "16x16+cortex GA": 0.250,
    "16x16 big GA": 0.268,
    "Pure STDP": round(float(pure_fit.max()), 3),
    "STDP+Evo": best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors_bar = ["#E74C3C"] * 5 + ["#F39C12", "#2ECC71"]
axes[1].barh(names, vals, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1].set_xlabel("Best Fitness"); axes[1].set_title("All Approaches Compared")
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/dopamine_stdp_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
