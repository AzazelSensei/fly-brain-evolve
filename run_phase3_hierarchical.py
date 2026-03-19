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
from src.simulator.visual_cortex import extract_features, _apply_filter, _max_pool_2d, make_gabor_bank
from src.simulator.hierarchical_stdp import train_and_evaluate_hierarchical

rng = np.random.default_rng(42)

NUM_FILTERS = 12
FILTER_SIZE = 5
STRIDE = 2
POOL_SIZE = 2
IMG_SIZE = 16
NUM_KC = 500
LR = 0.0003
REWARD = 1.0
PUNISH = -0.1
W_INIT = 5.0
W_MIN = 0.0
W_MAX = 15.0
MAX_RATE = 500.0
INPUT_WEIGHT = 100e-9
TAU_ELIG = 0.040
TAU_KC = 0.020
TAU_MBON = 0.020
EPOCHS = 3
POP = 20
GENS = 40


def prepare_images(n_train_per_class=30, n_test_per_class=10):
    train_imgs, train_labs = [], []
    test_imgs, test_labs = [], []
    for d in range(10):
        mask = all_labels == str(d)
        di = all_images[mask]
        ch = rng.choice(len(di), size=n_train_per_class + n_test_per_class, replace=False)
        for i, idx in enumerate(ch):
            img = np.array(Image.fromarray(
                di[idx].reshape(28, 28).astype(np.uint8)
            ).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            flat = img.flatten() / 255.0
            if i < n_train_per_class:
                train_imgs.append(flat)
                train_labs.append(d)
            else:
                test_imgs.append(flat)
                test_labs.append(d)
    return (np.array(train_imgs), np.array(train_labs, dtype=np.int32),
            np.array(test_imgs), np.array(test_labs, dtype=np.int32))


def apply_filters(image_flat, filters, img_size, stride, pool_size):
    image = image_flat.reshape(img_size, img_size)
    features = []
    for f in filters:
        response = _apply_filter(image, f, stride)
        if pool_size > 1 and response.shape[0] >= pool_size:
            response = _max_pool_2d(response, pool_size)
        features.append(response.flatten())
    result = np.concatenate(features)
    if result.max() > 0:
        result = result / result.max()
    return result


def make_random_filters(num_filters, filter_size, rng_seed):
    r = np.random.default_rng(rng_seed)
    filters = []
    for _ in range(num_filters):
        f = r.normal(0, 0.5, (filter_size, filter_size))
        f -= f.mean()
        norm = np.linalg.norm(f)
        if norm > 0:
            f /= norm
        filters.append(f)
    return filters


def make_gabor_init_filters(num_filters, filter_size):
    gabor = make_gabor_bank()
    gabor_5x5 = [g for g in gabor if g.shape[0] == filter_size]
    gabor_other = [g for g in gabor if g.shape[0] != filter_size]
    filters = []
    for g in gabor_5x5:
        filters.append(g.copy())
        if len(filters) >= num_filters:
            break
    r = np.random.default_rng(123)
    while len(filters) < num_filters:
        f = r.normal(0, 0.5, (filter_size, filter_size))
        f -= f.mean()
        norm = np.linalg.norm(f)
        if norm > 0:
            f /= norm
        filters.append(f)
    return filters[:num_filters]


class HierarchicalGenome:
    def __init__(self, config, cortex_filters, pn_kc, kc_thresh,
                 kc_apl_w=2.0, apl_kc_w=200.0, w_init=5.0):
        self.config = config
        self.cortex_filters = [f.copy() for f in cortex_filters]
        self.pn_kc = pn_kc
        self.kc_thresh = kc_thresh
        self.kc_apl_w = kc_apl_w
        self.apl_kc_w = apl_kc_w
        self.w_init = w_init

    @staticmethod
    def random(config, rng, num_filters=12, filter_size=5, kc_pn_k=6, gabor_init=True):
        if gabor_init:
            filters = make_gabor_init_filters(num_filters, filter_size)
        else:
            filters = make_random_filters(num_filters, filter_size, rng.integers(1e9))
        pn_kc = np.zeros((config.num_pn, config.num_kc))
        for kc in range(config.num_kc):
            chosen = rng.choice(config.num_pn, size=min(kc_pn_k, config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))
        kc_thresh = rng.uniform(-48, -42, size=config.num_kc)
        return HierarchicalGenome(config, filters, pn_kc, kc_thresh)

    def copy(self):
        return HierarchicalGenome(
            self.config, self.cortex_filters, self.pn_kc.copy(),
            self.kc_thresh.copy(), self.kc_apl_w, self.apl_kc_w, self.w_init)

    def extract_all_features(self, images):
        n = len(images)
        sample = apply_filters(images[0], self.cortex_filters, IMG_SIZE, STRIDE, POOL_SIZE)
        n_feat = len(sample)
        result = np.zeros((n, n_feat))
        result[0] = sample
        for i in range(1, n):
            result[i] = apply_filters(images[i], self.cortex_filters, IMG_SIZE, STRIDE, POOL_SIZE)
        return result

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
    if rng.random() < 0.3:
        fi = rng.integers(len(c.cortex_filters))
        c.cortex_filters[fi] = c.cortex_filters[fi] + rng.normal(0, 0.1, c.cortex_filters[fi].shape)
        c.cortex_filters[fi] -= c.cortex_filters[fi].mean()
        norm = np.linalg.norm(c.cortex_filters[fi])
        if norm > 0:
            c.cortex_filters[fi] /= norm
    if rng.random() < 0.05:
        fi = rng.integers(len(c.cortex_filters))
        size = c.cortex_filters[fi].shape[0]
        new_f = rng.normal(0, 0.5, (size, size))
        new_f -= new_f.mean()
        norm = np.linalg.norm(new_f)
        if norm > 0:
            new_f /= norm
        c.cortex_filters[fi] = new_f
    mask = c.pn_kc > 0
    c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
    c.pn_kc[~mask] = 0
    idx = rng.choice(len(c.kc_thresh), size=20, replace=False)
    c.kc_thresh[idx] += rng.normal(0, 1, size=20)
    c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    c.w_init = float(np.clip(c.w_init + rng.normal(0, 0.3), 1.0, 10.0))
    return c


def crossover(a, b, rng):
    n_f = len(a.cortex_filters)
    f_mask = rng.random(n_f) < 0.5
    filters = [a.cortex_filters[i].copy() if f_mask[i] else b.cortex_filters[i].copy() for i in range(n_f)]
    kc_mask = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(kc_mask[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(kc_mask, a.kc_thresh, b.kc_thresh).copy()
    w_init = (a.w_init + b.w_init) / 2
    return HierarchicalGenome(a.config, filters, pn_kc, kc_thresh, w_init=w_init)


train_imgs, train_labs, test_imgs, test_labs = prepare_images(30, 10)
print(f"Images: train={len(train_labs)}, test={len(test_labs)}, size={IMG_SIZE}x{IMG_SIZE}")

test_filters = make_gabor_init_filters(NUM_FILTERS, FILTER_SIZE)
sample_feat = apply_filters(train_imgs[0], test_filters, IMG_SIZE, STRIDE, POOL_SIZE)
n_features = len(sample_feat)
print(f"Visual cortex: {NUM_FILTERS} filters ({FILTER_SIZE}x{FILTER_SIZE}), stride={STRIDE}, pool={POOL_SIZE}")
print(f"Features per image: {n_features}")

brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons ({n_features} PN, {NUM_KC} KC, 10 MBON)")
print(f"STDP: lr={LR}, reward={REWARD}, punish={PUNISH}, epochs={EPOCHS}")

tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()


def build_features_for_population(pop, train_imgs, test_imgs, epochs):
    ps = len(pop)
    n_train = len(train_imgs)
    n_test = len(test_imgs)

    sample = pop[0].extract_all_features(train_imgs[:1])
    nf = sample.shape[1]

    total_train = epochs * n_train
    pop_train_feat = np.zeros((ps, total_train, nf))
    pop_test_feat = np.zeros((ps, n_test, nf))
    all_train_labels = np.zeros(total_train, dtype=np.int32)

    for gi, g in enumerate(pop):
        train_feat = g.extract_all_features(train_imgs)
        test_feat = g.extract_all_features(test_imgs)
        pop_test_feat[gi] = test_feat
        for ep in range(epochs):
            perm = rng.permutation(n_train)
            start = ep * n_train
            pop_train_feat[gi, start:start + n_train] = train_feat[perm]
            if gi == 0:
                all_train_labels[start:start + n_train] = train_labs[perm]

    return pop_train_feat, all_train_labels, pop_test_feat


def run_evaluation(pop, brain_cfg, train_imgs, train_labs, test_imgs, test_labs):
    pop_train_feat, train_labels_full, pop_test_feat = build_features_for_population(
        pop, train_imgs, test_imgs, EPOCHS)

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

    return train_and_evaluate_hierarchical(
        pop_W_exc, pop_W_inh, pop_kc_mbon, pop_V_thresh,
        pop_train_feat, train_labels_full,
        pop_test_feat, test_labs,
        tau_m, V_rest, V_reset, g_L,
        0.0, -0.080, 0.005, 0.010,
        dt_val, num_steps, refr_steps,
        brain_cfg.num_neurons, brain_cfg.num_pn, brain_cfg.num_mbon, brain_cfg.num_kc,
        brain_cfg.kc_start, brain_cfg.kc_end, brain_cfg.mbon_start, brain_cfg.mbon_end,
        INPUT_WEIGHT, MAX_RATE,
        TAU_KC, TAU_MBON, TAU_ELIG,
        LR, W_MIN, W_MAX,
        REWARD, PUNISH,
        rng.integers(1e9),
    )


print("\n--- Phase 3a: Gabor Init + STDP (warmup) ---")
warmup_pop = [HierarchicalGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9)),
              NUM_FILTERS, FILTER_SIZE) for _ in range(3)]
t0 = time.time()
warmup_fit = run_evaluation(warmup_pop, brain_cfg, train_imgs, train_labs, test_imgs, test_labs)
print(f"Gabor+STDP baseline ({len(warmup_pop)} brains): best={warmup_fit.max():.4f} mean={warmup_fit.mean():.4f} ({time.time()-t0:.1f}s)")

print("\n--- Phase 3b: Evolved Filters + STDP ---")
pop = [HierarchicalGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9)),
       NUM_FILTERS, FILTER_SIZE) for _ in range(POP)]

history = []
t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = run_evaluation(pop, brain_cfg, train_imgs, train_labs, test_imgs, test_labs)
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
print(f"PHASE 3: HIERARCHICAL SPARSE CODING RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"Gabor+STDP baseline: {warmup_fit.max():.4f}")
print(f"Phase 2 best (HOG+STDP): 0.732")
print(f"{'='*50}")

with open("docs/journal/phase3_hierarchical_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total, 1),
               "gabor_baseline": round(float(warmup_fit.max()), 4),
               "neurons": brain_cfg.num_neurons, "n_features": n_features,
               "params": {"lr": LR, "reward": REWARD, "punishment": PUNISH,
                          "epochs": EPOCHS, "num_filters": NUM_FILTERS,
                          "filter_size": FILTER_SIZE, "stride": STRIDE,
                          "pool_size": POOL_SIZE, "num_kc": NUM_KC}}, f, indent=2)

best_genome = pop[np.argmax(fit)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 3: Evolved Visual Cortex + Mushroom Body", fontsize=14, fontweight="bold")

axes[0, 0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-",
                color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0, 0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-",
                color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0, 0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance")
axes[0, 0].axhline(y=0.732, color="green", ls="--", alpha=0.5, label="Phase 2 (0.732)")
axes[0, 0].set_xlabel("Generation"); axes[0, 0].set_ylabel("Fitness")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title("Fitness Over Generations")

n_show = min(NUM_FILTERS, 12)
cols = 4
rows = (n_show + cols - 1) // cols
for i in range(n_show):
    ax = axes[0, 1].inset_axes([
        (i % cols) / cols, 1 - (i // cols + 1) / rows,
        0.9 / cols, 0.9 / rows
    ])
    ax.imshow(best_genome.cortex_filters[i], cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 1].axis("off")
axes[0, 1].set_title(f"Evolved {FILTER_SIZE}x{FILTER_SIZE} Filters", fontsize=11)

comparison = {
    "GA only (best)": 0.268,
    "HOG+GA": 0.237,
    "HOG+STDP v1": 0.278,
    "HOG+STDP v2": 0.598,
    "HOG+STDP v3": 0.732,
    f"Gabor+STDP": round(float(warmup_fit.max()), 3),
    f"Evolved+STDP": best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors_bar = ["#E74C3C"] * 5 + ["#F39C12", "#2ECC71"]
axes[1, 0].barh(names, vals, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1, 0].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1, 0].set_xlabel("Best Fitness"); axes[1, 0].set_title("All Approaches Compared")
for i, v in enumerate(vals):
    axes[1, 0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

sample_img = train_imgs[0]
sample_feat = apply_filters(sample_img, best_genome.cortex_filters, IMG_SIZE, STRIDE, POOL_SIZE)
axes[1, 1].bar(range(len(sample_feat)), sample_feat, color="#45B7D1", alpha=0.7, width=1)
axes[1, 1].set_title(f"Feature Vector ({n_features} dims)")
axes[1, 1].set_xlabel("Feature Index"); axes[1, 1].set_ylabel("Value")

plt.tight_layout()
fig.savefig("docs/figures/phase3_hierarchical_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
