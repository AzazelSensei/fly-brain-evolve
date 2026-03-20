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
from src.simulator.visual_cortex import _apply_filter, _max_pool_2d, make_gabor_bank
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


def extract_all_features(images, filters):
    sample = apply_filters(images[0], filters, IMG_SIZE, STRIDE, POOL_SIZE)
    n_feat = len(sample)
    result = np.zeros((len(images), n_feat))
    result[0] = sample
    for i in range(1, len(images)):
        result[i] = apply_filters(images[i], filters, IMG_SIZE, STRIDE, POOL_SIZE)
    return result


def make_gabor_init_filters(num_filters, filter_size):
    gabor = make_gabor_bank()
    gabor_match = [g for g in gabor if g.shape[0] == filter_size]
    filters = [g.copy() for g in gabor_match[:num_filters]]
    r = np.random.default_rng(123)
    while len(filters) < num_filters:
        f = r.normal(0, 0.5, (filter_size, filter_size))
        f -= f.mean()
        norm = np.linalg.norm(f)
        if norm > 0:
            f /= norm
        filters.append(f)
    return filters[:num_filters]


class FilterGenome:
    def __init__(self, filters):
        self.filters = [f.copy() for f in filters]

    @staticmethod
    def random(num_filters, filter_size, rng_seed, gabor_init=True):
        if gabor_init:
            filters = make_gabor_init_filters(num_filters, filter_size)
            r = np.random.default_rng(rng_seed)
            for i in range(len(filters)):
                filters[i] += r.normal(0, 0.05, filters[i].shape)
                filters[i] -= filters[i].mean()
                norm = np.linalg.norm(filters[i])
                if norm > 0:
                    filters[i] /= norm
        else:
            r = np.random.default_rng(rng_seed)
            filters = []
            for _ in range(num_filters):
                f = r.normal(0, 0.5, (filter_size, filter_size))
                f -= f.mean()
                norm = np.linalg.norm(f)
                if norm > 0:
                    f /= norm
                filters.append(f)
        return FilterGenome(filters)

    def copy(self):
        return FilterGenome(self.filters)


def mutate_filters(g, rng):
    c = g.copy()
    n_mutate = max(1, rng.integers(1, 4))
    for _ in range(n_mutate):
        fi = rng.integers(len(c.filters))
        if rng.random() < 0.85:
            c.filters[fi] = c.filters[fi] + rng.normal(0, 0.15, c.filters[fi].shape)
        else:
            size = c.filters[fi].shape[0]
            c.filters[fi] = rng.normal(0, 0.5, (size, size))
        c.filters[fi] -= c.filters[fi].mean()
        norm = np.linalg.norm(c.filters[fi])
        if norm > 0:
            c.filters[fi] /= norm
    return c


def crossover_filters(a, b, rng):
    n_f = len(a.filters)
    mask = rng.random(n_f) < 0.5
    filters = [a.filters[i].copy() if mask[i] else b.filters[i].copy() for i in range(n_f)]
    return FilterGenome(filters)


class ConnGenome:
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
        return ConnGenome(config, pn_kc, kc_thresh)

    def copy(self):
        return ConnGenome(self.config, self.pn_kc.copy(), self.kc_thresh.copy(),
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


def mutate_conn(g, rng):
    c = g.copy()
    mask = c.pn_kc > 0
    c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8)
    c.pn_kc[~mask] = 0
    idx = rng.choice(len(c.kc_thresh), size=20, replace=False)
    c.kc_thresh[idx] += rng.normal(0, 1, size=20)
    c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    c.w_init = float(np.clip(c.w_init + rng.normal(0, 0.3), 1.0, 10.0))
    return c


def crossover_conn(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(m, a.kc_thresh, b.kc_thresh).copy()
    w_init = (a.w_init + b.w_init) / 2
    return ConnGenome(a.config, pn_kc, kc_thresh, w_init=w_init)


def evaluate_filters(filter_pop, conn_template, brain_cfg, train_imgs, train_labs,
                     test_imgs, test_labs):
    ps = len(filter_pop)
    n = brain_cfg.num_neurons
    n_train = len(train_imgs)
    n_test = len(test_imgs)
    total_train = EPOCHS * n_train

    We, Wi = conn_template.build_weight_matrices()
    Vt = conn_template.build_threshold_vector()
    kc_mbon_init = np.full((brain_cfg.num_kc, brain_cfg.num_mbon), conn_template.w_init)

    pop_W_exc = np.zeros((ps, n, n))
    pop_W_inh = np.zeros((ps, n, n))
    pop_V_thresh = np.zeros((ps, n))
    pop_kc_mbon = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    for gi in range(ps):
        pop_W_exc[gi] = We
        pop_W_inh[gi] = Wi
        pop_V_thresh[gi] = Vt
        pop_kc_mbon[gi] = kc_mbon_init

    sample = extract_all_features(train_imgs[:1], filter_pop[0].filters)
    nf = sample.shape[1]
    pop_train_feat = np.zeros((ps, total_train, nf))
    pop_test_feat = np.zeros((ps, n_test, nf))
    all_train_labels = np.zeros(total_train, dtype=np.int32)

    for gi, fg in enumerate(filter_pop):
        train_feat = extract_all_features(train_imgs, fg.filters)
        pop_test_feat[gi] = extract_all_features(test_imgs, fg.filters)
        for ep in range(EPOCHS):
            perm = rng.permutation(n_train)
            start = ep * n_train
            pop_train_feat[gi, start:start + n_train] = train_feat[perm]
            if gi == 0:
                all_train_labels[start:start + n_train] = train_labs[perm]

    return train_and_evaluate_hierarchical(
        pop_W_exc, pop_W_inh, pop_kc_mbon, pop_V_thresh,
        pop_train_feat, all_train_labels,
        pop_test_feat, test_labs,
        *build_sim_params(brain_cfg), rng.integers(1e9))


def evaluate_conn(conn_pop, frozen_filters, brain_cfg, train_imgs, train_labs,
                  test_imgs, test_labs):
    ps = len(conn_pop)
    n = brain_cfg.num_neurons
    n_train = len(train_imgs)
    n_test = len(test_imgs)
    total_train = EPOCHS * n_train

    train_feat = extract_all_features(train_imgs, frozen_filters)
    test_feat = extract_all_features(test_imgs, frozen_filters)

    sample = train_feat[:1]
    nf = sample.shape[1]
    pop_train_feat = np.zeros((ps, total_train, nf))
    pop_test_feat = np.zeros((ps, n_test, nf))
    all_train_labels = np.zeros(total_train, dtype=np.int32)

    for gi in range(ps):
        pop_test_feat[gi] = test_feat
        for ep in range(EPOCHS):
            perm = rng.permutation(n_train)
            start = ep * n_train
            pop_train_feat[gi, start:start + n_train] = train_feat[perm]
            if gi == 0:
                all_train_labels[start:start + n_train] = train_labs[perm]

    pop_W_exc = np.zeros((ps, n, n))
    pop_W_inh = np.zeros((ps, n, n))
    pop_V_thresh = np.zeros((ps, n))
    pop_kc_mbon = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    for gi, g in enumerate(conn_pop):
        We, Wi = g.build_weight_matrices()
        pop_W_exc[gi] = We
        pop_W_inh[gi] = Wi
        pop_V_thresh[gi] = g.build_threshold_vector()
        pop_kc_mbon[gi] = np.full((brain_cfg.num_kc, brain_cfg.num_mbon), g.w_init)

    return train_and_evaluate_hierarchical(
        pop_W_exc, pop_W_inh, pop_kc_mbon, pop_V_thresh,
        pop_train_feat, all_train_labels,
        pop_test_feat, test_labs,
        *build_sim_params(brain_cfg), rng.integers(1e9))


tau_m_g, V_rest_g, V_reset_g, g_L_g = None, None, None, None


def build_sim_params(brain_cfg):
    global tau_m_g, V_rest_g, V_reset_g, g_L_g
    if tau_m_g is None:
        tau_m_g, V_rest_g, V_reset_g, g_L_g = brain_cfg.build_params()
    return (tau_m_g, V_rest_g, V_reset_g, g_L_g,
            0.0, -0.080, 0.005, 0.010,
            dt_val, num_steps, refr_steps,
            brain_cfg.num_neurons, brain_cfg.num_pn, brain_cfg.num_mbon, brain_cfg.num_kc,
            brain_cfg.kc_start, brain_cfg.kc_end, brain_cfg.mbon_start, brain_cfg.mbon_end,
            INPUT_WEIGHT, MAX_RATE,
            TAU_KC, TAU_MBON, TAU_ELIG,
            LR, W_MIN, W_MAX,
            REWARD, PUNISH)


train_imgs, train_labs, test_imgs, test_labs = prepare_images(30, 10)
print(f"Images: train={len(train_labs)}, test={len(test_labs)}")

test_filters = make_gabor_init_filters(NUM_FILTERS, FILTER_SIZE)
sample_feat = apply_filters(train_imgs[0], test_filters, IMG_SIZE, STRIDE, POOL_SIZE)
n_features = len(sample_feat)
brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
print(f"Brain: {brain_cfg.num_neurons} neurons ({n_features} PN, {NUM_KC} KC, 10 MBON)")
print(f"Staged evolution: filters first, then connectivity")

STAGE1_POP = 20
STAGE1_GENS = 40
STAGE2_POP = 20
STAGE2_GENS = 40

conn_template = ConnGenome.random(brain_cfg, np.random.default_rng(42))

print(f"\n{'='*60}")
print(f"STAGE 1: Evolve filters (connectivity fixed, STDP active)")
print(f"POP={STAGE1_POP}, GENS={STAGE1_GENS}")
print(f"{'='*60}")

filter_pop = [FilterGenome.random(NUM_FILTERS, FILTER_SIZE, rng.integers(1e9))
              for _ in range(STAGE1_POP)]

print("JIT warmup...")
_ = evaluate_filters(filter_pop[:2], conn_template, brain_cfg,
                     train_imgs, train_labs, test_imgs, test_labs)
print("Done.\n")

stage1_history = []
t0_total = time.time()

for gen in range(STAGE1_GENS):
    t0 = time.time()
    fit = evaluate_filters(filter_pop, conn_template, brain_cfg,
                           train_imgs, train_labs, test_imgs, test_labs)
    bi = np.argmax(fit)
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2)}
    stage1_history.append(h)

    if gen % 5 == 0 or gen == STAGE1_GENS - 1:
        el = time.time() - t0_total
        print(f"S1 Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s) [{el:.0f}s]")

    si = np.argsort(fit)[::-1]
    new = [filter_pop[i].copy() for i in si[:4]]
    while len(new) < STAGE1_POP:
        ti = rng.choice(STAGE1_POP, size=3, replace=False)
        pa = filter_pop[ti[np.argmax(fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(STAGE1_POP, size=3, replace=False)
            ch = crossover_filters(pa, filter_pop[ti2[np.argmax(fit[ti2])]], rng)
        else:
            ch = pa.copy()
        new.append(mutate_filters(ch, rng))
    filter_pop = new

stage1_time = time.time() - t0_total
best_filter_genome = filter_pop[np.argmax(fit)]
stage1_best = max(h["best"] for h in stage1_history)
print(f"\nStage 1 complete: best={stage1_best:.4f} ({stage1_time:.0f}s)")

frozen_filters = [f.copy() for f in best_filter_genome.filters]

print(f"\n{'='*60}")
print(f"STAGE 2: Evolve connectivity (filters frozen, STDP active)")
print(f"POP={STAGE2_POP}, GENS={STAGE2_GENS}")
print(f"{'='*60}\n")

conn_pop = [ConnGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9)))
            for _ in range(STAGE2_POP)]

stage2_history = []
t0_stage2 = time.time()

for gen in range(STAGE2_GENS):
    t0 = time.time()
    fit = evaluate_conn(conn_pop, frozen_filters, brain_cfg,
                        train_imgs, train_labs, test_imgs, test_labs)
    bi = np.argmax(fit)
    h = {"gen": gen, "best": round(float(fit[bi]), 4), "mean": round(float(fit.mean()), 4),
         "time": round(time.time() - t0, 2)}
    stage2_history.append(h)

    if gen % 5 == 0 or gen == STAGE2_GENS - 1:
        el = time.time() - t0_stage2
        print(f"S2 Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} ({h['time']:.1f}s) [{el:.0f}s]")

    si = np.argsort(fit)[::-1]
    new = [conn_pop[i].copy() for i in si[:4]]
    while len(new) < STAGE2_POP:
        ti = rng.choice(STAGE2_POP, size=3, replace=False)
        pa = conn_pop[ti[np.argmax(fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(STAGE2_POP, size=3, replace=False)
            ch = crossover_conn(pa, conn_pop[ti2[np.argmax(fit[ti2])]], rng)
        else:
            ch = pa.copy()
        new.append(mutate_conn(ch, rng))
    conn_pop = new

total_time = time.time() - t0_total
stage2_best = max(h["best"] for h in stage2_history)

print(f"\n{'='*60}")
print(f"PHASE 3 STAGED EVOLUTION RESULT")
print(f"Stage 1 (filters): best={stage1_best:.4f}")
print(f"Stage 2 (connectivity): best={stage2_best:.4f}")
print(f"Total time: {total_time:.0f}s")
print(f"Phase 2 (HOG+STDP v3): 0.732")
print(f"Phase 3 joint: 0.621")
print(f"{'='*60}")

with open("docs/journal/phase3_staged_results.json", "w") as f:
    json.dump({
        "stage1": {"history": stage1_history, "best": stage1_best,
                   "time_s": round(stage1_time, 1)},
        "stage2": {"history": stage2_history, "best": stage2_best,
                   "time_s": round(time.time() - t0_stage2, 1)},
        "total_time_s": round(total_time, 1),
        "params": {"num_filters": NUM_FILTERS, "filter_size": FILTER_SIZE,
                   "num_kc": NUM_KC, "lr": LR, "epochs": EPOCHS}
    }, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 3 Staged: Evolve Filters, Then Connectivity", fontsize=14, fontweight="bold")

axes[0, 0].plot([h["gen"] for h in stage1_history], [h["best"] for h in stage1_history],
                "o-", color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0, 0].plot([h["gen"] for h in stage1_history], [h["mean"] for h in stage1_history],
                "s-", color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0, 0].axhline(y=0.1, color="gray", ls="--", alpha=0.5)
axes[0, 0].set_xlabel("Generation"); axes[0, 0].set_ylabel("Fitness")
axes[0, 0].set_title("Stage 1: Filter Evolution"); axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot([h["gen"] for h in stage2_history], [h["best"] for h in stage2_history],
                "o-", color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0, 1].plot([h["gen"] for h in stage2_history], [h["mean"] for h in stage2_history],
                "s-", color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0, 1].axhline(y=0.732, color="green", ls="--", alpha=0.5, label="HOG+STDP v3")
axes[0, 1].axhline(y=stage1_best, color="purple", ls="--", alpha=0.5, label=f"Stage 1 ({stage1_best:.2f})")
axes[0, 1].set_xlabel("Generation"); axes[0, 1].set_ylabel("Fitness")
axes[0, 1].set_title("Stage 2: Connectivity Evolution"); axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

n_show = min(NUM_FILTERS, 12)
cols = 4; rows = (n_show + cols - 1) // cols
for i in range(n_show):
    ax = axes[1, 0].inset_axes([
        (i % cols) / cols, 1 - (i // cols + 1) / rows, 0.9 / cols, 0.9 / rows])
    ax.imshow(frozen_filters[i], cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
axes[1, 0].axis("off")
axes[1, 0].set_title(f"Best Evolved Filters ({FILTER_SIZE}x{FILTER_SIZE})")

comparison = {
    "GA only": 0.268, "HOG+STDP v3": 0.732,
    "Phase3 joint": 0.621, f"Staged S1": stage1_best, f"Staged S2": stage2_best,
}
names = list(comparison.keys())
vals = list(comparison.values())
colors_bar = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
axes[1, 1].barh(names, vals, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1, 1].axvline(x=0.1, color="gray", ls="--", alpha=0.5)
axes[1, 1].set_xlabel("Best Fitness"); axes[1, 1].set_title("Comparison")
for i, v in enumerate(vals):
    axes[1, 1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/phase3_staged_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
