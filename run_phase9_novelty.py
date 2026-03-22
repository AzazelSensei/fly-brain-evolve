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
dt_val = 0.0001
num_steps = int(0.1 / dt_val)
refr_steps = int(0.002 / dt_val)

data = np.load("data/mnist_cache.npz", allow_pickle=True)
all_images, all_labels = data["data"], data["target"]

from src.simulator.growing_brain import BrainConfig
from src.simulator.dopamine_stdp import train_and_evaluate_meta

rng = np.random.default_rng(42)

NUM_KC = 500
MAX_RATE = 500.0
INPUT_WEIGHT = 100e-9
POP = 25
GENS = 60
EPOCHS = 5
NOVELTY_WEIGHT = 0.05
NOVELTY_K = 5


def compute_hog(image_flat, img_size=16, cell_size=4, n_bins=8):
    img = image_flat.reshape(img_size, img_size)
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]; gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2); angle = np.arctan2(gy, gx) % np.pi
    histograms = np.zeros((img_size // cell_size, img_size // cell_size, n_bins))
    for cy in range(img_size // cell_size):
        for cx in range(img_size // cell_size):
            y0, y1 = cy*cell_size, (cy+1)*cell_size; x0, x1 = cx*cell_size, (cx+1)*cell_size
            for py in range(cell_size):
                for px in range(cell_size):
                    bi = int(angle[y0+py, x0+px] / np.pi * n_bins) % n_bins
                    histograms[cy, cx, bi] += magnitude[y0+py, x0+px]
    f = histograms.flatten(); n = np.linalg.norm(f)
    if n > 0: f /= n
    return f


def compute_intensity(image_flat, target_size=8):
    img = image_flat.reshape(16, 16)
    pooled = np.zeros((target_size, target_size))
    step = 16 // target_size
    for y in range(target_size):
        for x in range(target_size):
            pooled[y, x] = img[y*step:(y+1)*step, x*step:(x+1)*step].mean()
    flat = pooled.flatten()
    if flat.max() > 0: flat /= flat.max()
    return flat


def prepare_data(n_train=30, n_test=10):
    tr_f, tr_l, te_f, te_l = [], [], [], []
    for d in range(10):
        mask = all_labels == str(d); di = all_images[mask]
        ch = rng.choice(len(di), size=n_train+n_test, replace=False)
        for i, idx in enumerate(ch):
            img = np.array(Image.fromarray(di[idx].reshape(28,28).astype(np.uint8)).resize((16,16), Image.BILINEAR))
            flat = img.flatten()/255.0
            combined = np.concatenate([compute_hog(flat, 16), compute_intensity(flat, 8)])
            if i < n_train: tr_f.append(combined); tr_l.append(d)
            else: te_f.append(combined); te_l.append(d)
    return np.array(tr_f), np.array(tr_l, dtype=np.int32), np.array(te_f), np.array(te_l, dtype=np.int32)


def encode_spikes(features, seed):
    r = np.random.default_rng(seed)
    n_imgs, n_feat = features.shape
    spikes = np.zeros((n_imgs, num_steps, n_feat), dtype=np.bool_)
    thresholds = features * MAX_RATE * dt_val
    for i in range(n_imgs):
        for pn in range(n_feat):
            if thresholds[i, pn] > 0:
                spikes[i, :, pn] = r.random(num_steps) < thresholds[i, pn]
    return spikes


def compute_novelty(pop, archive, k=NOVELTY_K):
    behaviors = []
    for g in pop:
        pn_kc_flat = g.pn_kc[g.pn_kc > 0]
        behavior = np.array([
            g.lr, g.reward, g.punish, g.tau_kc, g.tau_mbon,
            g.fb_str, g.fb_inh, g.kc_decay, float(g.max_passes),
            np.mean(g.kc_thresh), np.std(g.kc_thresh),
            len(pn_kc_flat), np.mean(pn_kc_flat) if len(pn_kc_flat) > 0 else 0,
        ])
        behaviors.append(behavior)
    behaviors = np.array(behaviors)

    all_behaviors = np.vstack([behaviors] + ([archive] if len(archive) > 0 else []))

    norms = np.linalg.norm(behaviors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    behaviors_normed = behaviors / norms

    all_norms = np.linalg.norm(all_behaviors, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1
    all_normed = all_behaviors / all_norms

    novelty_scores = np.zeros(len(pop))
    for i in range(len(pop)):
        dists = np.linalg.norm(all_normed - behaviors_normed[i], axis=1)
        dists = np.sort(dists)
        novelty_scores[i] = np.mean(dists[1:k+1]) if len(dists) > k else np.mean(dists[1:])

    return novelty_scores, behaviors


class NoveltyGenome:
    def __init__(self, config, pn_kc, kc_thresh, kc_apl_w=2.0, apl_kc_w=200.0,
                 w_init=5.0, fb_str=0.3, fb_inh=0.4, max_passes=5, conf_thresh=0.5, kc_decay=0.9,
                 lr=0.00021, tau_elig=0.040, tau_kc=0.031, tau_mbon=0.031,
                 reward=2.49, punish=-0.22, w_min=0.0, w_max=17.8):
        self.config = config
        self.pn_kc = pn_kc
        self.kc_thresh = kc_thresh
        self.kc_apl_w = kc_apl_w
        self.apl_kc_w = apl_kc_w
        self.w_init = w_init
        self.fb_str = fb_str
        self.fb_inh = fb_inh
        self.max_passes = max_passes
        self.conf_thresh = conf_thresh
        self.kc_decay = kc_decay
        self.lr = lr
        self.tau_elig = tau_elig
        self.tau_kc = tau_kc
        self.tau_mbon = tau_mbon
        self.reward = reward
        self.punish = punish
        self.w_min = w_min
        self.w_max = w_max

    @staticmethod
    def random(config, rng, kc_pn_k=6):
        pn_kc = np.zeros((config.num_pn, config.num_kc))
        for kc in range(config.num_kc):
            chosen = rng.choice(config.num_pn, size=min(kc_pn_k, config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))
        kc_thresh = rng.uniform(-48, -42, size=config.num_kc)
        return NoveltyGenome(config, pn_kc, kc_thresh,
                             fb_str=rng.uniform(0.1, 0.8), fb_inh=rng.uniform(0.1, 0.6),
                             max_passes=int(rng.choice([3, 4, 5])),
                             conf_thresh=rng.uniform(0.1, 0.6), kc_decay=rng.uniform(0.5, 0.95),
                             lr=10**rng.uniform(-4, -3),
                             reward=rng.uniform(1.5, 3.5), punish=rng.uniform(-0.4, -0.1),
                             w_max=rng.uniform(12, 25),
                             tau_kc=rng.uniform(0.015, 0.045), tau_mbon=rng.uniform(0.015, 0.045))

    def copy(self):
        return NoveltyGenome(self.config, self.pn_kc.copy(), self.kc_thresh.copy(),
                             self.kc_apl_w, self.apl_kc_w, self.w_init,
                             self.fb_str, self.fb_inh, self.max_passes,
                             self.conf_thresh, self.kc_decay,
                             self.lr, self.tau_elig, self.tau_kc, self.tau_mbon,
                             self.reward, self.punish, self.w_min, self.w_max)

    def build_weight_matrices(self):
        n = self.config.num_neurons; W_exc = np.zeros((n,n)); W_inh = np.zeros((n,n)); cfg = self.config
        pi, ki = np.nonzero(self.pn_kc)
        for idx in range(len(pi)):
            W_exc[cfg.pn_start + pi[idx], cfg.kc_start + ki[idx]] = self.pn_kc[pi[idx], ki[idx]] * 1e-9
        for kc_i in range(cfg.num_kc):
            W_exc[cfg.kc_start + kc_i, cfg.apl_start] = self.kc_apl_w * 1e-9
            W_inh[cfg.apl_start, cfg.kc_start + kc_i] = self.apl_kc_w * 1e-9
        return W_exc, W_inh

    def build_threshold_vector(self):
        n = self.config.num_neurons; cfg = self.config; V = np.full(n, -0.055)
        V[cfg.kc_start:cfg.kc_end] = self.kc_thresh * 1e-3
        V[cfg.mbon_start:cfg.mbon_end] = -0.050; V[cfg.apl_start:cfg.apl_end] = -0.045
        return V


def mutate(g, rng):
    c = g.copy()
    mask = c.pn_kc > 0
    c.pn_kc = np.clip(c.pn_kc + rng.normal(0, 0.3, c.pn_kc.shape) * mask, 0, 8); c.pn_kc[~mask] = 0
    idx = rng.choice(len(c.kc_thresh), size=20, replace=False)
    c.kc_thresh[idx] += rng.normal(0, 1, size=20); c.kc_thresh = np.clip(c.kc_thresh, -52, -38)
    c.w_init = float(np.clip(c.w_init + rng.normal(0, 0.3), 1.0, 10.0))
    c.fb_str = float(np.clip(c.fb_str + rng.normal(0, 0.1), 0.0, 5.0))
    c.fb_inh = float(np.clip(c.fb_inh + rng.normal(0, 0.05), 0.0, 2.0))
    c.conf_thresh = float(np.clip(c.conf_thresh + rng.normal(0, 0.05), 0.01, 0.95))
    c.kc_decay = float(np.clip(c.kc_decay + rng.normal(0, 0.05), 0.1, 0.99))
    if rng.random() < 0.15:
        c.max_passes = int(np.clip(c.max_passes + rng.choice([-1, 1]), 1, 5))
    c.lr = float(np.clip(c.lr * np.exp(rng.normal(0, 0.3)), 1e-5, 0.01))
    c.reward = float(np.clip(c.reward + rng.normal(0, 0.15), 0.1, 5.0))
    c.punish = float(np.clip(c.punish + rng.normal(0, 0.05), -2.0, -0.01))
    c.w_max = float(np.clip(c.w_max + rng.normal(0, 0.5), 5.0, 50.0))
    c.tau_kc = float(np.clip(c.tau_kc + rng.normal(0, 0.003), 0.005, 0.060))
    c.tau_mbon = float(np.clip(c.tau_mbon + rng.normal(0, 0.003), 0.005, 0.060))
    return c


def crossover(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(m, a.kc_thresh, b.kc_thresh).copy()
    pick = lambda x, y: x if rng.random() < 0.5 else y
    return NoveltyGenome(a.config, pn_kc, kc_thresh,
                         w_init=(a.w_init+b.w_init)/2,
                         fb_str=pick(a.fb_str, b.fb_str), fb_inh=pick(a.fb_inh, b.fb_inh),
                         max_passes=pick(a.max_passes, b.max_passes),
                         conf_thresh=pick(a.conf_thresh, b.conf_thresh),
                         kc_decay=pick(a.kc_decay, b.kc_decay),
                         lr=np.sqrt(a.lr*b.lr),
                         reward=pick(a.reward, b.reward), punish=pick(a.punish, b.punish),
                         w_max=pick(a.w_max, b.w_max),
                         tau_kc=pick(a.tau_kc, b.tau_kc), tau_mbon=pick(a.tau_mbon, b.tau_mbon))


def run_eval(pop, brain_cfg, train_spikes, train_labels, test_spikes, test_labels, tau_m, V_rest, V_reset, g_L):
    ps = len(pop); n = brain_cfg.num_neurons
    pWe = np.zeros((ps,n,n)); pWi = np.zeros((ps,n,n)); pVt = np.zeros((ps,n))
    pKM = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    pFS = np.zeros(ps); pFI = np.zeros(ps); pMP = np.zeros(ps)
    pCT = np.zeros(ps); pCD = np.zeros(ps)
    p_lr = np.zeros(ps); p_te = np.zeros(ps); p_tk = np.zeros(ps)
    p_tm = np.zeros(ps); p_rw = np.zeros(ps); p_pu = np.zeros(ps)
    p_wmin = np.zeros(ps); p_wmax = np.zeros(ps)
    for gi, g in enumerate(pop):
        We, Wi = g.build_weight_matrices()
        pWe[gi]=We; pWi[gi]=Wi; pVt[gi]=g.build_threshold_vector()
        pKM[gi]=np.full((brain_cfg.num_kc, brain_cfg.num_mbon), g.w_init)
        pFS[gi]=g.fb_str; pFI[gi]=g.fb_inh; pMP[gi]=g.max_passes
        pCT[gi]=g.conf_thresh; pCD[gi]=g.kc_decay
        p_lr[gi]=g.lr; p_te[gi]=g.tau_elig; p_tk[gi]=g.tau_kc
        p_tm[gi]=g.tau_mbon; p_rw[gi]=g.reward; p_pu[gi]=g.punish
        p_wmin[gi]=g.w_min; p_wmax[gi]=g.w_max
    return train_and_evaluate_meta(
        pWe, pWi, pKM, pVt, train_spikes, train_labels, test_spikes, test_labels,
        tau_m, V_rest, V_reset, g_L, 0.0, -0.080, 0.005, 0.010,
        dt_val, num_steps, refr_steps,
        brain_cfg.num_neurons, brain_cfg.num_pn, brain_cfg.num_mbon, brain_cfg.num_kc,
        brain_cfg.kc_start, brain_cfg.kc_end, brain_cfg.mbon_start, brain_cfg.mbon_end,
        INPUT_WEIGHT, p_tk, p_tm, p_te, p_lr, p_wmin, p_wmax, p_rw, p_pu,
        pFS, pFI, pMP, pCT, pCD)


train_feats, train_labs, test_feats, test_labs = prepare_data(30, 10)
n_features = train_feats.shape[1]
brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()
print(f"Phase 9: Open-Ended Evolution (Novelty Search)")
print(f"Brain: {brain_cfg.num_neurons} neurons | Novelty weight: {NOVELTY_WEIGHT}")

test_spikes = encode_spikes(test_feats, rng.integers(1e9))
all_ts, all_tl = [], []
for ep in range(EPOCHS):
    es = encode_spikes(train_feats, rng.integers(1e9))
    perm = rng.permutation(len(train_labs))
    all_ts.append(es[perm]); all_tl.append(train_labs[perm])
train_spikes = np.concatenate(all_ts); train_labels_full = np.concatenate(all_tl)
print(f"Training: {len(train_labels_full)} | Testing: {len(test_labs)}")

print("\nJIT warmup...")
wp = [NoveltyGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(2)]
_ = run_eval(wp, brain_cfg, train_spikes[:50], train_labels_full[:50],
             test_spikes[:10], test_labs[:10], tau_m, V_rest, V_reset, g_L)
print("Done.\n")

pop = [NoveltyGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]
archive = []
history = []; t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    raw_fit = run_eval(pop, brain_cfg, train_spikes, train_labels_full,
                       test_spikes, test_labs, tau_m, V_rest, V_reset, g_L)

    novelty_scores, behaviors = compute_novelty(pop, archive)

    combined_fit = raw_fit + NOVELTY_WEIGHT * novelty_scores

    best_idx = np.argmax(raw_fit)
    if gen % 3 == 0:
        archive.append(behaviors[best_idx])
    if len(archive) > 100:
        archive = archive[-100:]

    bg = pop[best_idx]
    h = {"gen": gen, "best": round(float(raw_fit[best_idx]),4),
         "mean": round(float(raw_fit.mean()),4),
         "novelty_mean": round(float(novelty_scores.mean()),4),
         "archive_size": len(archive),
         "time": round(time.time()-t0,2)}
    history.append(h)

    if gen % 5 == 0 or gen == GENS-1:
        el = time.time()-t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} "
              f"nov={h['novelty_mean']:.3f} arch={len(archive)} ({h['time']:.0f}s) [{el:.0f}s]")

    si = np.argsort(combined_fit)[::-1]
    new = [pop[i].copy() for i in si[:4]]
    while len(new) < POP:
        ti = rng.choice(POP, size=3, replace=False)
        pa = pop[ti[np.argmax(combined_fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(POP, size=3, replace=False)
            ch = crossover(pa, pop[ti2[np.argmax(combined_fit[ti2])]], rng)
        else: ch = pa.copy()
        new.append(mutate(ch, rng))
    pop = new

total = time.time()-t0_total
best = max(h["best"] for h in history)
print(f"\n{'='*60}")
print(f"PHASE 9: OPEN-ENDED EVOLUTION (NOVELTY SEARCH) RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"Phase 8 (no novelty): 0.965")
print(f"Archive size: {len(archive)}")
print(f"{'='*60}")

with open("docs/journal/phase9_novelty_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total,1),
               "novelty_weight": NOVELTY_WEIGHT, "archive_size": len(archive)}, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Phase 9: Open-Ended Evolution with Novelty Search", fontsize=14, fontweight="bold")

axes[0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-",
             color="#E74C3C", lw=2, label="Best", markersize=3)
axes[0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-",
             color="#3498DB", lw=1.5, label="Mean", markersize=2)
axes[0].axhline(y=0.965, color="green", ls="--", alpha=0.5, label="Phase 8 (0.965)")
axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Fitness")

axes[1].plot([h["gen"] for h in history], [h["novelty_mean"] for h in history], "o-",
             color="#9B59B6", lw=2, markersize=3)
axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Mean Novelty")
axes[1].set_title("Population Novelty"); axes[1].grid(True, alpha=0.3)

comp = {"GA only": 0.268, "STDP": 0.732, "MultiPass": 0.922, "Meta-Evo": 0.948,
        "Multi-Modal": 0.965, "Novelty": best}
colors = ["#E74C3C"]*5 + ["#2ECC71"]
axes[2].barh(list(comp.keys()), list(comp.values()), color=colors, alpha=0.7, edgecolor="black")
axes[2].set_xlabel("Best Fitness"); axes[2].set_title("All Phases")
for i, v in enumerate(comp.values()):
    axes[2].text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/phase9_novelty_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Figure saved")
