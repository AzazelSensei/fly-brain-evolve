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
from src.simulator.dopamine_stdp import train_and_evaluate_multipass

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
EPOCHS = 4
POP = 20
GENS = 80


def compute_hog_simple(image_flat, img_size=16, cell_size=4, n_bins=8):
    img = image_flat.reshape(img_size, img_size)
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) % np.pi
    histograms = np.zeros((img_size // cell_size, img_size // cell_size, n_bins))
    for cy in range(img_size // cell_size):
        for cx in range(img_size // cell_size):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            for py in range(cell_size):
                for px in range(cell_size):
                    bi = int(angle[y0+py, x0+px] / np.pi * n_bins) % n_bins
                    histograms[cy, cx, bi] += magnitude[y0+py, x0+px]
    f = histograms.flatten()
    n = np.linalg.norm(f)
    if n > 0: f /= n
    return f


def prepare_data(n_train_per_class=30, n_test_per_class=10):
    tr_f, tr_l, te_f, te_l = [], [], [], []
    for d in range(10):
        mask = all_labels == str(d)
        di = all_images[mask]
        ch = rng.choice(len(di), size=n_train_per_class + n_test_per_class, replace=False)
        for i, idx in enumerate(ch):
            img = np.array(Image.fromarray(di[idx].reshape(28,28).astype(np.uint8)).resize((16,16), Image.BILINEAR))
            hog = compute_hog_simple(img.flatten() / 255.0, 16)
            if i < n_train_per_class: tr_f.append(hog); tr_l.append(d)
            else: te_f.append(hog); te_l.append(d)
    return np.array(tr_f), np.array(tr_l, dtype=np.int32), np.array(te_f), np.array(te_l, dtype=np.int32)


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


class MultipassGenome:
    def __init__(self, config, pn_kc, kc_thresh, kc_apl_w=2.0, apl_kc_w=200.0,
                 w_init=5.0, fb_str=0.5, fb_inh=0.1, max_passes=2, conf_thresh=0.5, kc_decay=0.5):
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

    @staticmethod
    def random(config, rng, kc_pn_k=6):
        pn_kc = np.zeros((config.num_pn, config.num_kc))
        for kc in range(config.num_kc):
            chosen = rng.choice(config.num_pn, size=min(kc_pn_k, config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))
        kc_thresh = rng.uniform(-48, -42, size=config.num_kc)
        return MultipassGenome(config, pn_kc, kc_thresh,
                               fb_str=rng.uniform(0.1, 0.8), fb_inh=rng.uniform(0.1, 0.6),
                               max_passes=int(rng.choice([2, 3, 4, 5])),
                               conf_thresh=rng.uniform(0.3, 0.8), kc_decay=rng.uniform(0.2, 0.7))

    def copy(self):
        return MultipassGenome(self.config, self.pn_kc.copy(), self.kc_thresh.copy(),
                               self.kc_apl_w, self.apl_kc_w, self.w_init,
                               self.fb_str, self.fb_inh, self.max_passes,
                               self.conf_thresh, self.kc_decay)

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
    c.fb_str = float(np.clip(c.fb_str + rng.normal(0, 0.15), 0.0, 5.0))
    c.fb_inh = float(np.clip(c.fb_inh + rng.normal(0, 0.05), 0.0, 2.0))
    c.conf_thresh = float(np.clip(c.conf_thresh + rng.normal(0, 0.1), 0.1, 0.95))
    c.kc_decay = float(np.clip(c.kc_decay + rng.normal(0, 0.1), 0.05, 0.95))
    if rng.random() < 0.1:
        c.max_passes = int(np.clip(c.max_passes + rng.choice([-1, 1]), 1, 5))
    return c


def crossover(a, b, rng):
    m = rng.random(a.config.num_kc) < 0.5
    pn_kc = np.where(m[np.newaxis, :], a.pn_kc, b.pn_kc).copy()
    kc_thresh = np.where(m, a.kc_thresh, b.kc_thresh).copy()
    pick = lambda x, y: x if rng.random() < 0.5 else y
    return MultipassGenome(a.config, pn_kc, kc_thresh, w_init=(a.w_init+b.w_init)/2,
                           fb_str=pick(a.fb_str, b.fb_str), fb_inh=pick(a.fb_inh, b.fb_inh),
                           max_passes=pick(a.max_passes, b.max_passes),
                           conf_thresh=pick(a.conf_thresh, b.conf_thresh),
                           kc_decay=pick(a.kc_decay, b.kc_decay))


train_feats, train_labs, test_feats, test_labs = prepare_data(30, 10)
n_features = train_feats.shape[1]
brain_cfg = BrainConfig(num_pn=n_features, num_kc=NUM_KC, num_mbon=10, num_apl=1)
tau_m, V_rest, V_reset, g_L = brain_cfg.build_params()
print(f"Brain: {brain_cfg.num_neurons} neurons | Phase 5 tuned: 5-pass + Attention + STDP")

test_spikes = encode_spikes(test_feats, num_steps, dt_val, MAX_RATE, rng.integers(1e9))
all_ts, all_tl = [], []
for ep in range(EPOCHS):
    es = encode_spikes(train_feats, num_steps, dt_val, MAX_RATE, rng.integers(1e9))
    perm = rng.permutation(len(train_labs))
    all_ts.append(es[perm]); all_tl.append(train_labs[perm])
train_spikes = np.concatenate(all_ts); train_labels_full = np.concatenate(all_tl)
print(f"Training: {len(train_labels_full)} | Testing: {len(test_labs)}")


def run_eval(pop):
    ps = len(pop); n = brain_cfg.num_neurons
    pWe = np.zeros((ps,n,n)); pWi = np.zeros((ps,n,n)); pVt = np.zeros((ps,n))
    pKM = np.zeros((ps, brain_cfg.num_kc, brain_cfg.num_mbon))
    pFS = np.zeros(ps); pFI = np.zeros(ps); pMP = np.zeros(ps); pCT = np.zeros(ps); pCD = np.zeros(ps)
    for gi, g in enumerate(pop):
        We, Wi = g.build_weight_matrices()
        pWe[gi]=We; pWi[gi]=Wi; pVt[gi]=g.build_threshold_vector()
        pKM[gi]=np.full((brain_cfg.num_kc, brain_cfg.num_mbon), g.w_init)
        pFS[gi]=g.fb_str; pFI[gi]=g.fb_inh; pMP[gi]=g.max_passes; pCT[gi]=g.conf_thresh; pCD[gi]=g.kc_decay
    return train_and_evaluate_multipass(
        pWe, pWi, pKM, pVt, train_spikes, train_labels_full, test_spikes, test_labs,
        tau_m, V_rest, V_reset, g_L, 0.0, -0.080, 0.005, 0.010,
        dt_val, num_steps, refr_steps,
        brain_cfg.num_neurons, brain_cfg.num_pn, brain_cfg.num_mbon, brain_cfg.num_kc,
        brain_cfg.kc_start, brain_cfg.kc_end, brain_cfg.mbon_start, brain_cfg.mbon_end,
        INPUT_WEIGHT, TAU_KC, TAU_MBON, TAU_ELIG, LR, W_MIN, W_MAX, REWARD, PUNISH,
        pFS, pFI, pMP, pCT, pCD)


print("\nJIT warmup...")
wp = [MultipassGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(2)]
_ = run_eval(wp)
print("Done.\n")

pop = [MultipassGenome.random(brain_cfg, np.random.default_rng(rng.integers(1e9))) for _ in range(POP)]
history = []; t0_total = time.time()

for gen in range(GENS):
    t0 = time.time()
    fit = run_eval(pop)
    bi = np.argmax(fit); bg = pop[bi]
    h = {"gen": gen, "best": round(float(fit[bi]),4), "mean": round(float(fit.mean()),4),
         "time": round(time.time()-t0,2), "passes": bg.max_passes,
         "conf": round(bg.conf_thresh,2), "decay": round(bg.kc_decay,2)}
    history.append(h)
    if gen % 5 == 0 or gen == GENS-1:
        el = time.time()-t0_total
        print(f"Gen {gen:3d}: best={h['best']:.4f} mean={h['mean']:.4f} "
              f"passes={h['passes']} conf={h['conf']:.2f} ({h['time']:.1f}s) [{el:.0f}s]")
    si = np.argsort(fit)[::-1]
    new = [pop[i].copy() for i in si[:4]]
    while len(new) < POP:
        ti = rng.choice(POP, size=3, replace=False)
        pa = pop[ti[np.argmax(fit[ti])]]
        if rng.random() < 0.4:
            ti2 = rng.choice(POP, size=3, replace=False)
            ch = crossover(pa, pop[ti2[np.argmax(fit[ti2])]], rng)
        else: ch = pa.copy()
        new.append(mutate(ch, rng))
    pop = new

total = time.time()-t0_total; best = max(h["best"] for h in history)
print(f"\n{'='*60}")
print(f"PHASE 5: MULTI-PASS + ATTENTION + STDP RESULT")
print(f"Best: {best:.4f}  Time: {total:.0f}s")
print(f"Phase 4 (single pass): 0.745")
print(f"Best genome: passes={pop[np.argmax(fit)].max_passes} conf={pop[np.argmax(fit)].conf_thresh:.2f} decay={pop[np.argmax(fit)].kc_decay:.2f}")
print(f"{'='*60}")

with open("docs/journal/phase5_tuned_results.json", "w") as f:
    json.dump({"history": history, "best": best, "time_s": round(total,1),
               "neurons": brain_cfg.num_neurons}, f, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 5: Multi-Pass Recurrence + Attention + STDP", fontsize=14, fontweight="bold")
ax[0].plot([h["gen"] for h in history], [h["best"] for h in history], "o-", color="#E74C3C", lw=2, label="Best", markersize=3)
ax[0].plot([h["gen"] for h in history], [h["mean"] for h in history], "s-", color="#3498DB", lw=1.5, label="Mean", markersize=2)
ax[0].axhline(y=0.745, color="green", ls="--", alpha=0.5, label="Phase 4 (0.745)")
ax[0].axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Chance")
ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

comp = {"GA only": 0.268, "STDP v2": 0.598, "STDP v3": 0.732, "Phase 4 attn": 0.745, "Phase 5 multi": best}
ax[1].barh(list(comp.keys()), list(comp.values()), color=["#E74C3C"]*4+["#2ECC71"], alpha=0.7, edgecolor="black")
ax[1].axvline(x=0.1, color="gray", ls="--", alpha=0.5); ax[1].set_xlabel("Best Fitness")
for i, v in enumerate(comp.values()): ax[1].text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); fig.savefig("docs/figures/phase5_tuned_results.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Figure saved")
