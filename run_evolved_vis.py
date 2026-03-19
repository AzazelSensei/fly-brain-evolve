import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import json
import numpy as np
from brian2 import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

prefs.codegen.target = "numpy"

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)
from src.neurons.models import get_neuron_params

dt_val = config["simulation"]["dt"]

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

def make_ns(p):
    return {"tau_m":p["tau_m"]*second,"V_rest":p["V_rest"]*volt,"V_thresh":p["V_thresh"]*volt,
            "V_reset":p["V_reset"]*volt,"tau_exc":p["tau_exc"]*second,"tau_inh":p["tau_inh"]*second,
            "g_L":p["g_L"]*siemens,"E_exc":p["E_exc"]*volt,"E_inh":p["E_inh"]*volt}

genome = np.load("docs/journal/best_genome.npz")
pn_kc = genome["pn_kc"]
kc_mbon = genome["kc_mbon"]
kc_thresh = genome["kc_thresh"]

patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
pattern_names = ["Horizontal", "Vertical"]


def run_evolved(pat, seed):
    start_scope()
    defaultclock.dt = dt_val * second

    ns_pn = make_ns(get_neuron_params("pn", config))
    ns_mb = make_ns(get_neuron_params("mbon", config))
    ns_ap = make_ns(get_neuron_params("apl", config))
    ns_kc = make_ns(get_neuron_params("kc", config))
    ns_kc["V_thresh"] = float(kc_thresh.mean()) * mV

    pn = NeuronGroup(64, eqs, threshold="v>V_thresh", reset="v=V_reset",
                     refractory=2*ms, method="euler", namespace=ns_pn, name="pn")
    pn.v = -70*mV
    kc = NeuronGroup(200, eqs, threshold="v>V_thresh", reset="v=V_reset",
                     refractory=2*ms, method="euler", namespace=ns_kc, name="kc")
    kc.v = -70*mV
    mb = NeuronGroup(2, eqs, threshold="v>V_thresh", reset="v=V_reset",
                     refractory=2*ms, method="euler", namespace=ns_mb, name="mb")
    mb.v = -70*mV
    ap = NeuronGroup(1, eqs, threshold="v>V_thresh", reset="v=V_reset",
                     refractory=2*ms, method="euler", namespace=ns_ap, name="ap")
    ap.v = -70*mV

    pi, ki = np.nonzero(pn_kc[:64, :200])
    pk = Synapses(pn, kc, "ws:1", on_pre="g_exc_post+=ws*nS", name="pk")
    pk.connect(i=pi.astype(int), j=ki.astype(int))
    pk.ws = pn_kc[pi, ki]

    km = Synapses(kc, mb, "w:1", on_pre="g_exc_post+=w*nS", name="km")
    km.connect()
    km.w = kc_mbon[:200, :2].flatten()

    ka = Synapses(kc, ap, on_pre="g_exc_post+=2*nS", name="ka")
    ka.connect()
    ak = Synapses(ap, kc, on_pre="g_inh_post+=200*nS", name="ak")
    ak.connect()

    rates = image_to_rates(pat, max_rate=100.0)
    fr = np.zeros(64); fr[:len(rates)] = rates
    idx, t = generate_poisson_spike_indices_and_times(fr, 0.1, dt=dt_val, seed=seed)
    inp = SpikeGeneratorGroup(64, idx, t*second, name="inp")
    isyn = Synapses(inp, pn, on_pre="g_exc_post+=50*nS", name="is")
    isyn.connect("i==j")

    pm = SpikeMonitor(pn, name="pm")
    kmon = SpikeMonitor(kc, name="km2")
    mmon = SpikeMonitor(mb, name="mm")
    amon = SpikeMonitor(ap, name="am")
    kv = StateMonitor(kc, "v", record=list(range(20)), name="kv")
    mv = StateMonitor(mb, "v", record=True, name="mv")

    net = Network(pn,kc,mb,ap,pk,km,ka,ak,inp,isyn,pm,kmon,mmon,amon,kv,mv)
    net.run(0.1*second)
    return pm, kmon, mmon, amon, kv, mv


fig = plt.figure(figsize=(24, 22))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Evolved Drosophila Mushroom Body - Live Neural Activity",
             fontsize=20, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(5, 2, hspace=0.35, wspace=0.25,
                       left=0.06, right=0.96, top=0.93, bottom=0.04)

all_kc_sets = []

for col, (pat, name) in enumerate(zip(patterns, pattern_names)):
    pm, kmon, mmon, amon, kv, mv = run_evolved(pat, seed=42+col)

    kc_active = set(np.array(kmon.i)) if kmon.num_spikes > 0 else set()
    all_kc_sets.append(kc_active)
    kc_count = len(kc_active)
    m_counts = np.array(mmon.count)

    ax0 = fig.add_subplot(gs[0, col])
    ax0.set_facecolor("#0D1117")
    im_ax = ax0.inset_axes([0.0, 0.0, 0.2, 1.0])
    im_ax.imshow(pat, cmap="inferno", interpolation="nearest")
    im_ax.set_xticks([]); im_ax.set_yticks([])
    if pm.num_spikes > 0:
        ax0.scatter(np.array(pm.t/ms), np.array(pm.i), s=2, c="#4ECDC4", marker="|")
    ax0.set_xlim(0,100); ax0.set_ylabel("PN", color="white")
    ax0.set_title(f"{name} - PN ({pm.num_spikes} spikes)", color="#4ECDC4", fontsize=12)
    ax0.tick_params(colors="gray")
    for sp in ax0.spines.values(): sp.set_color("#333")

    ax1 = fig.add_subplot(gs[1, col])
    ax1.set_facecolor("#0D1117")
    if kmon.num_spikes > 0:
        ax1.scatter(np.array(kmon.t/ms), np.array(kmon.i), s=4, c="#FF6B6B", marker="|")
    ax1.set_xlim(0,100); ax1.set_ylabel("KC", color="white")
    ax1.set_title(f"KC: {kmon.num_spikes} spikes, {kc_count}/200 active ({kc_count/200*100:.1f}%)",
                  color="#FF6B6B", fontsize=12)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_color("#333")

    ax2 = fig.add_subplot(gs[2, col])
    ax2.set_facecolor("#0D1117")
    cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(kv.v)))
    for i in range(len(kv.v)):
        ax2.plot(np.array(kv.t/ms), np.array(kv.v[i]/mV), color=cmap[i], alpha=0.6, lw=0.7)
    ax2.axhline(y=float(kc_thresh.mean()), color="white", ls="--", alpha=0.4)
    ax2.set_ylabel("mV", color="white")
    ax2.set_title("KC Voltage Traces", color="#FF6B6B", fontsize=11)
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values(): sp.set_color("#333")

    ax3 = fig.add_subplot(gs[3, col])
    ax3.set_facecolor("#0D1117")
    for i in range(2):
        ax3.plot(np.array(mv.t/ms), np.array(mv.v[i]/mV), lw=2,
                color=["#45B7D1","#F39C12"][i], label=f"MBON {i} ({m_counts[i]} spikes)")
    ax3.axhline(y=-50, color="white", ls="--", alpha=0.3)
    ax3.set_ylabel("mV", color="white")
    predicted = int(np.argmax(m_counts)) if m_counts.sum() > 0 else "?"
    correct = "CORRECT" if (predicted == col) else "WRONG"
    ax3.set_title(f"MBON Output [{m_counts[0]},{m_counts[1]}] -> class {predicted} ({correct})",
                  color="#45B7D1" if correct=="CORRECT" else "red", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, facecolor="#0D1117", edgecolor="gray", labelcolor="white")
    ax3.tick_params(colors="gray")
    for sp in ax3.spines.values(): sp.set_color("#333")

ax_bottom = fig.add_subplot(gs[4, :])
ax_bottom.set_facecolor("#0D1117")
ax_bottom.axis("off")

h_set, v_set = all_kc_sets[0], all_kc_sets[1]
overlap = h_set & v_set
h_only = h_set - v_set
v_only = v_set - h_set
silent = set(range(200)) - (h_set | v_set)

color_map = np.array(["#1A1A3E"]*200)
for i in h_only: color_map[i] = "#4ECDC4"
for i in v_only: color_map[i] = "#F39C12"
for i in overlap: color_map[i] = "#FFD93D"

for i in range(200):
    row = i // 40
    c = i % 40
    rect = plt.Rectangle((c*0.5, 4-row*0.8), 0.45, 0.7,
                         facecolor=color_map[i], edgecolor="#333", lw=0.3)
    ax_bottom.add_patch(rect)

ax_bottom.set_xlim(-0.5, 20.5); ax_bottom.set_ylim(-0.5, 5.5)
ax_bottom.set_title(
    f"Evolved KC Population Map | H-only: {len(h_only)} (teal) | V-only: {len(v_only)} (orange) | "
    f"Both: {len(overlap)} (yellow) | Silent: {len(silent)} (dark)",
    color="white", fontsize=12
)

import matplotlib.patches as mpatches
leg = [mpatches.Patch(color="#4ECDC4", label=f"Horizontal only ({len(h_only)})"),
       mpatches.Patch(color="#F39C12", label=f"Vertical only ({len(v_only)})"),
       mpatches.Patch(color="#FFD93D", label=f"Both ({len(overlap)})"),
       mpatches.Patch(color="#1A1A3E", label=f"Silent ({len(silent)})")]
ax_bottom.legend(handles=leg, fontsize=9, facecolor="#0D1117", edgecolor="gray",
                labelcolor="white", loc="upper right")

fig.savefig("docs/figures/evolved_brain_live.png", dpi=200, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig)

summary = {
    "horizontal": {"pn_spikes": int(all_kc_sets[0] is not None), "kc_active": len(h_set)},
    "vertical": {"kc_active": len(v_set)},
    "overlap": len(overlap), "h_only": len(h_only), "v_only": len(v_only), "silent": len(silent),
}
with open("docs/journal/evolved_brain_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)
