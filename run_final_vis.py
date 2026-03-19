import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from brian2 import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

prefs.codegen.target = "numpy"

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

def run_pattern(pattern_label, pn_active_mask, seed_offset=0):
    start_scope()
    defaultclock.dt = 0.05*ms
    rng = np.random.default_rng(42 + seed_offset)

    num_pn = 64
    num_kc = 200
    num_mbon = 2

    ns_pn = {"tau_m": 10*ms, "V_rest": -70*mV, "V_thresh": -55*mV, "V_reset": -70*mV,
             "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV}
    ns_kc = {"tau_m": 20*ms, "V_rest": -70*mV, "V_thresh": -45*mV, "V_reset": -70*mV,
             "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV}
    ns_mbon = {"tau_m": 15*ms, "V_rest": -70*mV, "V_thresh": -50*mV, "V_reset": -70*mV,
               "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV}
    ns_apl = {"tau_m": 10*ms, "V_rest": -70*mV, "V_thresh": -45*mV, "V_reset": -70*mV,
              "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV}

    pn = NeuronGroup(num_pn, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_pn, name="pn")
    pn.v = -70*mV
    kc = NeuronGroup(num_kc, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_kc, name="kc")
    kc.v = -70*mV
    mb = NeuronGroup(num_mbon, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_mbon, name="mbon")
    mb.v = -70*mV
    apl = NeuronGroup(1, eqs, threshold="v > V_thresh", reset="v = V_reset",
                      refractory=2*ms, method="euler", namespace=ns_apl, name="apl")
    apl.v = -70*mV

    dt_val = 0.05e-3
    idx_list, time_list = [], []
    for pn_idx in range(num_pn):
        if not pn_active_mask[pn_idx]:
            continue
        rate = 100.0
        num_steps = int(0.098 / dt_val)
        spike_mask = rng.random(num_steps) < (rate * dt_val)
        spike_steps = np.where(spike_mask)[0]
        t = (spike_steps * dt_val) + 0.001
        idx_list.extend([pn_idx] * len(t))
        time_list.extend(t.tolist())
    order = np.argsort(time_list)
    idx_arr = np.array(idx_list, dtype=int)[order]
    time_arr = np.array(time_list)[order]

    inp = SpikeGeneratorGroup(num_pn, idx_arr, time_arr * second, name="inp")
    inp_syn = Synapses(inp, pn, on_pre="g_exc_post += 50*nS", name="inp_syn")
    inp_syn.connect("i == j")

    pn_kc_syn = Synapses(pn, kc, "w_s : 1", on_pre="g_exc_post += w_s*nS", name="pn_kc_syn")
    conn_rng = np.random.default_rng(42)
    for kc_idx in range(num_kc):
        chosen = conn_rng.choice(num_pn, size=6, replace=False)
        for p in chosen:
            pn_kc_syn.connect(i=int(p), j=int(kc_idx))
    pn_kc_syn.w_s = conn_rng.uniform(1.5, 4.0, size=len(pn_kc_syn))

    kc_mbon_syn = Synapses(kc, mb, "w_m : 1", on_pre="g_exc_post += w_m*nS", name="kc_mbon_syn")
    kc_mbon_syn.connect()
    kc_mbon_syn.w_m = conn_rng.uniform(2.0, 5.0, size=len(kc_mbon_syn))

    kc_apl_syn = Synapses(kc, apl, on_pre="g_exc_post += 2*nS", name="kc_apl_syn")
    kc_apl_syn.connect()
    apl_kc_syn = Synapses(apl, kc, on_pre="g_inh_post += 200*nS", name="apl_kc_syn")
    apl_kc_syn.connect()

    pn_mon = SpikeMonitor(pn, name="pm")
    kc_mon = SpikeMonitor(kc, name="km")
    mbon_mon = SpikeMonitor(mb, name="mm")
    apl_mon = SpikeMonitor(apl, name="am")
    kc_v = StateMonitor(kc, "v", record=list(range(20)), name="kv")
    mbon_v = StateMonitor(mb, "v", record=True, name="mv")

    run(100*ms)

    active = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0
    kc_active_set = set(np.array(kc_mon.i)) if kc_mon.num_spikes > 0 else set()

    return {
        "pn_mon": pn_mon, "kc_mon": kc_mon, "mbon_mon": mbon_mon, "apl_mon": apl_mon,
        "kc_v": kc_v, "mbon_v": mbon_v,
        "stats": {
            "pn_spikes": int(pn_mon.num_spikes),
            "kc_spikes": int(kc_mon.num_spikes),
            "kc_active": active,
            "kc_sparsity_pct": round(active / num_kc * 100, 1),
            "mbon_spikes": int(mbon_mon.num_spikes),
            "apl_spikes": int(apl_mon.num_spikes),
        },
        "kc_active_set": kc_active_set,
    }


h_mask = np.zeros(64, dtype=bool)
for row in range(0, 8, 2):
    h_mask[row*8:(row+1)*8] = True

v_mask = np.zeros(64, dtype=bool)
for col in range(0, 8, 2):
    for row in range(8):
        v_mask[row*8 + col] = True

h_result = run_pattern("Horizontal", h_mask, seed_offset=0)
v_result = run_pattern("Vertical", v_mask, seed_offset=1)

overlap = h_result["kc_active_set"] & v_result["kc_active_set"]
h_only = h_result["kc_active_set"] - v_result["kc_active_set"]
v_only = v_result["kc_active_set"] - h_result["kc_active_set"]

summary = {
    "horizontal": h_result["stats"],
    "vertical": v_result["stats"],
    "kc_overlap": len(overlap),
    "kc_h_only": len(h_only),
    "kc_v_only": len(v_only),
}
with open("docs/journal/corrected_sim_results.json", "w") as f:
    json.dump(summary, f, indent=2)

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Drosophila Mushroom Body - Live Neural Activity",
             fontsize=20, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(5, 2, hspace=0.4, wspace=0.25,
                       left=0.06, right=0.96, top=0.93, bottom=0.04)

for col, (label, res, mask) in enumerate([
    ("Horizontal Stripes", h_result, h_mask),
    ("Vertical Stripes", v_result, v_mask),
]):
    ax0 = fig.add_subplot(gs[0, col])
    ax0.set_facecolor("#0D1117")
    pattern = mask.reshape(8, 8).astype(float)
    im_ax = ax0.inset_axes([0.0, 0.0, 0.25, 1.0])
    im_ax.imshow(pattern, cmap="inferno", interpolation="nearest")
    im_ax.set_xticks([]); im_ax.set_yticks([])

    pm = res["pn_mon"]
    if pm.num_spikes > 0:
        ax0.scatter(np.array(pm.t/ms), np.array(pm.i), s=2, c="#4ECDC4", marker="|", alpha=0.8)
    ax0.set_xlim(0, 100); ax0.set_ylabel("PN", color="white", fontsize=9)
    ax0.set_title(f"{label} - PN ({pm.num_spikes} spikes)", color="#4ECDC4", fontsize=11)
    ax0.tick_params(colors="gray")
    for sp in ax0.spines.values(): sp.set_color("gray")

    ax1 = fig.add_subplot(gs[1, col])
    ax1.set_facecolor("#0D1117")
    km = res["kc_mon"]
    if km.num_spikes > 0:
        ax1.scatter(np.array(km.t/ms), np.array(km.i), s=3, c="#FF6B6B", marker="|", alpha=0.9)
    s = res["stats"]
    ax1.set_xlim(0, 100); ax1.set_ylabel("KC", color="white", fontsize=9)
    ax1.set_title(f"KC: {s['kc_spikes']} spikes, {s['kc_active']}/200 active ({s['kc_sparsity_pct']}%)",
                  color="#FF6B6B", fontsize=11)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_color("gray")

    ax2 = fig.add_subplot(gs[2, col])
    ax2.set_facecolor("#0D1117")
    kv = res["kc_v"]
    colors_kc = plt.cm.plasma(np.linspace(0.2, 0.9, len(kv.v)))
    for i in range(len(kv.v)):
        ax2.plot(np.array(kv.t/ms), np.array(kv.v[i]/mV), color=colors_kc[i], alpha=0.6, lw=0.7)
    ax2.axhline(y=-45, color="white", ls="--", alpha=0.3, lw=0.8)
    ax2.set_ylabel("mV", color="white", fontsize=9)
    ax2.set_title("KC Voltage Traces (20 sample)", color="#FF6B6B", fontsize=10)
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values(): sp.set_color("gray")

    ax3 = fig.add_subplot(gs[3, col])
    ax3.set_facecolor("#0D1117")
    mv = res["mbon_v"]
    mbon_colors = ["#45B7D1", "#F39C12"]
    for i in range(2):
        ax3.plot(np.array(mv.t/ms), np.array(mv.v[i]/mV), color=mbon_colors[i], lw=1.5,
                label=f"MBON {i}")
    ax3.axhline(y=-50, color="white", ls="--", alpha=0.3, lw=0.8)
    ax3.set_ylabel("mV", color="white", fontsize=9)
    mm = res["mbon_mon"]
    m0 = np.sum(np.array(mm.i) == 0) if mm.num_spikes > 0 else 0
    m1 = np.sum(np.array(mm.i) == 1) if mm.num_spikes > 0 else 0
    ax3.set_title(f"MBON Output: [{m0}, {m1}] spikes", color="#45B7D1", fontsize=11)
    ax3.legend(fontsize=8, facecolor="#0D1117", edgecolor="gray", labelcolor="white")
    ax3.tick_params(colors="gray")
    for sp in ax3.spines.values(): sp.set_color("gray")

ax_bottom = fig.add_subplot(gs[4, :])
ax_bottom.set_facecolor("#0D1117")
ax_bottom.axis("off")

kc_ids = list(range(200))
h_active = np.array([1 if i in h_result["kc_active_set"] else 0 for i in kc_ids])
v_active = np.array([1 if i in v_result["kc_active_set"] else 0 for i in kc_ids])
both = h_active + v_active

color_map = {0: "#1A1A3E", 1: "#FF6B6B", 2: "#FFD93D"}
colors = [color_map.get(int(b), "#1A1A3E") for b in both]

for i, c in enumerate(colors):
    row = i // 40
    col_idx = i % 40
    rect = plt.Rectangle((col_idx * 0.5, 4 - row * 0.8), 0.45, 0.7,
                         facecolor=c, edgecolor="#333", lw=0.3)
    ax_bottom.add_patch(rect)

ax_bottom.set_xlim(-0.5, 20.5)
ax_bottom.set_ylim(-0.5, 5.5)
ax_bottom.set_title(
    f"KC Population Response Comparison | "
    f"H-only: {len(h_only)} (red) | V-only: {len(v_only)} (red) | Both: {len(overlap)} (yellow) | "
    f"Silent: {200 - len(h_result['kc_active_set'] | v_result['kc_active_set'])}",
    color="white", fontsize=11
)

import matplotlib.patches as mpatches
leg = [
    mpatches.Patch(color="#FF6B6B", label="Active (one pattern)"),
    mpatches.Patch(color="#FFD93D", label="Active (both patterns)"),
    mpatches.Patch(color="#1A1A3E", label="Silent"),
]
ax_bottom.legend(handles=leg, fontsize=9, facecolor="#0D1117", edgecolor="gray",
                labelcolor="white", loc="upper right")

fig.savefig("docs/figures/brain_live_activity.png", dpi=200, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig)
