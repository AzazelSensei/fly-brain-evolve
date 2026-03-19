import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from brian2 import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

prefs.codegen.target = "numpy"

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

def run_sim(apl_w_nS, pn_kc_w_range, kc_thresh_mV, input_w_nS=50, num_pn=32, num_kc=100, num_mbon=2):
    start_scope()
    defaultclock.dt = 0.05*ms
    rng = np.random.default_rng(42)

    ns_pn = {"tau_m": 10*ms, "V_rest": -70*mV, "V_thresh": -55*mV, "V_reset": -70*mV,
             "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV}
    ns_kc = {"tau_m": 20*ms, "V_rest": -70*mV, "V_thresh": kc_thresh_mV*mV, "V_reset": -70*mV,
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
        num_steps = int(0.098 / dt_val)
        spike_mask = rng.random(num_steps) < (100.0 * dt_val)
        spike_steps = np.where(spike_mask)[0]
        t = (spike_steps * dt_val) + 0.001
        idx_list.extend([pn_idx] * len(t))
        time_list.extend(t.tolist())
    order = np.argsort(time_list)
    idx_arr = np.array(idx_list, dtype=int)[order]
    time_arr = np.array(time_list)[order]

    inp = SpikeGeneratorGroup(num_pn, idx_arr, time_arr * second, name="inp")
    inp_syn = Synapses(inp, pn, on_pre="g_exc_post += %d*nS" % input_w_nS, name="inp_syn")
    inp_syn.connect("i == j")

    pn_kc_syn = Synapses(pn, kc, "w_s : 1", on_pre="g_exc_post += w_s*nS", name="pn_kc_syn")
    for kc_idx in range(num_kc):
        chosen = rng.choice(num_pn, size=6, replace=False)
        for p in chosen:
            pn_kc_syn.connect(i=int(p), j=int(kc_idx))
    pn_kc_syn.w_s = rng.uniform(pn_kc_w_range[0], pn_kc_w_range[1], size=len(pn_kc_syn))

    kc_mbon_syn = Synapses(kc, mb, "w_m : 1", on_pre="g_exc_post += w_m*nS", name="kc_mbon_syn")
    kc_mbon_syn.connect()
    kc_mbon_syn.w_m = rng.uniform(0.5, 2.0, size=len(kc_mbon_syn))

    kc_apl_syn = Synapses(kc, apl, on_pre="g_exc_post += 2*nS", name="kc_apl_syn")
    kc_apl_syn.connect()
    apl_kc_syn = Synapses(apl, kc, on_pre="g_inh_post += %d*nS" % apl_w_nS, name="apl_kc_syn")
    apl_kc_syn.connect()

    pn_mon = SpikeMonitor(pn, name="pm")
    kc_mon = SpikeMonitor(kc, name="km")
    mbon_mon = SpikeMonitor(mb, name="mm")
    apl_mon = SpikeMonitor(apl, name="am")

    run(100*ms)

    active = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0
    return {
        "pn_spikes": int(pn_mon.num_spikes),
        "kc_spikes": int(kc_mon.num_spikes),
        "kc_active": active,
        "kc_sparsity_pct": round(active / num_kc * 100, 1),
        "mbon_spikes": int(mbon_mon.num_spikes),
        "apl_spikes": int(apl_mon.num_spikes),
    }


configs = [
    {"apl_w": 50, "pn_kc_w": (3, 8), "kc_thresh": -50, "label": "baseline"},
    {"apl_w": 100, "pn_kc_w": (3, 8), "kc_thresh": -50, "label": "apl=100"},
    {"apl_w": 200, "pn_kc_w": (3, 8), "kc_thresh": -50, "label": "apl=200"},
    {"apl_w": 50, "pn_kc_w": (3, 8), "kc_thresh": -48, "label": "thresh=-48"},
    {"apl_w": 50, "pn_kc_w": (3, 8), "kc_thresh": -45, "label": "thresh=-45"},
    {"apl_w": 100, "pn_kc_w": (2, 5), "kc_thresh": -48, "label": "combo1"},
    {"apl_w": 200, "pn_kc_w": (2, 5), "kc_thresh": -48, "label": "combo2"},
    {"apl_w": 200, "pn_kc_w": (1, 3), "kc_thresh": -45, "label": "combo3"},
    {"apl_w": 300, "pn_kc_w": (1, 3), "kc_thresh": -45, "label": "combo4"},
    {"apl_w": 200, "pn_kc_w": (1, 3), "kc_thresh": -48, "label": "combo5"},
]

results = []
for cfg in configs:
    r = run_sim(cfg["apl_w"], cfg["pn_kc_w"], cfg["kc_thresh"])
    r["config"] = cfg["label"]
    r["apl_w_nS"] = cfg["apl_w"]
    r["pn_kc_w_range"] = list(cfg["pn_kc_w"])
    r["kc_thresh_mV"] = cfg["kc_thresh"]
    results.append(r)

with open("tune_sparsity_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, ax = plt.subplots(figsize=(12, 6))
labels = [r["config"] for r in results]
sparsities = [r["kc_sparsity_pct"] for r in results]
colors = ["green" if 5 <= s <= 15 else "orange" if 15 < s <= 30 else "red" for s in sparsities]
bars = ax.bar(labels, sparsities, color=colors, alpha=0.7, edgecolor="black")
ax.axhspan(5, 15, alpha=0.1, color="green", label="Target (5-15%)")
ax.set_ylabel("KC Activation (%)")
ax.set_title("Sparsity Tuning Results")
ax.legend()
for bar, s in zip(bars, sparsities):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{s}%",
            ha="center", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig("docs/figures/sparsity_tuning.png", dpi=200, bbox_inches="tight")
plt.close(fig)
