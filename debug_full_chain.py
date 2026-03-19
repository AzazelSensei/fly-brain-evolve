import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from brian2 import *

prefs.codegen.target = "numpy"

start_scope()
defaultclock.dt = 0.05*ms

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

ns_pn = {
    "tau_m": 10*ms, "V_rest": -70*mV, "V_thresh": -55*mV, "V_reset": -70*mV,
    "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV,
}
ns_kc = {
    "tau_m": 20*ms, "V_rest": -70*mV, "V_thresh": -50*mV, "V_reset": -70*mV,
    "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV,
}
ns_mbon = {
    "tau_m": 15*ms, "V_rest": -70*mV, "V_thresh": -50*mV, "V_reset": -70*mV,
    "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV,
}

num_pn = 32
num_kc = 100
num_mbon = 2

pn = NeuronGroup(num_pn, eqs, threshold="v > V_thresh", reset="v = V_reset",
                 refractory=2*ms, method="euler", namespace=ns_pn, name="pn")
pn.v = -70*mV

kc = NeuronGroup(num_kc, eqs, threshold="v > V_thresh", reset="v = V_reset",
                 refractory=2*ms, method="euler", namespace=ns_kc, name="kc")
kc.v = -70*mV

mbon = NeuronGroup(num_mbon, eqs, threshold="v > V_thresh", reset="v = V_reset",
                   refractory=2*ms, method="euler", namespace=ns_mbon, name="mbon")
mbon.v = -70*mV

apl = NeuronGroup(1, eqs, threshold="v > V_thresh", reset="v = V_reset",
                  refractory=2*ms, method="euler", namespace={
                      "tau_m": 10*ms, "V_rest": -70*mV, "V_thresh": -45*mV, "V_reset": -70*mV,
                      "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV,
                  }, name="apl")
apl.v = -70*mV

rng = np.random.default_rng(42)
dt_val = 0.05e-3
idx_list, time_list = [], []
for pn_idx in range(num_pn):
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
for kc_idx in range(num_kc):
    chosen = rng.choice(num_pn, size=6, replace=False)
    for pn_idx in chosen:
        pn_kc_syn.connect(i=int(pn_idx), j=int(kc_idx))
pn_kc_syn.w_s = rng.uniform(3, 8, size=len(pn_kc_syn))

kc_mbon_syn = Synapses(kc, mbon, "w_m : 1", on_pre="g_exc_post += w_m*nS", name="kc_mbon_syn")
kc_mbon_syn.connect()
kc_mbon_syn.w_m = rng.uniform(0.5, 2.0, size=len(kc_mbon_syn))

kc_apl_syn = Synapses(kc, apl, on_pre="g_exc_post += 2*nS", name="kc_apl_syn")
kc_apl_syn.connect()

apl_kc_syn = Synapses(apl, kc, on_pre="g_inh_post += 50*nS", name="apl_kc_syn")
apl_kc_syn.connect()

pn_mon = SpikeMonitor(pn, name="pn_mon")
kc_mon = SpikeMonitor(kc, name="kc_mon")
mbon_mon = SpikeMonitor(mbon, name="mbon_mon")
apl_mon = SpikeMonitor(apl, name="apl_mon")
kc_v_mon = StateMonitor(kc, "v", record=list(range(10)), name="kc_v_mon")
mbon_v_mon = StateMonitor(mbon, "v", record=True, name="mbon_v_mon")

run(100*ms)

active_kcs = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0

result = {
    "pn_spikes": int(pn_mon.num_spikes),
    "kc_spikes": int(kc_mon.num_spikes),
    "kc_active": active_kcs,
    "kc_total": num_kc,
    "kc_sparsity_pct": round(active_kcs / num_kc * 100, 1),
    "mbon_spikes": int(mbon_mon.num_spikes),
    "apl_spikes": int(apl_mon.num_spikes),
    "kc_peak_mV": round(float(kc_v_mon.v[:].max() / mV), 2),
}

with open("debug_full_chain_result.json", "w") as f:
    json.dump(result, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle("Mushroom Body Simulation - Corrected Units", fontsize=14, fontweight="bold")

if pn_mon.num_spikes > 0:
    axes[0].scatter(np.array(pn_mon.t/ms), np.array(pn_mon.i), s=2, c="#4ECDC4", marker="|")
axes[0].set_ylabel("PN Index")
axes[0].set_title(f"PN: {pn_mon.num_spikes} spikes")

if kc_mon.num_spikes > 0:
    axes[1].scatter(np.array(kc_mon.t/ms), np.array(kc_mon.i), s=2, c="#FF6B6B", marker="|")
axes[1].set_ylabel("KC Index")
axes[1].set_title(f"KC: {kc_mon.num_spikes} spikes, {active_kcs}/{num_kc} active ({active_kcs/num_kc*100:.1f}%)")

for i in range(min(10, num_kc)):
    axes[2].plot(np.array(kc_v_mon.t/ms), np.array(kc_v_mon.v[i]/mV), alpha=0.7, lw=0.8)
axes[2].axhline(y=-50, color="red", ls="--", alpha=0.5, label="threshold")
axes[2].set_ylabel("KC Voltage (mV)")
axes[2].set_title("KC Membrane Traces (10 sample)")
axes[2].legend()

for i in range(num_mbon):
    axes[3].plot(np.array(mbon_v_mon.t/ms), np.array(mbon_v_mon.v[i]/mV), lw=1.5, label=f"MBON {i}")
axes[3].axhline(y=-50, color="red", ls="--", alpha=0.5)
axes[3].set_ylabel("MBON Voltage (mV)")
axes[3].set_title(f"MBON: {mbon_mon.num_spikes} spikes")
axes[3].legend()

if apl_mon.num_spikes > 0:
    axes[4].scatter(np.array(apl_mon.t/ms), [0]*apl_mon.num_spikes, s=20, c="#FFD93D", marker="D")
axes[4].set_ylabel("APL")
axes[4].set_title(f"APL Inhibitor: {apl_mon.num_spikes} spikes")
axes[4].set_xlabel("Time (ms)")

plt.tight_layout()
fig.savefig("docs/figures/sim_corrected_chain.png", dpi=200, bbox_inches="tight")
plt.close(fig)
