import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from brian2 import *

prefs.codegen.target = "numpy"

start_scope()
eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""
ns = {
    "tau_m": 20*ms, "V_rest": -70*mV, "V_thresh": -50*mV, "V_reset": -70*mV,
    "tau_exc": 5*ms, "tau_inh": 10*ms, "g_L": 25*nS, "E_exc": 0*mV, "E_inh": -80*mV,
}

pn = NeuronGroup(6, eqs, threshold="v > V_thresh", reset="v = V_reset",
                 refractory=2*ms, method="euler", namespace=ns, name="pn")
pn.v = -70*mV

kc = NeuronGroup(1, eqs, threshold="v > V_thresh", reset="v = V_reset",
                 refractory=2*ms, method="euler", namespace=ns, name="kc")
kc.v = -70*mV

rng = np.random.default_rng(42)
idx_list, time_list = [], []
for i in range(6):
    n = rng.poisson(10)
    t = np.sort(rng.uniform(0, 0.1, n))
    idx_list.extend([i] * n)
    time_list.extend(t)
order = np.argsort(time_list)
idx_arr = np.array(idx_list, dtype=int)[order]
time_arr = np.array(time_list)[order]

inp = SpikeGeneratorGroup(6, idx_arr, time_arr * second, name="inp")
inp_syn = Synapses(inp, pn, on_pre="g_exc_post += 10*nS", name="inp_syn")
inp_syn.connect("i == j")

syn = Synapses(pn, kc, on_pre="g_exc_post += 10*nS", name="syn")
syn.connect()

pm = SpikeMonitor(pn, name="pm")
km = SpikeMonitor(kc, name="km")
kv = StateMonitor(kc, "v", record=True, name="kv")
pv = StateMonitor(pn, "v", record=[0], name="pv")

defaultclock.dt = 0.01*ms
run(100*ms)

result = {
    "pn_spikes": int(pm.num_spikes),
    "kc_spikes": int(km.num_spikes),
    "kc_peak_mV": round(float(kv.v[0].max() / mV), 2),
    "pn0_peak_mV": round(float(pv.v[0].max() / mV), 2),
}
with open("debug_single_result.json", "w") as f:
    json.dump(result, f, indent=2)
