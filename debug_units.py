import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from brian2 import *

results = []
for w in [1, 5, 10, 20, 50, 100, 200]:
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
    n = NeuronGroup(1, eqs, threshold="v > V_thresh", reset="v = V_reset",
                    refractory=2*ms, method="euler", namespace=ns)
    n.v = -70*mV
    inp = SpikeGeneratorGroup(1, [0], [5*ms])
    s = Synapses(inp, n, on_pre="g_exc_post += %d*nS" % w)
    s.connect()
    m = SpikeMonitor(n)
    vm = StateMonitor(n, "v", record=True)
    run(50*ms)
    peak = float(vm.v[0].max()/mV)
    fired = bool(m.num_spikes > 0)
    results.append({"w_nS": w, "fired": fired, "peak_mV": round(peak, 2)})

with open("debug_units_results.json", "w") as f:
    json.dump(results, f, indent=2)
