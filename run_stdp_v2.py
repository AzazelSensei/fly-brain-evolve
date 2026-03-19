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
import yaml

prefs.codegen.target = "numpy"

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

from src.connectome.loader import generate_synthetic_connectome
from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)
from src.neurons.models import get_neuron_params

dt_val = config["simulation"]["dt"]
connectome = generate_synthetic_connectome(config["connectome"], seed=42)
patterns = [make_horizontal_stripes(8), make_vertical_stripes(8)]
labels = [0, 1]

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

def make_ns(params):
    return {"tau_m": params["tau_m"]*second, "V_rest": params["V_rest"]*volt,
            "V_thresh": params["V_thresh"]*volt, "V_reset": params["V_reset"]*volt,
            "tau_exc": params["tau_exc"]*second, "tau_inh": params["tau_inh"]*second,
            "g_L": params["g_L"]*siemens, "E_exc": params["E_exc"]*volt, "E_inh": params["E_inh"]*volt}


def build(kc_mbon_w, enable_stdp, pat, seed):
    ns_pn = make_ns(get_neuron_params("pn", config))
    ns_kc = make_ns(get_neuron_params("kc", config))
    ns_mb = make_ns(get_neuron_params("mbon", config))
    ns_ap = make_ns(get_neuron_params("apl", config))

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

    pi, ki = np.nonzero(connectome["pn_kc"][:64, :200])
    pk = Synapses(pn, kc, "ws:1", on_pre="g_exc_post+=ws*nS", name="pk")
    pk.connect(i=pi.astype(int), j=ki.astype(int))
    raw = connectome["pn_kc"][pi, ki]
    pk.ws = raw/raw.max()*2.5+1.5

    if enable_stdp:
        km = Synapses(kc, mb,
            "dApre/dt=-Apre/(20*ms):1 (event-driven)\ndApost/dt=-Apost/(20*ms):1 (event-driven)\nw:1",
            on_pre="g_exc_post+=w*nS\nApre+=1\nw=clip(w+Apost*(-0.012),0,15)",
            on_post="Apost+=1\nw=clip(w+Apre*0.01,0,15)",
            name="km")
    else:
        km = Synapses(kc, mb, "w:1", on_pre="g_exc_post+=w*nS", name="km")
    km.connect()
    km.w = kc_mbon_w

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

    kmon = SpikeMonitor(kc, name="kmon")
    mmon = SpikeMonitor(mb, name="mmon")
    net = Network(pn,kc,mb,ap,pk,km,ka,ak,inp,isyn,kmon,mmon)
    return net, km, kmon, mmon


rng = np.random.default_rng(42)
w = np.full(400, 8.0)
epochs = 30
results = []

for ep in range(epochs):
    for pi in rng.permutation(2):
        start_scope()
        net, km_syn, _, _ = build(w, True, patterns[pi], rng.integers(1e6))
        defaultclock.dt = dt_val*second
        net.run(0.1*second)
        w = np.array(km_syn.w).copy()

    ok, tot = 0, 0
    resp = []
    for pi in range(2):
        for tr in range(10):
            start_scope()
            _, _, kmon, mmon = build(w, False, patterns[pi], rng.integers(1e6))
            defaultclock.dt = dt_val*second
            run(0.1*second)
            c = np.array(mmon.count)
            kc_a = len(set(np.array(kmon.i))) if kmon.num_spikes > 0 else 0
            pred = int(np.argmax(c)) if c.sum()>0 else rng.integers(2)
            if pred == labels[pi]: ok += 1
            tot += 1
            resp.append({"p":int(pi),"m0":int(c[0]),"m1":int(c[1]),"kc":kc_a})

    acc = ok/tot
    results.append({"epoch":ep,"acc":round(acc,3),"w_mean":round(float(w.mean()),3),
                     "w_std":round(float(w.std()),3),"w_min":round(float(w.min()),3),
                     "w_max":round(float(w.max()),3),"resp":resp})

with open("docs/journal/stdp_v2_results.json","w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("STDP Learning V2 - Higher Initial Weights", fontsize=14, fontweight="bold")

ex = [r["epoch"] for r in results]
axes[0,0].plot(ex, [r["acc"] for r in results], "o-", color="#FF6B6B", lw=2)
axes[0,0].axhline(y=0.5, color="gray", ls="--", alpha=0.5)
axes[0,0].axhline(y=0.8, color="green", ls="--", alpha=0.5)
axes[0,0].set_ylabel("Accuracy"); axes[0,0].set_title("Test Accuracy")
axes[0,0].set_ylim(0, 1.05); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(ex, [r["w_mean"] for r in results], "s-", color="#45B7D1", lw=2, label="mean")
axes[0,1].plot(ex, [r["w_min"] for r in results], "--", color="#45B7D1", alpha=0.5, label="min")
axes[0,1].plot(ex, [r["w_max"] for r in results], "--", color="#F39C12", alpha=0.5, label="max")
axes[0,1].set_ylabel("Weight"); axes[0,1].set_title("KC-MBON Weights"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

last = results[-1]["resp"]
for pi, name in [(0,"Horizontal"),(1,"Vertical")]:
    m0s = [r["m0"] for r in last if r["p"]==pi]
    m1s = [r["m1"] for r in last if r["p"]==pi]
    x = np.arange(len(m0s))
    axes[1,pi].bar(x-0.15, m0s, 0.3, label="MBON0", color="#45B7D1", alpha=0.7)
    axes[1,pi].bar(x+0.15, m1s, 0.3, label="MBON1", color="#F39C12", alpha=0.7)
    axes[1,pi].set_xlabel("Trial"); axes[1,pi].set_ylabel("Spikes")
    axes[1,pi].set_title(f"{name} - MBON per trial"); axes[1,pi].legend(); axes[1,pi].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/stdp_v2_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
