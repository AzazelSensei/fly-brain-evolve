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
from src.neurons.plasticity import get_stdp_params

dt_val = config["simulation"]["dt"]
connectome = generate_synthetic_connectome(config["connectome"], seed=42)
pattern_h = make_horizontal_stripes(8)
pattern_v = make_vertical_stripes(8)
patterns = [pattern_h, pattern_v]
labels = [0, 1]

eqs = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""


def ns(params):
    return {
        "tau_m": params["tau_m"]*second, "V_rest": params["V_rest"]*volt,
        "V_thresh": params["V_thresh"]*volt, "V_reset": params["V_reset"]*volt,
        "tau_exc": params["tau_exc"]*second, "tau_inh": params["tau_inh"]*second,
        "g_L": params["g_L"]*siemens, "E_exc": params["E_exc"]*volt, "E_inh": params["E_inh"]*volt,
    }


def build_fresh(kc_mbon_weights, enable_stdp, input_pattern, input_seed):
    ns_pn = ns(get_neuron_params("pn", config))
    ns_kc = ns(get_neuron_params("kc", config))
    ns_mbon = ns(get_neuron_params("mbon", config))
    ns_apl = ns(get_neuron_params("apl", config))

    pn = NeuronGroup(64, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_pn, name="pn")
    pn.v = -70*mV
    kc = NeuronGroup(200, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_kc, name="kc")
    kc.v = -70*mV
    mb = NeuronGroup(2, eqs, threshold="v > V_thresh", reset="v = V_reset",
                     refractory=2*ms, method="euler", namespace=ns_mbon, name="mbon")
    mb.v = -70*mV
    apl_n = NeuronGroup(1, eqs, threshold="v > V_thresh", reset="v = V_reset",
                        refractory=2*ms, method="euler", namespace=ns_apl, name="apl")
    apl_n.v = -70*mV

    pi, ki = np.nonzero(connectome["pn_kc"][:64, :200])
    pn_kc = Synapses(pn, kc, "w_s : 1", on_pre="g_exc_post += w_s*nS", name="pn_kc")
    pn_kc.connect(i=pi.astype(int), j=ki.astype(int))
    raw = connectome["pn_kc"][pi, ki]
    pn_kc.w_s = raw / raw.max() * 2.5 + 1.5

    stdp_p = get_stdp_params(config)
    if enable_stdp:
        kc_mb = Synapses(kc, mb,
            "dApre/dt = -Apre/tau_pre : 1 (event-driven)\ndApost/dt = -Apost/tau_post : 1 (event-driven)\nw : 1",
            on_pre="g_exc_post += w*nS\nApre += 1\nw = clip(w + Apost*A_post, w_min, w_max)",
            on_post="Apost += 1\nw = clip(w + Apre*A_pre, w_min, w_max)",
            namespace={"tau_pre": stdp_p["tau_pre"]*second, "tau_post": stdp_p["tau_post"]*second,
                       "A_pre": stdp_p["A_pre"], "A_post": stdp_p["A_post"], "w_max": 10.0, "w_min": 0.0},
            name="kc_mb")
    else:
        kc_mb = Synapses(kc, mb, "w : 1", on_pre="g_exc_post += w*nS", name="kc_mb")
    kc_mb.connect()
    kc_mb.w = kc_mbon_weights

    kc_apl = Synapses(kc, apl_n, on_pre="g_exc_post += 2*nS", name="kc_apl")
    kc_apl.connect()
    apl_kc = Synapses(apl_n, kc, on_pre="g_inh_post += 200*nS", name="apl_kc")
    apl_kc.connect()

    rates = image_to_rates(input_pattern, max_rate=100.0)
    full_rates = np.zeros(64)
    full_rates[:len(rates)] = rates
    indices, times = generate_poisson_spike_indices_and_times(full_rates, 0.1, dt=dt_val, seed=input_seed)

    inp = SpikeGeneratorGroup(64, indices, times*second, name="inp")
    inp_syn = Synapses(inp, pn, on_pre="g_exc_post += 50*nS", name="isyn")
    inp_syn.connect("i == j")

    km = SpikeMonitor(kc, name="km")
    mm = SpikeMonitor(mb, name="mm")

    net = Network(pn, kc, mb, apl_n, pn_kc, kc_mb, kc_apl, apl_kc, inp, inp_syn, km, mm)
    return net, kc_mb, km, mm


rng = np.random.default_rng(42)
kc_mbon_w = np.full(200 * 2, 3.0)

training_epochs = 20
epoch_results = []

for epoch in range(training_epochs):
    start_scope()

    order = rng.permutation(len(patterns))
    for p_idx in order:
        start_scope()
        net, kc_mb_syn, _, _ = build_fresh(kc_mbon_w, True, patterns[p_idx], rng.integers(1e6))
        defaultclock.dt = dt_val * second
        net.run(0.1 * second)
        kc_mbon_w = np.array(kc_mb_syn.w).copy()

    correct = 0
    total = 0
    responses = []
    for p_idx in range(len(patterns)):
        for trial in range(5):
            start_scope()
            net, _, km, mm = build_fresh(kc_mbon_w, False, patterns[p_idx], rng.integers(1e6))
            defaultclock.dt = dt_val * second
            net.run(0.1 * second)

            counts = np.array(mm.count)
            kc_active = len(set(np.array(km.i))) if km.num_spikes > 0 else 0
            if counts.sum() > 0:
                predicted = int(np.argmax(counts))
            else:
                predicted = rng.integers(2)
            if predicted == labels[p_idx]:
                correct += 1
            total += 1
            responses.append({"p": int(p_idx), "m0": int(counts[0]), "m1": int(counts[1]), "kc": kc_active})

    accuracy = correct / total
    r = {
        "epoch": epoch, "accuracy": round(accuracy, 3),
        "w_mean": round(float(kc_mbon_w.mean()), 4),
        "w_std": round(float(kc_mbon_w.std()), 4),
        "w_min": round(float(kc_mbon_w.min()), 4),
        "w_max": round(float(kc_mbon_w.max()), 4),
        "responses": responses,
    }
    epoch_results.append(r)

with open("docs/journal/baseline_stdp_results.json", "w") as f:
    json.dump(epoch_results, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Baseline STDP Learning - Binary Classification", fontsize=14, fontweight="bold")

ex = [r["epoch"] for r in epoch_results]
acc = [r["accuracy"] for r in epoch_results]
axes[0].plot(ex, acc, "o-", color="#FF6B6B", lw=2, markersize=6)
axes[0].axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance (50%)")
axes[0].axhline(y=0.8, color="green", ls="--", alpha=0.5, label="Target (80%)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Test Accuracy Over Training")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

wm = [r["w_mean"] for r in epoch_results]
ws = [r["w_std"] for r in epoch_results]
axes[1].plot(ex, wm, "s-", color="#45B7D1", lw=2)
axes[1].fill_between(ex, np.array(wm)-np.array(ws), np.array(wm)+np.array(ws), alpha=0.2, color="#45B7D1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Weight")
axes[1].set_title("KC-MBON Weight Distribution")
axes[1].grid(True, alpha=0.3)

last = epoch_results[-1]["responses"]
h_m0 = np.mean([r["m0"] for r in last if r["p"] == 0])
h_m1 = np.mean([r["m1"] for r in last if r["p"] == 0])
v_m0 = np.mean([r["m0"] for r in last if r["p"] == 1])
v_m1 = np.mean([r["m1"] for r in last if r["p"] == 1])
x = np.arange(2)
w = 0.35
axes[2].bar(x - w/2, [h_m0, v_m0], w, label="MBON 0", color="#45B7D1", alpha=0.7)
axes[2].bar(x + w/2, [h_m1, v_m1], w, label="MBON 1", color="#F39C12", alpha=0.7)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["Horizontal", "Vertical"])
axes[2].set_ylabel("Mean Spike Count")
axes[2].set_title("MBON Response (Final Epoch)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/baseline_stdp_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
