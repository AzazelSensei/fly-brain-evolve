import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import yaml
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import (
    start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    SpikeGeneratorGroup, Network, defaultclock,
    ms, mV, nS, second,
)
from src.connectome.loader import generate_synthetic_connectome
from src.neurons.models import get_neuron_equations, get_neuron_params
from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)


def build_custom_network(config, connectome, kc_thresh_mV, pn_kc_weight_scale, apl_weight_nS):
    eqs = get_neuron_equations()

    pn_params = get_neuron_params("pn", config)
    kc_params = get_neuron_params("kc", config)
    mbon_params = get_neuron_params("mbon", config)
    apl_params = get_neuron_params("apl", config)

    pn = NeuronGroup(
        connectome["num_pn"], eqs,
        threshold="v > V_thresh", reset="v = V_reset",
        refractory=pn_params["refractory"] * second, method="euler",
        namespace={
            "tau_m": pn_params["tau_m"] * second,
            "V_rest": pn_params["V_rest"] * mV / mV * mV,
            "V_thresh": pn_params["V_thresh"] * mV / mV * mV,
            "V_reset": pn_params["V_reset"] * mV / mV * mV,
            "tau_exc": pn_params["tau_exc"] * second,
            "tau_inh": pn_params["tau_inh"] * second,
            "g_L": pn_params["g_L"] * nS / nS * nS,
            "E_exc": pn_params["E_exc"] * mV / mV * mV,
            "E_inh": pn_params["E_inh"] * mV / mV * mV,
        },
        name="pn",
    )
    pn.v = pn_params["V_rest"] * mV / mV * mV

    kc = NeuronGroup(
        connectome["num_kc"], eqs,
        threshold="v > V_thresh", reset="v = V_reset",
        refractory=kc_params["refractory"] * second, method="euler",
        namespace={
            "tau_m": kc_params["tau_m"] * second,
            "V_rest": kc_params["V_rest"] * mV / mV * mV,
            "V_thresh": kc_thresh_mV * mV,
            "V_reset": kc_params["V_reset"] * mV / mV * mV,
            "tau_exc": kc_params["tau_exc"] * second,
            "tau_inh": kc_params["tau_inh"] * second,
            "g_L": kc_params["g_L"] * nS / nS * nS,
            "E_exc": kc_params["E_exc"] * mV / mV * mV,
            "E_inh": kc_params["E_inh"] * mV / mV * mV,
        },
        name="kc",
    )
    kc.v = kc_params["V_rest"] * mV / mV * mV

    mbon = NeuronGroup(
        connectome["num_mbon"], eqs,
        threshold="v > V_thresh", reset="v = V_reset",
        refractory=mbon_params["refractory"] * second, method="euler",
        namespace={
            "tau_m": mbon_params["tau_m"] * second,
            "V_rest": mbon_params["V_rest"] * mV / mV * mV,
            "V_thresh": mbon_params["V_thresh"] * mV / mV * mV,
            "V_reset": mbon_params["V_reset"] * mV / mV * mV,
            "tau_exc": mbon_params["tau_exc"] * second,
            "tau_inh": mbon_params["tau_inh"] * second,
            "g_L": mbon_params["g_L"] * nS / nS * nS,
            "E_exc": mbon_params["E_exc"] * mV / mV * mV,
            "E_inh": mbon_params["E_inh"] * mV / mV * mV,
        },
        name="mbon",
    )
    mbon.v = mbon_params["V_rest"] * mV / mV * mV

    apl = NeuronGroup(
        1, eqs,
        threshold="v > V_thresh", reset="v = V_reset",
        refractory=apl_params["refractory"] * second, method="euler",
        namespace={
            "tau_m": apl_params["tau_m"] * second,
            "V_rest": apl_params["V_rest"] * mV / mV * mV,
            "V_thresh": apl_params["V_thresh"] * mV / mV * mV,
            "V_reset": apl_params["V_reset"] * mV / mV * mV,
            "tau_exc": apl_params["tau_exc"] * second,
            "tau_inh": apl_params["tau_inh"] * second,
            "g_L": apl_params["g_L"] * nS / nS * nS,
            "E_exc": apl_params["E_exc"] * mV / mV * mV,
            "E_inh": apl_params["E_inh"] * mV / mV * mV,
        },
        name="apl",
    )
    apl.v = apl_params["V_rest"] * mV / mV * mV

    pn_kc_matrix = connectome["pn_kc"]
    pn_indices, kc_indices = np.nonzero(pn_kc_matrix)
    pn_kc_syn = Synapses(pn, kc, "w_syn : 1", on_pre="g_exc_post += w_syn * nS", name="pn_kc_syn")
    pn_kc_syn.connect(i=pn_indices.astype(int), j=kc_indices.astype(int))
    pn_kc_syn.w_syn = pn_kc_matrix[pn_indices, kc_indices] * pn_kc_weight_scale

    kc_mbon_syn = Synapses(kc, mbon, "w : 1", on_pre="g_exc_post += w * nS", name="kc_mbon_syn")
    kc_all, mbon_all = np.nonzero(connectome["kc_mbon"])
    kc_mbon_syn.connect(i=kc_all.astype(int), j=mbon_all.astype(int))
    kc_mbon_syn.w = connectome["kc_mbon"][kc_all, mbon_all]

    kc_apl_syn = Synapses(kc, apl, on_pre="g_exc_post += 0.5*nS", name="kc_apl_syn")
    kc_apl_syn.connect()

    apl_kc_syn = Synapses(apl, kc,
                          on_pre="g_inh_post += {} * nS".format(apl_weight_nS),
                          name="apl_kc_syn")
    apl_kc_syn.connect()

    pn_mon = SpikeMonitor(pn, name="pn_mon")
    kc_mon = SpikeMonitor(kc, name="kc_mon")
    mbon_mon = SpikeMonitor(mbon, name="mbon_mon")

    net = Network(pn, kc, mbon, apl, pn_kc_syn, kc_mbon_syn, kc_apl_syn, apl_kc_syn,
                  pn_mon, kc_mon, mbon_mon)
    return net, {"pn": pn, "kc": kc, "mbon": mbon, "apl": apl,
                 "pn_monitor": pn_mon, "kc_monitor": kc_mon, "mbon_monitor": mbon_mon}


with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

thresholds_mV = [-50, -48, -45, -42, -40, -38, -35]
weight_scales = [1.0, 0.5, 0.3, 0.2, 0.1]
apl_w = 50.0

results = []
print("KC Threshold Sweep (APL=50nS, PN->KC scale varies)")
print("=" * 80)

for thresh in thresholds_mV:
    for w_scale in weight_scales:
        start_scope()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        net, comps = build_custom_network(config, connectome, thresh, w_scale, apl_w)

        pattern = make_horizontal_stripes(8)
        rates = image_to_rates(pattern, max_rate=100.0)
        full_rates = np.zeros(64)
        full_rates[:len(rates)] = rates
        indices, times = generate_poisson_spike_indices_and_times(full_rates, 0.1, seed=42)

        input_group = SpikeGeneratorGroup(64, indices, times * second, name="input")
        input_syn = Synapses(input_group, comps["pn"], on_pre="g_exc_post += 5*nS", name="input_syn")
        input_syn.connect("i == j")
        net.add(input_group)
        net.add(input_syn)

        defaultclock.dt = 0.1 * ms
        net.run(0.1 * second)

        kc_mon = comps["kc_monitor"]
        active = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0
        sparsity_pct = active / 200 * 100

        r = {
            "threshold_mV": thresh,
            "weight_scale": w_scale,
            "active_kcs": active,
            "sparsity_pct": round(sparsity_pct, 1),
            "kc_spikes": int(kc_mon.num_spikes),
            "mbon_spikes": int(comps["mbon_monitor"].num_spikes),
        }
        results.append(r)
        marker = " <-- TARGET" if 5 <= sparsity_pct <= 15 else ""
        print(f"  Vth={thresh:4d}mV  w_scale={w_scale:.1f}  | Active: {active:3d}/200 ({sparsity_pct:5.1f}%) | KC spikes: {int(kc_mon.num_spikes):5d}{marker}", flush=True)

with open("docs/journal/threshold_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("KC Threshold & PN-KC Weight Scale vs Sparsity", fontsize=14, fontweight="bold")

for w_scale in weight_scales:
    subset = [r for r in results if r["weight_scale"] == w_scale]
    x = [r["threshold_mV"] for r in subset]
    y = [r["sparsity_pct"] for r in subset]
    axes[0].plot(x, y, "o-", label=f"w_scale={w_scale}", markersize=6)

axes[0].axhspan(5, 15, alpha=0.15, color="green", label="Target (5-15%)")
axes[0].set_xlabel("KC Threshold (mV)")
axes[0].set_ylabel("KC Activation (%)")
axes[0].set_title("Sparsity vs Threshold")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()

for w_scale in weight_scales:
    subset = [r for r in results if r["weight_scale"] == w_scale]
    x = [r["threshold_mV"] for r in subset]
    y = [r["kc_spikes"] for r in subset]
    axes[1].plot(x, y, "s-", label=f"w_scale={w_scale}", markersize=6)

axes[1].set_xlabel("KC Threshold (mV)")
axes[1].set_ylabel("Total KC Spikes")
axes[1].set_title("KC Activity vs Threshold")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()

plt.tight_layout()
fig.savefig("docs/figures/threshold_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nThreshold sweep figure saved")
