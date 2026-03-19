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
    start_scope, Synapses, SpikeGeneratorGroup, Network,
    defaultclock, ms, nS, second,
)
from src.connectome.loader import generate_synthetic_connectome
from src.connectome.builder import MushroomBodyBuilder
from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["connectome"]["num_pn"] = 64
config["connectome"]["num_kc"] = 200
config["connectome"]["num_mbon"] = 2

apl_weights = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
results = []

for apl_w in apl_weights:
    start_scope()
    connectome = generate_synthetic_connectome(config["connectome"], seed=42)
    builder = MushroomBodyBuilder(config, connectome)
    net, components = builder.build_network(enable_stdp=False, enable_monitors=True)

    net.remove(components["apl_kc_syn"])
    apl_kc_new = Synapses(
        components["apl"], components["kc"],
        on_pre="g_inh_post += {} * nS".format(apl_w),
        name="apl_kc_new",
    )
    apl_kc_new.connect()
    net.add(apl_kc_new)

    pattern = make_horizontal_stripes(8)
    rates = image_to_rates(pattern, max_rate=100.0)
    num_pn = connectome["num_pn"]
    full_rates = np.zeros(num_pn)
    full_rates[:len(rates)] = rates
    indices, times = generate_poisson_spike_indices_and_times(full_rates, 0.1, seed=42)

    input_group = SpikeGeneratorGroup(num_pn, indices, times * second, name="sweep_input")
    input_syn = Synapses(
        input_group, components["pn"],
        on_pre="g_exc_post += 5*nS",
        name="sweep_syn",
    )
    input_syn.connect("i == j")
    net.add(input_group)
    net.add(input_syn)

    defaultclock.dt = 0.1 * ms
    net.run(0.1 * second)

    kc_mon = components["kc_monitor"]
    active_kcs = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0
    sparsity = active_kcs / connectome["num_kc"]

    r = {
        "apl_weight_nS": apl_w,
        "active_kcs": active_kcs,
        "sparsity_pct": round(sparsity * 100, 1),
        "kc_spikes": int(kc_mon.num_spikes),
        "mbon_spikes": int(components["mbon_monitor"].num_spikes),
    }
    results.append(r)
    print(f"APL={apl_w:6.1f} nS | Active KC: {active_kcs:3d}/200 ({sparsity*100:5.1f}%) | KC spikes: {int(kc_mon.num_spikes):5d} | MBON: {int(components['mbon_monitor'].num_spikes):3d}", flush=True)

with open("docs/journal/sparsity_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

weights_list = [r["apl_weight_nS"] for r in results]
sparsities = [r["sparsity_pct"] for r in results]
kc_spikes_list = [r["kc_spikes"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("APL Inhibition Strength vs KC Sparsity", fontsize=14, fontweight="bold")

axes[0].semilogx(weights_list, sparsities, "o-", color="#FF6B6B", lw=2, markersize=8)
axes[0].axhspan(5, 15, alpha=0.15, color="green", label="Target (5-15%)")
axes[0].set_xlabel("APL→KC Weight (nS)")
axes[0].set_ylabel("KC Activation (%)")
axes[0].set_title("Sparsity vs Inhibition")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogx(weights_list, kc_spikes_list, "s-", color="#4ECDC4", lw=2, markersize=8)
axes[1].set_xlabel("APL→KC Weight (nS)")
axes[1].set_ylabel("Total KC Spikes")
axes[1].set_title("KC Activity vs Inhibition")
axes[1].grid(True, alpha=0.3)

mbon_list = [r["mbon_spikes"] for r in results]
axes[2].semilogx(weights_list, mbon_list, "^-", color="#45B7D1", lw=2, markersize=8)
axes[2].set_xlabel("APL→KC Weight (nS)")
axes[2].set_ylabel("Total MBON Spikes")
axes[2].set_title("MBON Output vs Inhibition")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("docs/figures/sparsity_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Sweep figure saved", flush=True)
