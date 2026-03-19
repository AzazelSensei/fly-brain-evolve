import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    SpikeGeneratorGroup, Network, defaultclock,
    ms, mV, nS, second, Hz,
)
from src.connectome.loader import generate_synthetic_connectome
from src.connectome.builder import MushroomBodyBuilder
from src.encoding.spike_encoder import (
    make_horizontal_stripes, make_vertical_stripes,
    image_to_rates, generate_poisson_spike_indices_and_times,
)


def run_and_visualize_simulation(config, pattern_name="horizontal", save_path=None):
    connectome = generate_synthetic_connectome(config["connectome"], seed=42)
    builder = MushroomBodyBuilder(config, connectome)
    net, components = builder.build_network(enable_stdp=False, enable_monitors=True)

    size = config["encoding"]["image_size"]
    if pattern_name == "horizontal":
        pattern = make_horizontal_stripes(size)
    else:
        pattern = make_vertical_stripes(size)

    max_rate = config["encoding"]["max_rate"]
    duration = 0.1
    dt = config["simulation"]["dt"]

    rates = image_to_rates(pattern, max_rate=max_rate)
    num_pn = connectome["num_pn"]
    full_rates = np.zeros(num_pn)
    full_rates[:len(rates)] = rates

    indices, times = generate_poisson_spike_indices_and_times(full_rates, duration, dt=dt, seed=42)

    input_group = SpikeGeneratorGroup(num_pn, indices, times * second, name="vis_input")
    input_syn = Synapses(input_group, components["pn"], on_pre="g_exc_post += 5*nS", name="vis_input_syn")
    input_syn.connect("i == j")
    net.add(input_group)
    net.add(input_syn)

    v_monitor_kc = StateMonitor(components["kc"], "v", record=list(range(min(10, connectome["num_kc"]))), name="v_mon_kc")
    v_monitor_mbon = StateMonitor(components["mbon"], "v", record=True, name="v_mon_mbon")
    net.add(v_monitor_kc)
    net.add(v_monitor_mbon)

    defaultclock.dt = 0.1 * ms
    net.run(duration * second)

    fig = _create_figure(
        pattern, pattern_name, components, v_monitor_kc, v_monitor_mbon,
        connectome, duration
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#1A1A2E")
        plt.close(fig)
    else:
        plt.show()
    return fig


def _create_figure(pattern, pattern_name, components, v_mon_kc, v_mon_mbon, connectome, duration):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(f"Brian2 Simulation Snapshot — {pattern_name.title()} Stripes Input",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.96, top=0.93, bottom=0.05)

    ax_pattern = fig.add_subplot(gs[0, 0])
    ax_pattern.set_facecolor("#1A1A2E")
    ax_pattern.imshow(pattern, cmap="inferno", interpolation="nearest")
    ax_pattern.set_title(f"Input Pattern: {pattern_name.title()}", color="white", fontsize=11)
    ax_pattern.tick_params(colors="gray")

    ax_pn = fig.add_subplot(gs[0, 1:])
    ax_pn.set_facecolor("#1A1A2E")
    pn_mon = components["pn_monitor"]
    if pn_mon.num_spikes > 0:
        ax_pn.scatter(np.array(pn_mon.t / ms), np.array(pn_mon.i),
                     s=1, c="#4ECDC4", marker="|", alpha=0.8)
    ax_pn.set_ylabel("PN Index", color="white", fontsize=9)
    ax_pn.set_title(f"Projection Neurons — {pn_mon.num_spikes} spikes", color="#4ECDC4", fontsize=11)
    ax_pn.set_xlim(0, duration * 1000)
    ax_pn.tick_params(colors="gray")
    ax_pn.spines["bottom"].set_color("gray")
    ax_pn.spines["left"].set_color("gray")
    ax_pn.spines["top"].set_visible(False)
    ax_pn.spines["right"].set_visible(False)

    ax_kc_raster = fig.add_subplot(gs[1, :2])
    ax_kc_raster.set_facecolor("#1A1A2E")
    kc_mon = components["kc_monitor"]
    if kc_mon.num_spikes > 0:
        ax_kc_raster.scatter(np.array(kc_mon.t / ms), np.array(kc_mon.i),
                            s=2, c="#FF6B6B", marker="|", alpha=0.9)
    active_kcs = len(set(np.array(kc_mon.i))) if kc_mon.num_spikes > 0 else 0
    sparsity = active_kcs / connectome["num_kc"] * 100
    ax_kc_raster.set_ylabel("KC Index", color="white", fontsize=9)
    ax_kc_raster.set_title(
        f"Kenyon Cells — {kc_mon.num_spikes} spikes, {active_kcs}/{connectome['num_kc']} active ({sparsity:.1f}%)",
        color="#FF6B6B", fontsize=11
    )
    ax_kc_raster.set_xlim(0, duration * 1000)
    ax_kc_raster.tick_params(colors="gray")
    ax_kc_raster.spines["bottom"].set_color("gray")
    ax_kc_raster.spines["left"].set_color("gray")
    ax_kc_raster.spines["top"].set_visible(False)
    ax_kc_raster.spines["right"].set_visible(False)

    ax_kc_hist = fig.add_subplot(gs[1, 2])
    ax_kc_hist.set_facecolor("#1A1A2E")
    if kc_mon.num_spikes > 0:
        kc_spike_counts = np.zeros(connectome["num_kc"])
        for idx in np.array(kc_mon.i):
            kc_spike_counts[idx] += 1
        ax_kc_hist.bar(range(connectome["num_kc"]), kc_spike_counts,
                       color="#FF6B6B", alpha=0.7, width=1.0)
    ax_kc_hist.set_xlabel("KC Index", color="white", fontsize=9)
    ax_kc_hist.set_ylabel("Spike Count", color="white", fontsize=9)
    ax_kc_hist.set_title("KC Activity Distribution", color="#FF6B6B", fontsize=10)
    ax_kc_hist.tick_params(colors="gray")
    ax_kc_hist.spines["bottom"].set_color("gray")
    ax_kc_hist.spines["left"].set_color("gray")
    ax_kc_hist.spines["top"].set_visible(False)
    ax_kc_hist.spines["right"].set_visible(False)

    ax_kc_v = fig.add_subplot(gs[2, :])
    ax_kc_v.set_facecolor("#1A1A2E")
    colors_kc = plt.cm.plasma(np.linspace(0.2, 0.9, len(v_mon_kc.v)))
    for i in range(len(v_mon_kc.v)):
        ax_kc_v.plot(np.array(v_mon_kc.t / ms), np.array(v_mon_kc.v[i] / mV),
                    color=colors_kc[i], alpha=0.7, lw=0.8, label=f"KC {i}" if i < 5 else None)
    ax_kc_v.axhline(y=-50, color="white", ls="--", alpha=0.3, lw=0.8)
    ax_kc_v.text(duration * 1000 * 0.99, -49, "threshold", color="white",
                alpha=0.5, fontsize=7, ha="right")
    ax_kc_v.set_xlabel("Time (ms)", color="white", fontsize=9)
    ax_kc_v.set_ylabel("Membrane Potential (mV)", color="white", fontsize=9)
    ax_kc_v.set_title("KC Membrane Voltage Traces (sample of 10)", color="#FF6B6B", fontsize=11)
    ax_kc_v.legend(fontsize=7, facecolor="#1A1A2E", edgecolor="gray",
                  labelcolor="white", loc="upper right", ncol=5)
    ax_kc_v.tick_params(colors="gray")
    ax_kc_v.spines["bottom"].set_color("gray")
    ax_kc_v.spines["left"].set_color("gray")
    ax_kc_v.spines["top"].set_visible(False)
    ax_kc_v.spines["right"].set_visible(False)

    ax_mbon = fig.add_subplot(gs[3, :2])
    ax_mbon.set_facecolor("#1A1A2E")
    mbon_mon = components["mbon_monitor"]
    if mbon_mon.num_spikes > 0:
        mbon_colors = ["#45B7D1", "#F39C12"]
        for idx in range(connectome["num_mbon"]):
            mask = np.array(mbon_mon.i) == idx
            if np.any(mask):
                ax_mbon.scatter(np.array(mbon_mon.t / ms)[mask], np.array(mbon_mon.i)[mask],
                               s=30, c=mbon_colors[idx % 2], marker="|", linewidths=2,
                               label=f"MBON {idx} ({np.sum(mask)} spikes)")
    ax_mbon.set_xlabel("Time (ms)", color="white", fontsize=9)
    ax_mbon.set_ylabel("MBON Index", color="white", fontsize=9)
    ax_mbon.set_title(f"MBON Output — {mbon_mon.num_spikes} total spikes", color="#45B7D1", fontsize=11)
    ax_mbon.set_xlim(0, duration * 1000)
    ax_mbon.legend(fontsize=9, facecolor="#1A1A2E", edgecolor="gray", labelcolor="white")
    ax_mbon.tick_params(colors="gray")
    ax_mbon.spines["bottom"].set_color("gray")
    ax_mbon.spines["left"].set_color("gray")
    ax_mbon.spines["top"].set_visible(False)
    ax_mbon.spines["right"].set_visible(False)

    ax_mbon_v = fig.add_subplot(gs[3, 2])
    ax_mbon_v.set_facecolor("#1A1A2E")
    mbon_colors = ["#45B7D1", "#F39C12"]
    for i in range(connectome["num_mbon"]):
        ax_mbon_v.plot(np.array(v_mon_mbon.t / ms), np.array(v_mon_mbon.v[i] / mV),
                      color=mbon_colors[i], alpha=0.9, lw=1.2, label=f"MBON {i}")
    ax_mbon_v.axhline(y=-50, color="white", ls="--", alpha=0.3, lw=0.8)
    ax_mbon_v.set_xlabel("Time (ms)", color="white", fontsize=9)
    ax_mbon_v.set_ylabel("Voltage (mV)", color="white", fontsize=9)
    ax_mbon_v.set_title("MBON Voltage Traces", color="#45B7D1", fontsize=10)
    ax_mbon_v.legend(fontsize=8, facecolor="#1A1A2E", edgecolor="gray", labelcolor="white")
    ax_mbon_v.tick_params(colors="gray")
    ax_mbon_v.spines["bottom"].set_color("gray")
    ax_mbon_v.spines["left"].set_color("gray")
    ax_mbon_v.spines["top"].set_visible(False)
    ax_mbon_v.spines["right"].set_visible(False)

    return fig


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    config["connectome"]["num_pn"] = 64
    config["connectome"]["num_kc"] = 200
    config["connectome"]["num_mbon"] = 2

    run_and_visualize_simulation(config, "horizontal", save_path="docs/figures/sim_snapshot_horizontal.png")
    print("Horizontal snapshot saved")

    run_and_visualize_simulation(config, "vertical", save_path="docs/figures/sim_snapshot_vertical.png")
    print("Vertical snapshot saved")
