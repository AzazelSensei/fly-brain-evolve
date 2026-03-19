import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import yaml
from src.connectome.loader import generate_synthetic_connectome


def plot_mushroom_body_circuit(config, connectome, save_path=None, figsize=(20, 14)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Drosophila Mushroom Body — Connectome Architecture", fontsize=16, fontweight="bold", y=0.98)

    _plot_circuit_diagram(axes[0, 0], config, connectome)
    _plot_connectivity_matrix(axes[0, 1], connectome)
    _plot_kc_input_distribution(axes[1, 0], connectome)
    _plot_network_graph(axes[1, 1], config, connectome)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


def _plot_circuit_diagram(ax, config, connectome):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mushroom Body Circuit Diagram", fontsize=13, fontweight="bold")

    layers = {
        "Input\n(Sensory)": (1, 5, "#E8E8E8", 0.8),
        f"PN\n({connectome['num_pn']})": (3, 5, "#4ECDC4", 1.2),
        f"KC\n({connectome['num_kc']})": (5.5, 5, "#FF6B6B", 1.8),
        f"MBON\n({connectome['num_mbon']})": (8.5, 5, "#45B7D1", 1.0),
    }

    for label, (x, y, color, radius) in layers.items():
        circle = plt.Circle((x, y), radius, color=color, alpha=0.3, ec=color, lw=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

    apl_x, apl_y = 5.5, 9
    apl_circle = plt.Circle((apl_x, apl_y), 0.6, color="#FFD93D", alpha=0.4, ec="#FFD93D", lw=2)
    ax.add_patch(apl_circle)
    ax.text(apl_x, apl_y, "APL\n(1)", ha="center", va="center", fontsize=9, fontweight="bold")

    arrow_props = dict(arrowstyle="->", lw=2, mutation_scale=15)

    ax.annotate("", xy=(2.7, 5), xytext=(1.9, 5), arrowprops=dict(**arrow_props, color="#4ECDC4"))
    ax.text(2.3, 5.5, "sensory\ninput", ha="center", va="bottom", fontsize=7, color="#4ECDC4")

    ax.annotate("", xy=(5.2, 5.3), xytext=(4.3, 5.3), arrowprops=dict(**arrow_props, color="#FF6B6B"))
    ax.text(4.75, 5.8, f"sparse\n~{config['connectome']['kc_pn_connections']}:1",
            ha="center", va="bottom", fontsize=7, color="#FF6B6B")

    ax.annotate("", xy=(8.0, 5), xytext=(7.3, 5), arrowprops=dict(**arrow_props, color="#45B7D1"))
    ax.text(7.65, 5.5, "plastic\n(STDP)", ha="center", va="bottom", fontsize=7, color="#45B7D1")

    ax.annotate("", xy=(5.5, 8.3), xytext=(5.5, 6.8),
                arrowprops=dict(arrowstyle="->", color="#FFD93D", lw=1.5, ls="--"))
    ax.text(6.1, 7.5, "excitation\n(KC→APL)", ha="left", va="center", fontsize=7, color="#666")

    ax.annotate("", xy=(5.5, 6.8), xytext=(5.5, 8.3),
                arrowprops=dict(arrowstyle="->", color="#FF0000", lw=1.5, ls="--"))
    ax.text(4.2, 7.5, "inhibition\n(APL→KC)", ha="center", va="center", fontsize=7, color="#FF0000")

    ax.text(5.5, -0.5,
            "Sparse coding: Each KC receives input from ~6 random PNs\n"
            "APL enforces winner-take-all competition among KCs",
            ha="center", va="top", fontsize=8, style="italic", color="#666")


def _plot_connectivity_matrix(ax, connectome):
    pn_kc = connectome["pn_kc"]
    binary = (pn_kc > 0).astype(float)
    ax.imshow(binary, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Kenyon Cells (KC)", fontsize=10)
    ax.set_ylabel("Projection Neurons (PN)", fontsize=10)
    ax.set_title("PN → KC Connectivity Matrix", fontsize=13, fontweight="bold")

    density = np.count_nonzero(pn_kc) / pn_kc.size * 100
    ax.text(0.98, 0.02, f"Density: {density:.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))


def _plot_kc_input_distribution(ax, connectome):
    pn_kc = connectome["pn_kc"]
    kc_inputs = np.count_nonzero(pn_kc, axis=0)
    kc_total_weight = pn_kc.sum(axis=0)

    ax.hist(kc_inputs, bins=range(0, max(kc_inputs) + 2), color="#FF6B6B", alpha=0.7,
            edgecolor="white", label="Input count")
    ax.set_xlabel("Number of PN inputs per KC", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("KC Input Distribution", fontsize=13, fontweight="bold")

    ax2 = ax.twinx()
    ax2.hist(kc_total_weight, bins=20, color="#45B7D1", alpha=0.4, label="Total weight")
    ax2.set_ylabel("Count (weight distribution)", fontsize=9, color="#45B7D1")

    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax.text(0.02, 0.85, f"Mean inputs: {kc_inputs.mean():.1f}\nMean weight: {kc_total_weight.mean():.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def _plot_network_graph(ax, config, connectome):
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.axis("off")
    ax.set_title("Network Topology (Sample)", fontsize=13, fontweight="bold")

    num_pn_show = min(20, connectome["num_pn"])
    num_kc_show = min(30, connectome["num_kc"])
    num_mbon_show = connectome["num_mbon"]

    pn_y = np.linspace(1, 9, num_pn_show)
    pn_x = np.full(num_pn_show, 1.0)

    kc_y = np.linspace(0.5, 9.5, num_kc_show)
    kc_x = np.full(num_kc_show, 5.0)

    mbon_y = np.linspace(3, 7, num_mbon_show)
    mbon_x = np.full(num_mbon_show, 9.0)

    pn_kc_sub = connectome["pn_kc"][:num_pn_show, :num_kc_show]
    lines = []
    colors = []
    for pn_idx in range(num_pn_show):
        for kc_idx in range(num_kc_show):
            if pn_kc_sub[pn_idx, kc_idx] > 0:
                lines.append([(pn_x[pn_idx], pn_y[pn_idx]), (kc_x[kc_idx], kc_y[kc_idx])])
                colors.append(pn_kc_sub[pn_idx, kc_idx])

    if lines:
        lc = LineCollection(lines, colors=plt.cm.Reds(np.array(colors) / max(colors)),
                           linewidths=0.3, alpha=0.4)
        ax.add_collection(lc)

    kc_mbon_sub = connectome["kc_mbon"][:num_kc_show, :num_mbon_show]
    lines2 = []
    for kc_idx in range(num_kc_show):
        for mbon_idx in range(num_mbon_show):
            if kc_mbon_sub[kc_idx, mbon_idx] > 0:
                lines2.append([(kc_x[kc_idx], kc_y[kc_idx]), (mbon_x[mbon_idx], mbon_y[mbon_idx])])

    if lines2:
        lc2 = LineCollection(lines2, colors="#45B7D1", linewidths=0.2, alpha=0.3)
        ax.add_collection(lc2)

    ax.scatter(pn_x, pn_y, s=30, c="#4ECDC4", zorder=5, edgecolors="white", linewidths=0.5)
    ax.scatter(kc_x, kc_y, s=15, c="#FF6B6B", zorder=5, edgecolors="white", linewidths=0.3)
    ax.scatter(mbon_x, mbon_y, s=80, c="#45B7D1", zorder=5, edgecolors="white", linewidths=1)

    apl_x, apl_y = 5.0, 11.0
    ax.scatter([apl_x], [apl_y], s=100, c="#FFD93D", zorder=5, edgecolors="black", linewidths=1, marker="D")

    ax.text(1.0, -1.2, f"PN ({num_pn_show})", ha="center", fontsize=9, color="#4ECDC4", fontweight="bold")
    ax.text(5.0, -1.2, f"KC ({num_kc_show})", ha="center", fontsize=9, color="#FF6B6B", fontweight="bold")
    ax.text(9.0, -1.2, f"MBON ({num_mbon_show})", ha="center", fontsize=9, color="#45B7D1", fontweight="bold")
    ax.text(5.0, 11.6, "APL", ha="center", fontsize=9, color="#FFD93D", fontweight="bold")

    legend_elements = [
        mpatches.Patch(color="#4ECDC4", label="Projection Neurons"),
        mpatches.Patch(color="#FF6B6B", label="Kenyon Cells"),
        mpatches.Patch(color="#45B7D1", label="MBON (Output)"),
        mpatches.Patch(color="#FFD93D", label="APL (Inhibitor)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)


def plot_input_patterns(save_path=None):
    from src.encoding.spike_encoder import make_horizontal_stripes, make_vertical_stripes

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Input Patterns & Spike Rate Encoding", fontsize=14, fontweight="bold")

    pattern_h = make_horizontal_stripes(8)
    pattern_v = make_vertical_stripes(8)

    axes[0].imshow(pattern_h, cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Pattern A: Horizontal Stripes", fontsize=10)
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")

    axes[1].imshow(pattern_v, cmap="gray_r", interpolation="nearest")
    axes[1].set_title("Pattern B: Vertical Stripes", fontsize=10)
    axes[1].set_xlabel("Pixel X")

    from src.encoding.spike_encoder import image_to_rates
    rates_h = image_to_rates(pattern_h, max_rate=100.0).reshape(8, 8)
    rates_v = image_to_rates(pattern_v, max_rate=100.0).reshape(8, 8)

    im2 = axes[2].imshow(rates_h, cmap="hot", interpolation="nearest", vmin=0, vmax=100)
    axes[2].set_title("Rate Encoding A (Hz)", fontsize=10)
    plt.colorbar(im2, ax=axes[2], label="Firing Rate (Hz)")

    im3 = axes[3].imshow(rates_v, cmap="hot", interpolation="nearest", vmin=0, vmax=100)
    axes[3].set_title("Rate Encoding B (Hz)", fontsize=10)
    plt.colorbar(im3, ax=axes[3], label="Firing Rate (Hz)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_spike_raster_comparison(config, connectome, save_path=None):
    from src.encoding.spike_encoder import (
        make_horizontal_stripes, make_vertical_stripes,
        image_to_rates, generate_poisson_spike_indices_and_times
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Spike Train Generation — Pattern Comparison", fontsize=14, fontweight="bold")

    patterns = [
        ("Horizontal", make_horizontal_stripes(8)),
        ("Vertical", make_vertical_stripes(8)),
    ]

    for col, (name, pattern) in enumerate(patterns):
        rates = image_to_rates(pattern, max_rate=100.0)
        indices, times = generate_poisson_spike_indices_and_times(rates, 0.1, seed=42 + col)

        axes[0, col].scatter(times * 1000, indices, s=1, c="black", marker="|")
        axes[0, col].set_ylabel("Neuron (PN) Index")
        axes[0, col].set_title(f"Poisson Spike Trains — {name}", fontsize=11)
        axes[0, col].set_xlim(0, 100)

        spike_counts = np.zeros(64)
        for i in indices:
            spike_counts[i] += 1
        axes[1, col].bar(range(64), spike_counts, color="#4ECDC4", alpha=0.7, width=1.0)
        axes[1, col].set_xlabel("Neuron (PN) Index")
        axes[1, col].set_ylabel("Spike Count")
        axes[1, col].set_title(f"Spike Count per Neuron — {name}", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    connectome = generate_synthetic_connectome(config["connectome"], seed=42)

    plot_mushroom_body_circuit(config, connectome, save_path="docs/figures/mushroom_body_architecture.png")
    plot_input_patterns(save_path="docs/figures/input_patterns.png")
    plot_spike_raster_comparison(config, connectome, save_path="docs/figures/spike_raster_comparison.png")
    print("All figures saved to docs/figures/")
