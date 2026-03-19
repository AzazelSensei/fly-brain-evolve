import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as pe
import yaml
from src.connectome.loader import generate_synthetic_connectome


def plot_detailed_brain(config, connectome, save_path=None):
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor("#0A0A2E")

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_weights = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[1, 2])

    fig.suptitle("Drosophila Mushroom Body — Neural Circuit Detail",
                 fontsize=18, fontweight="bold", color="white", y=0.97)

    _draw_neural_network(ax_main, config, connectome)
    _draw_weight_distribution(ax_weights, connectome)
    _draw_circuit_stats(ax_stats, config, connectome)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#0A0A2E")
        plt.close(fig)
    else:
        plt.show()
    return fig


def _draw_neural_network(ax, config, connectome):
    ax.set_facecolor("#0A0A2E")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.axis("off")
    ax.set_title("Neural Circuit (subset view)", fontsize=14, color="white", pad=15)

    num_pn = min(30, connectome["num_pn"])
    num_kc = min(60, connectome["num_kc"])
    num_mbon = connectome["num_mbon"]

    pn_angles = np.linspace(np.pi * 0.7, np.pi * 0.3, num_pn)
    pn_x = 2.0 + 2.5 * np.cos(pn_angles)
    pn_y = 5.0 + 3.5 * np.sin(pn_angles)

    kc_x_center, kc_y_center = 5.5, 5.0
    kc_angles = np.linspace(0, 2 * np.pi, num_kc, endpoint=False)
    kc_radii = 1.5 + 0.5 * np.sin(kc_angles * 3)
    kc_x = kc_x_center + kc_radii * np.cos(kc_angles)
    kc_y = kc_y_center + kc_radii * np.sin(kc_angles)

    mbon_x = np.array([9.0, 9.0])
    mbon_y = np.array([6.5, 3.5])

    apl_x, apl_y = 5.5, 9.5

    pn_kc = connectome["pn_kc"][:num_pn, :num_kc]
    lines_pn_kc = []
    weights_pn_kc = []
    for i in range(num_pn):
        for j in range(num_kc):
            w = pn_kc[i, j]
            if w > 0:
                lines_pn_kc.append([(pn_x[i], pn_y[i]), (kc_x[j], kc_y[j])])
                weights_pn_kc.append(w)

    if lines_pn_kc:
        w_arr = np.array(weights_pn_kc)
        w_norm = w_arr / w_arr.max()
        colors = plt.cm.Greens(w_norm * 0.7 + 0.2)
        colors[:, 3] = w_norm * 0.3 + 0.05
        lc1 = LineCollection(lines_pn_kc, colors=colors, linewidths=0.4)
        ax.add_collection(lc1)

    kc_mbon = connectome["kc_mbon"][:num_kc, :num_mbon]
    lines_kc_mbon = []
    weights_kc_mbon = []
    for i in range(num_kc):
        for j in range(num_mbon):
            w = kc_mbon[i, j]
            if w > 0:
                lines_kc_mbon.append([(kc_x[i], kc_y[i]), (mbon_x[j], mbon_y[j])])
                weights_kc_mbon.append(w)

    if lines_kc_mbon:
        w_arr2 = np.array(weights_kc_mbon)
        w_norm2 = w_arr2 / w_arr2.max()
        colors2 = plt.cm.Blues(w_norm2 * 0.6 + 0.3)
        colors2[:, 3] = w_norm2 * 0.15 + 0.03
        lc2 = LineCollection(lines_kc_mbon, colors=colors2, linewidths=0.3)
        ax.add_collection(lc2)

    for i in range(min(num_kc, 20)):
        ax.plot([kc_x[i], apl_x], [kc_y[i], apl_y],
                color="#FFD93D", alpha=0.08, lw=0.3, ls="--")

    ax.scatter(pn_x, pn_y, s=40, c="#4ECDC4", zorder=10,
              edgecolors="#2ECC71", linewidths=0.5, alpha=0.9)

    ax.scatter(kc_x, kc_y, s=15, c="#FF6B6B", zorder=10,
              edgecolors="#E74C3C", linewidths=0.3, alpha=0.85)

    ax.scatter(mbon_x, mbon_y, s=200, c="#45B7D1", zorder=10,
              edgecolors="white", linewidths=1.5, marker="s")

    ax.scatter([apl_x], [apl_y], s=250, c="#FFD93D", zorder=10,
              edgecolors="white", linewidths=1.5, marker="D")

    text_props = dict(fontsize=10, fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=3, foreground="#0A0A2E")])

    ax.text(0.5, 8.5, f"PN\n({num_pn})", color="#4ECDC4", ha="center", **text_props)
    ax.text(5.5, 1.5, f"KC ({num_kc})", color="#FF6B6B", ha="center", **text_props)
    ax.text(10.0, 5.0, f"MBON\n({num_mbon})", color="#45B7D1", ha="center", **text_props)
    ax.text(apl_x, 10.3, "APL", color="#FFD93D", ha="center", **text_props)

    ax.text(3.5, 8.0, "sparse\nexcitatory", color="#2ECC71", fontsize=7,
            ha="center", alpha=0.8, style="italic")
    ax.text(7.5, 7.5, "plastic\n(STDP)", color="#45B7D1", fontsize=7,
            ha="center", alpha=0.8, style="italic")
    ax.text(6.5, 9.0, "global\ninhibition", color="#FFD93D", fontsize=7,
            ha="center", alpha=0.8, style="italic")

    ax.text(5.5, -0.5,
            f"Synapses shown: PN→KC ({len(lines_pn_kc)}), KC→MBON ({len(lines_kc_mbon)})",
            color="gray", fontsize=8, ha="center")


def _draw_weight_distribution(ax, connectome):
    ax.set_facecolor("#0A0A2E")

    pn_kc_weights = connectome["pn_kc"][connectome["pn_kc"] > 0]
    kc_mbon_weights = connectome["kc_mbon"].flatten()

    ax.hist(pn_kc_weights, bins=30, color="#4ECDC4", alpha=0.7,
            label=f"PN→KC (n={len(pn_kc_weights)})", density=True, edgecolor="#0A0A2E")
    ax.hist(kc_mbon_weights, bins=30, color="#45B7D1", alpha=0.5,
            label=f"KC→MBON (n={len(kc_mbon_weights)})", density=True, edgecolor="#0A0A2E")

    ax.set_xlabel("Synaptic Weight", color="white", fontsize=9)
    ax.set_ylabel("Density", color="white", fontsize=9)
    ax.set_title("Weight Distributions (Pre-Evolution)", color="white", fontsize=11, pad=10)
    ax.legend(fontsize=8, facecolor="#1A1A3E", edgecolor="gray", labelcolor="white")
    ax.tick_params(colors="gray")
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_circuit_stats(ax, config, connectome):
    ax.set_facecolor("#0A0A2E")
    ax.axis("off")
    ax.set_title("Circuit Statistics", color="white", fontsize=11, pad=10)

    pn_kc = connectome["pn_kc"]
    kc_mbon = connectome["kc_mbon"]
    num_pn = connectome["num_pn"]
    num_kc = connectome["num_kc"]
    num_mbon = connectome["num_mbon"]

    total_pn_kc_synapses = np.count_nonzero(pn_kc)
    total_kc_mbon_synapses = np.count_nonzero(kc_mbon)
    pn_kc_density = total_pn_kc_synapses / (num_pn * num_kc) * 100
    kc_inputs_per_kc = np.count_nonzero(pn_kc, axis=0).mean()
    pn_outputs_per_pn = np.count_nonzero(pn_kc, axis=1).mean()

    stats = [
        ("NEURONS", ""),
        ("  Projection Neurons (PN)", str(num_pn)),
        ("  Kenyon Cells (KC)", str(num_kc)),
        ("  Output Neurons (MBON)", str(num_mbon)),
        ("  APL Inhibitor", "1"),
        ("  Total", str(num_pn + num_kc + num_mbon + 1)),
        ("", ""),
        ("SYNAPSES", ""),
        ("  PN → KC", f"{total_pn_kc_synapses:,}"),
        ("  KC → MBON", f"{total_kc_mbon_synapses:,}"),
        ("  KC → APL", f"{num_kc:,}"),
        ("  APL → KC", f"{num_kc:,}"),
        ("  Total", f"{total_pn_kc_synapses + total_kc_mbon_synapses + 2*num_kc:,}"),
        ("", ""),
        ("CONNECTIVITY", ""),
        ("  PN→KC density", f"{pn_kc_density:.1f}%"),
        ("  KC inputs per KC", f"{kc_inputs_per_kc:.1f}"),
        ("  PN outputs per PN", f"{pn_outputs_per_pn:.1f}"),
        ("  KC→MBON", "all-to-all"),
        ("", ""),
        ("PLASTICITY", ""),
        ("  STDP type", "Anti-Hebbian"),
        ("  Plastic synapses", "KC → MBON"),
        ("  Learning bias", "LTD > LTP"),
    ]

    y_pos = 0.95
    for label, value in stats:
        if value == "" and label.isupper():
            ax.text(0.05, y_pos, label, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", color="#F39C12", family="monospace")
        elif label == "":
            pass
        else:
            ax.text(0.05, y_pos, label, transform=ax.transAxes,
                    fontsize=8, color="#CCC", family="monospace")
            ax.text(0.85, y_pos, value, transform=ax.transAxes,
                    fontsize=8, color="white", family="monospace", ha="right", fontweight="bold")
        y_pos -= 0.042


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    connectome = generate_synthetic_connectome(config["connectome"], seed=42)
    plot_detailed_brain(config, connectome, save_path="docs/figures/brain_detail.png")
    print("Brain detail figure saved")
