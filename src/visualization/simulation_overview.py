import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def plot_simulation_pipeline(save_path=None):
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_title("EvoDrosophila — Full Simulation & Evolution Pipeline",
                 fontsize=18, fontweight="bold", pad=20)

    _draw_input_stage(ax)
    _draw_mushroom_body(ax)
    _draw_learning_stage(ax)
    _draw_evolution_loop(ax)
    _draw_output_stage(ax)
    _draw_flow_arrows(ax)
    _draw_legend(ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


def _rounded_box(ax, x, y, w, h, color, text, fontsize=9, alpha=0.3, textcolor="black", bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=textcolor)


def _draw_input_stage(ax):
    _rounded_box(ax, 0.5, 11, 4, 3.5, "#E8E8E8", "", alpha=0.15)
    ax.text(2.5, 14.2, "INPUT STAGE", ha="center", fontsize=11, fontweight="bold", color="#555")

    ax.text(1.2, 13.5, "8x8 Binary\nPatterns", ha="center", fontsize=8, color="#333")

    pattern = np.zeros((8, 8))
    for r in range(0, 8, 2):
        pattern[r, :] = 1
    ax_inset1 = ax.inset_axes([0.025, 0.78, 0.06, 0.06])
    ax_inset1.imshow(pattern, cmap="gray_r", interpolation="nearest")
    ax_inset1.set_xticks([])
    ax_inset1.set_yticks([])
    ax_inset1.set_title("A", fontsize=7)

    pattern_v = np.zeros((8, 8))
    for c in range(0, 8, 2):
        pattern_v[:, c] = 1
    ax_inset2 = ax.inset_axes([0.095, 0.78, 0.06, 0.06])
    ax_inset2.imshow(pattern_v, cmap="gray_r", interpolation="nearest")
    ax_inset2.set_xticks([])
    ax_inset2.set_yticks([])
    ax_inset2.set_title("B", fontsize=7)

    _rounded_box(ax, 1.0, 11.3, 3.0, 1.2, "#FFF3CD", "Rate Coding\npixel → Hz\n(0-100 Hz Poisson)", fontsize=8)


def _draw_mushroom_body(ax):
    _rounded_box(ax, 5.5, 9, 7.5, 6, "#F0F0FF", "", alpha=0.15)
    ax.text(9.25, 14.7, "MUSHROOM BODY NETWORK (Brian2)", ha="center",
            fontsize=12, fontweight="bold", color="#333")

    _rounded_box(ax, 6.0, 13.0, 2.5, 1.2, "#4ECDC4",
                 "Projection Neurons\n(PN: 150)\nτ=10ms, Vth=-55mV", fontsize=8, alpha=0.4)

    _rounded_box(ax, 6.0, 10.8, 2.5, 1.5, "#FF6B6B",
                 "Kenyon Cells\n(KC: 200-2000)\nτ=20ms, Vth=-50mV\nSPARSE CODING", fontsize=8, alpha=0.4)

    _rounded_box(ax, 6.0, 9.3, 2.5, 1.0, "#45B7D1",
                 "MBON (Output: 2)\nτ=15ms, Vth=-50mV", fontsize=8, alpha=0.4)

    _rounded_box(ax, 9.5, 11.5, 2.5, 1.5, "#FFD93D",
                 "APL Inhibitor\n(1 neuron)\nGlobal → KC\nWinner-take-all", fontsize=8, alpha=0.5)

    ax.annotate("", xy=(8.5, 12.5), xytext=(7.25, 13.0),
                arrowprops=dict(arrowstyle="-|>", color="#FF6B6B", lw=1.5))
    ax.text(7.5, 13.1, "sparse\n~6:1", fontsize=7, color="#FF6B6B", ha="center")

    ax.annotate("", xy=(7.25, 10.3), xytext=(7.25, 10.8),
                arrowprops=dict(arrowstyle="-|>", color="#45B7D1", lw=1.5))
    ax.text(7.9, 10.5, "STDP\n(plastic)", fontsize=7, color="#45B7D1", ha="center")

    ax.annotate("", xy=(9.5, 11.8), xytext=(8.5, 11.5),
                arrowprops=dict(arrowstyle="-|>", color="#FFD93D", lw=1.2, ls="--"))

    ax.annotate("", xy=(8.5, 11.2), xytext=(9.5, 11.5),
                arrowprops=dict(arrowstyle="-|>", color="#FF0000", lw=1.2, ls="--"))
    ax.text(9.0, 10.8, "inhibition", fontsize=6, color="#FF0000", ha="center")

    _rounded_box(ax, 9.5, 9.5, 3.0, 1.5, "#E8DAEF",
                 "DAN\n(Dopaminergic)\nReward signal\n→ STDP modulation", fontsize=8, alpha=0.4)
    ax.annotate("", xy=(8.5, 10.0), xytext=(9.5, 10.2),
                arrowprops=dict(arrowstyle="-|>", color="#8E44AD", lw=1.2, ls="-."))


def _draw_learning_stage(ax):
    _rounded_box(ax, 14, 11, 4, 4, "#E8F8F5", "", alpha=0.15)
    ax.text(16, 14.7, "LEARNING & EVALUATION", ha="center",
            fontsize=11, fontweight="bold", color="#555")

    _rounded_box(ax, 14.3, 13.3, 3.4, 1.2, "#2ECC71",
                 "Training Phase\nSTDP ON | N epochs\nPresent all stimuli", fontsize=8, alpha=0.35)

    _rounded_box(ax, 14.3, 11.5, 3.4, 1.2, "#3498DB",
                 "Test Phase\nSTDP OFF | K trials\nMeasure accuracy", fontsize=8, alpha=0.35)

    ax.annotate("", xy=(16, 12.7), xytext=(16, 13.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))


def _draw_evolution_loop(ax):
    _rounded_box(ax, 0.5, 1, 21, 7.5, "#FFF5F5", "", alpha=0.1)
    ax.text(11, 8.2, "NEUROEVOLUTION PIPELINE", ha="center",
            fontsize=13, fontweight="bold", color="#C0392B")

    _rounded_box(ax, 1, 5.5, 3, 2, "#E74C3C",
                 "Population\n(50 genomes)\nEach = full\nconnectome", fontsize=8, alpha=0.3, bold=True)

    _rounded_box(ax, 5, 5.5, 3, 2, "#F39C12",
                 "Fitness\nEvaluation\nBrian2 sim per\ngenome", fontsize=8, alpha=0.3)

    _rounded_box(ax, 9, 6, 3.5, 1.5, "#27AE60",
                 "Selection\nTournament (k=3)\n+ Elitism (top 5)", fontsize=8, alpha=0.3)

    _rounded_box(ax, 9, 4, 3.5, 1.5, "#8E44AD",
                 "Crossover\nKC-based uniform\nP=0.3", fontsize=8, alpha=0.3)

    _rounded_box(ax, 13.5, 5.5, 4, 2, "#E67E22",
                 "Mutation Operators\n• add/remove synapse\n• mutate weights\n• mutate threshold/tau\n• rewire connections\n• add/remove KC",
                 fontsize=7, alpha=0.3)

    _rounded_box(ax, 18.5, 5.5, 2.5, 2, "#1ABC9C",
                 "New\nGeneration\n→ repeat", fontsize=8, alpha=0.3, bold=True)

    arrow_kw = dict(arrowstyle="-|>", color="#C0392B", lw=2)
    ax.annotate("", xy=(5, 6.5), xytext=(4, 6.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(9, 6.75), xytext=(8, 6.75), arrowprops=arrow_kw)
    ax.annotate("", xy=(9, 4.75), xytext=(9, 5.5),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5))
    ax.annotate("", xy=(13.5, 6.5), xytext=(12.5, 5.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(18.5, 6.5), xytext=(17.5, 6.5), arrowprops=arrow_kw)

    ax.annotate("", xy=(2.5, 7.5), xytext=(19.75, 7.5),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2.5,
                               connectionstyle="arc3,rad=0.15"))
    ax.text(11, 7.85, "generation loop (100 generations)", ha="center",
            fontsize=9, fontweight="bold", color="#C0392B")

    _rounded_box(ax, 1, 1.5, 6, 2.5, "#D5F5E3",
                 "Fitness Function\n\nF = accuracy + λ₁·sparsity_bonus − λ₂·complexity_penalty\n\n"
                 "accuracy: correct classification rate (0-1)\n"
                 "sparsity: KC activation near 10% target\n"
                 "complexity: penalize excess synapses",
                 fontsize=7, alpha=0.4)

    _rounded_box(ax, 8, 1.5, 5.5, 2.5, "#FADBD8",
                 "Genome Encoding\n\n"
                 "• PN→KC adjacency matrix (sparse)\n"
                 "• KC→MBON weight matrix (dense, plastic)\n"
                 "• KC threshold vector (per-neuron)\n"
                 "• KC time constant vector (per-neuron)\n"
                 "• APL connection weights",
                 fontsize=7, alpha=0.4)

    _rounded_box(ax, 14.5, 1.5, 6.5, 2.5, "#D6EAF8",
                 "Expected Outcome\n\n"
                 "• Evolution discovers optimal PN→KC wiring\n"
                 "• KC threshold tuning for pattern discrimination\n"
                 "• STDP learns differential MBON responses\n"
                 "• Sparse coding is maintained (~10% KC active)\n"
                 "• Target: >80% binary classification accuracy",
                 fontsize=7, alpha=0.4)


def _draw_output_stage(ax):
    _rounded_box(ax, 18.5, 11, 3, 4, "#FDEBD0", "", alpha=0.15)
    ax.text(20, 14.7, "OUTPUT", ha="center", fontsize=11, fontweight="bold", color="#555")

    _rounded_box(ax, 18.8, 13.3, 2.4, 1.2, "#F1948A",
                 "MBON Spike\nCounts\nargmax → class", fontsize=8, alpha=0.4)

    _rounded_box(ax, 18.8, 11.5, 2.4, 1.2, "#AED6F1",
                 "Accuracy\nSparsity\nFitness Score", fontsize=8, alpha=0.4)

    ax.annotate("", xy=(20, 12.7), xytext=(20, 13.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))


def _draw_flow_arrows(ax):
    ax.annotate("", xy=(5.5, 13.5), xytext=(4.5, 12.5),
                arrowprops=dict(arrowstyle="-|>", color="#4ECDC4", lw=2.5))
    ax.text(4.5, 13.3, "spike\ntrains", fontsize=7, color="#4ECDC4", ha="center")

    ax.annotate("", xy=(14, 13.9), xytext=(13, 13.5),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    ax.annotate("", xy=(18.5, 13.9), xytext=(17.7, 13.9),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    ax.annotate("", xy=(6.5, 9), xytext=(6.5, 7.5),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2, ls="--"))
    ax.text(7.5, 8.3, "genome → network\n(each evaluation)", fontsize=7,
            color="#C0392B", ha="center", style="italic")


def _draw_legend(ax):
    legend_elements = [
        mpatches.Patch(color="#4ECDC4", alpha=0.4, label="Projection Neurons (PN)"),
        mpatches.Patch(color="#FF6B6B", alpha=0.4, label="Kenyon Cells (KC)"),
        mpatches.Patch(color="#45B7D1", alpha=0.4, label="Mushroom Body Output (MBON)"),
        mpatches.Patch(color="#FFD93D", alpha=0.5, label="APL Global Inhibitor"),
        mpatches.Patch(color="#E8DAEF", alpha=0.4, label="Dopaminergic Neurons (DAN)"),
        mpatches.Patch(color="#C0392B", alpha=0.3, label="Evolution Pipeline"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
             framealpha=0.9, title="Components", title_fontsize=9,
             bbox_to_anchor=(0.0, 1.0))


if __name__ == "__main__":
    plot_simulation_pipeline(save_path="docs/figures/simulation_pipeline.png")
    print("Simulation pipeline figure saved")
