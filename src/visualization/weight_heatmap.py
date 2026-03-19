import matplotlib.pyplot as plt
import numpy as np


def plot_weight_matrix(weights, title="Synaptic Weights", xlabel="Post", ylabel="Pre", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_pn_kc_connectivity(pn_kc_matrix, save_path=None):
    binary = (pn_kc_matrix > 0).astype(float)
    return plot_weight_matrix(
        binary, title="PN -> KC Connectivity", xlabel="KC", ylabel="PN", save_path=save_path
    )
