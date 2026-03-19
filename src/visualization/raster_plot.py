import matplotlib.pyplot as plt
import numpy as np


def plot_raster(spike_monitor_i, spike_monitor_t, title="Spike Raster", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(spike_monitor_t * 1000, spike_monitor_i, s=1, c="black", marker="|")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_multi_raster(monitors, titles, save_path=None):
    n = len(monitors)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (spike_i, spike_t), title in zip(axes, monitors, titles):
        ax.scatter(spike_t * 1000, spike_i, s=1, c="black", marker="|")
        ax.set_ylabel("Neuron Index")
        ax.set_title(title)
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig
