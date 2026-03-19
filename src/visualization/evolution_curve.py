import json
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_curve(history, save_path=None):
    generations = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    std = [h["std_fitness"] for h in history]

    mean_arr = np.array(mean)
    std_arr = np.array(std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best, label="Best Fitness", linewidth=2, color="blue")
    ax.plot(generations, mean, label="Mean Fitness", linewidth=1, color="orange")
    ax.fill_between(
        generations, mean_arr - std_arr, mean_arr + std_arr,
        alpha=0.2, color="orange",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolution Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_from_log(log_path, save_path=None):
    with open(log_path) as f:
        history = json.load(f)
    return plot_fitness_curve(history, save_path=save_path)
