import json
import yaml
import numpy as np
from src.encoding.spike_encoder import make_horizontal_stripes, make_vertical_stripes
from src.evolution.population import run_evolution
from src.evolution.fitness import evaluate_fitness


def run_binary_experiment(config_path="configs/experiment_binary.yaml", output_dir="results"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open("configs/default.yaml") as f:
        default_config = yaml.safe_load(f)

    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        elif isinstance(config[key], dict) and isinstance(default_config[key], dict):
            merged = dict(default_config[key])
            merged.update(config[key])
            config[key] = merged

    size = config["encoding"]["image_size"]
    pattern_h = make_horizontal_stripes(size)
    pattern_v = make_vertical_stripes(size)

    patterns = [pattern_h, pattern_v]
    labels = [0, 1]

    log_path = os.path.join(output_dir, "evolution_log.json")
    best_genome, history = run_evolution(config, patterns, labels, log_path=log_path)

    final_fitness = evaluate_fitness(best_genome, patterns, labels, config, seed=0)

    result = {
        "final_fitness": float(final_fitness),
        "generations": len(history),
        "best_generation_fitness": max(h["best_fitness"] for h in history),
        "num_kc": best_genome.num_kc,
        "num_synapses": best_genome.num_synapses,
    }

    with open(os.path.join(output_dir, "experiment_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    np.savez(
        os.path.join(output_dir, "best_genome.npz"),
        pn_kc=best_genome.pn_kc,
        kc_mbon=best_genome.kc_mbon,
        kc_thresholds=best_genome.kc_thresholds,
        kc_tau_m=best_genome.kc_tau_m,
    )

    return best_genome, history, result


if __name__ == "__main__":
    best_genome, history, result = run_binary_experiment()
    print(f"Final fitness: {result['final_fitness']:.4f}")
    print(f"Best generation fitness: {result['best_generation_fitness']:.4f}")
