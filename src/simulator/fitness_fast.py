import numpy as np
from src.simulator.fast_lif import (
    simulate_batch, build_neuron_params, build_weight_matrix,
    build_threshold_vector, generate_input_spikes_batch, NUM_NEURONS,
)


def evaluate_population(population, patterns, labels, config, n_trials=6, seed=42):
    rng = np.random.default_rng(seed)
    dt = config["simulation"]["dt"]
    duration = config["simulation"]["stimulus_duration"]
    num_steps = int(duration / dt)
    refr_steps = int(0.002 / dt)
    num_pn = 64
    trials_per_pattern = n_trials // len(patterns)

    tau_m, V_rest, V_reset, g_L = build_neuron_params()

    batch_size = len(population) * len(patterns) * trials_per_pattern
    batch_W_exc = np.zeros((batch_size, NUM_NEURONS, NUM_NEURONS))
    batch_W_inh = np.zeros((batch_size, NUM_NEURONS, NUM_NEURONS))
    batch_V_thresh = np.zeros((batch_size, NUM_NEURONS))
    batch_input = np.zeros((batch_size, num_steps, num_pn), dtype=np.bool_)
    batch_labels = np.zeros(batch_size, dtype=np.int32)

    idx = 0
    for g_idx, genome in enumerate(population):
        W_exc, W_inh = build_weight_matrix(genome)
        V_thresh = build_threshold_vector(genome)

        for p_idx in range(len(patterns)):
            pat = patterns[p_idx].flatten()
            rates = np.clip(pat, 0, 1) * 100.0

            for trial in range(trials_per_pattern):
                batch_W_exc[idx] = W_exc
                batch_W_inh[idx] = W_inh
                batch_V_thresh[idx] = V_thresh

                for pn in range(min(len(rates), num_pn)):
                    prob = rates[pn] * dt
                    batch_input[idx, :, pn] = rng.random(num_steps) < prob

                batch_labels[idx] = labels[p_idx]
                idx += 1

    all_mbon, all_kc = simulate_batch(
        batch_W_exc, batch_W_inh, batch_input, batch_V_thresh,
        V_rest, V_reset, g_L, 0.0, -0.080,
        tau_m, 0.005, 0.010,
        dt, num_steps, refr_steps,
    )

    fitnesses = np.zeros(len(population))
    trials_per_genome = len(patterns) * trials_per_pattern

    for g_idx in range(len(population)):
        start = g_idx * trials_per_genome
        end = start + trials_per_genome

        correct = 0
        for t_idx in range(start, end):
            counts = all_mbon[t_idx]
            pred = np.argmax(counts) if counts.sum() > 0 else rng.integers(2)
            if pred == batch_labels[t_idx]:
                correct += 1

        accuracy = correct / trials_per_genome
        kc_active_mean = np.mean([np.sum(all_kc[t]) for t in range(start, end)]) / 200
        sparsity_score = max(0, 1.0 - abs(kc_active_mean - 0.1) / 0.1) * 0.1
        complexity = np.count_nonzero(population[g_idx].pn_kc) / 10000 * 0.01

        fitnesses[g_idx] = accuracy + sparsity_score - complexity

    return fitnesses
