import numpy as np


def mutate(genome, mutation_rate=0.1, seed=None):
    rng = np.random.default_rng(seed)
    child = genome.copy()

    if rng.random() < mutation_rate:
        _mutate_weights(child, rng)

    if rng.random() < mutation_rate * 0.5:
        _add_synapse(child, rng)

    if rng.random() < mutation_rate * 0.3:
        _remove_synapse(child, rng)

    if rng.random() < mutation_rate * 0.5:
        _mutate_threshold(child, rng)

    if rng.random() < mutation_rate * 0.3:
        _mutate_tau(child, rng)

    if rng.random() < mutation_rate * 0.2:
        _rewire(child, rng)

    return child


def _mutate_weights(genome, rng):
    noise = rng.normal(0, 0.05, size=genome.kc_mbon.shape)
    genome.kc_mbon = np.clip(genome.kc_mbon + noise, 0.0, 1.0)

    mask = genome.pn_kc > 0
    pn_kc_noise = rng.normal(0, 0.02, size=genome.pn_kc.shape)
    genome.pn_kc = np.where(mask, np.clip(genome.pn_kc + pn_kc_noise, 0.01, 2.0), genome.pn_kc)


def _add_synapse(genome, rng):
    zero_mask = genome.pn_kc == 0
    zero_positions = np.argwhere(zero_mask)
    if len(zero_positions) == 0:
        return
    idx = rng.integers(len(zero_positions))
    pn_idx, kc_idx = zero_positions[idx]
    genome.pn_kc[pn_idx, kc_idx] = rng.uniform(0.1, 0.8)


def _remove_synapse(genome, rng):
    nonzero_positions = np.argwhere(genome.pn_kc > 0)
    if len(nonzero_positions) < 2:
        return
    idx = rng.integers(len(nonzero_positions))
    pn_idx, kc_idx = nonzero_positions[idx]
    kc_input_count = np.count_nonzero(genome.pn_kc[:, kc_idx])
    if kc_input_count > 2:
        genome.pn_kc[pn_idx, kc_idx] = 0.0


def _mutate_threshold(genome, rng):
    num_to_mutate = max(1, genome.num_kc // 10)
    indices = rng.choice(genome.num_kc, size=num_to_mutate, replace=False)
    noise = rng.normal(0, 0.002, size=num_to_mutate)
    genome.kc_thresholds[indices] = np.clip(
        genome.kc_thresholds[indices] + noise, -0.060, -0.040
    )


def _mutate_tau(genome, rng):
    num_to_mutate = max(1, genome.num_kc // 10)
    indices = rng.choice(genome.num_kc, size=num_to_mutate, replace=False)
    noise = rng.normal(0, 0.002, size=num_to_mutate)
    genome.kc_tau_m[indices] = np.clip(
        genome.kc_tau_m[indices] + noise, 0.005, 0.050
    )


def _rewire(genome, rng):
    nonzero_positions = np.argwhere(genome.pn_kc > 0)
    if len(nonzero_positions) == 0:
        return
    idx = rng.integers(len(nonzero_positions))
    old_pn, kc_idx = nonzero_positions[idx]
    kc_input_count = np.count_nonzero(genome.pn_kc[:, kc_idx])
    if kc_input_count <= 2:
        return
    weight = genome.pn_kc[old_pn, kc_idx]
    genome.pn_kc[old_pn, kc_idx] = 0.0
    new_pn = rng.integers(genome.num_pn)
    while genome.pn_kc[new_pn, kc_idx] > 0:
        new_pn = rng.integers(genome.num_pn)
    genome.pn_kc[new_pn, kc_idx] = weight
