import numpy as np


def image_to_rates(image, max_rate=100.0, min_rate=0.0):
    flat = image.flatten().astype(np.float64)
    flat = np.clip(flat, 0.0, 1.0)
    return flat * (max_rate - min_rate) + min_rate


def generate_spike_trains(rates, duration, dt=0.0001, seed=None):
    rng = np.random.default_rng(seed)
    num_neurons = len(rates)
    num_steps = int(duration / dt)
    spikes = {}
    for i in range(num_neurons):
        prob_per_step = rates[i] * dt
        spike_times = []
        for t_idx in range(num_steps):
            if rng.random() < prob_per_step:
                spike_times.append(t_idx * dt)
        spikes[i] = np.array(spike_times)
    return spikes


def generate_poisson_spike_indices_and_times(rates, duration, dt=0.00005, seed=None):
    rng = np.random.default_rng(seed)
    num_neurons = len(rates)
    num_steps = int(duration / dt)
    offset = dt
    all_indices = []
    all_times = []
    for i in range(num_neurons):
        prob_per_step = rates[i] * dt
        spike_mask = rng.random(num_steps) < prob_per_step
        spike_step_indices = np.where(spike_mask)[0]
        times = spike_step_indices * dt + offset
        all_indices.extend([i] * len(times))
        all_times.extend(times)
    order = np.argsort(all_times)
    return np.array(all_indices, dtype=int)[order], np.array(all_times)[order]


def make_horizontal_stripes(size=8):
    pattern = np.zeros((size, size))
    for row in range(0, size, 2):
        pattern[row, :] = 1.0
    return pattern


def make_vertical_stripes(size=8):
    pattern = np.zeros((size, size))
    for col in range(0, size, 2):
        pattern[:, col] = 1.0
    return pattern
