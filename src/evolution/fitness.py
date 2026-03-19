import numpy as np
from brian2 import Network, SpikeMonitor, ms, second, nS, SpikeGeneratorGroup, defaultclock
from src.connectome.builder import MushroomBodyBuilder
from src.encoding.spike_encoder import image_to_rates, generate_poisson_spike_indices_and_times


def evaluate_fitness(genome, patterns, labels, config, seed=42):
    rng = np.random.default_rng(seed)
    connectome = genome.to_connectome_dict()
    duration = config["simulation"]["stimulus_duration"]
    dt = config["simulation"]["dt"]
    training_epochs = config["simulation"].get("training_epochs", 5)
    test_trials = config["simulation"].get("test_trials", 20)
    max_rate = config["encoding"]["max_rate"]
    input_w = config.get("synapse_weights", {}).get("input_to_pn", 50.0)

    builder = MushroomBodyBuilder(config, connectome)

    _run_training(
        builder, patterns, labels, duration, dt, max_rate,
        training_epochs, input_w, seed=rng.integers(1e6)
    )

    test_correct = 0
    test_total = 0

    for trial in range(min(test_trials, len(patterns) * 10)):
        from brian2 import start_scope
        start_scope()

        pattern_idx = trial % len(patterns)
        net, components = builder.build_network(enable_stdp=False, enable_monitors=True)

        rates = image_to_rates(patterns[pattern_idx], max_rate=max_rate)
        full_rates = np.zeros(genome.num_pn)
        full_rates[:len(rates)] = rates
        indices, times = generate_poisson_spike_indices_and_times(
            full_rates, duration, dt=dt, seed=rng.integers(1e6)
        )

        if len(indices) == 0:
            continue

        input_group = SpikeGeneratorGroup(
            genome.num_pn, indices, times * second, name="test_input"
        )
        input_syn = _connect_input(input_group, components["pn"], net, input_w)

        defaultclock.dt = dt * second
        net.run(duration * second)

        mbon_counts = np.array(components["mbon_monitor"].count)
        if mbon_counts.sum() > 0:
            predicted = np.argmax(mbon_counts)
        else:
            predicted = rng.integers(len(set(labels)))

        if predicted == labels[pattern_idx]:
            test_correct += 1
        test_total += 1

    accuracy = test_correct / max(test_total, 1)

    kc_mon = components["kc_monitor"]
    kc_sparsity = _compute_sparsity(kc_mon, genome.num_kc)
    sparsity_bonus = config["evolution"]["sparsity_bonus_weight"] * kc_sparsity
    complexity_penalty = config["evolution"]["complexity_penalty_weight"] * (genome.num_synapses / 10000.0)

    return accuracy + sparsity_bonus - complexity_penalty


def _run_training(builder, patterns, labels, duration, dt, max_rate, epochs, input_w, seed):
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        from brian2 import start_scope
        start_scope()

        net, components = builder.build_network(enable_stdp=True, enable_monitors=False)
        order = rng.permutation(len(patterns))

        for idx in order:
            rates = image_to_rates(patterns[idx], max_rate=max_rate)
            full_rates = np.zeros(builder.num_pn)
            full_rates[:len(rates)] = rates
            indices, times = generate_poisson_spike_indices_and_times(
                full_rates, duration, dt=dt, seed=rng.integers(1e6)
            )
            if len(indices) == 0:
                continue

            input_group = SpikeGeneratorGroup(
                builder.num_pn, indices, times * second,
                name="train_input"
            )
            input_syn = _connect_input(input_group, components["pn"], net, input_w)

            defaultclock.dt = dt * second
            net.run(duration * second)

            net.remove(input_group)
            net.remove(input_syn)


def _connect_input(input_group, pn_group, net, weight_nS=50.0):
    from brian2 import Synapses
    syn = Synapses(input_group, pn_group,
                   on_pre="g_exc_post += %.1f*nS" % weight_nS,
                   name="input_syn_temp")
    syn.connect("i == j")
    net.add(input_group)
    net.add(syn)
    return syn


def _compute_sparsity(kc_monitor, num_kc):
    active_kcs = len(set(kc_monitor.i))
    if num_kc == 0:
        return 0.0
    fraction_active = active_kcs / num_kc
    target_sparsity = 0.1
    return max(0, 1.0 - abs(fraction_active - target_sparsity) / target_sparsity)
