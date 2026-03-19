import yaml
import numpy as np
from brian2 import second, SpikeGeneratorGroup, defaultclock, ms
from src.connectome.loader import generate_synthetic_connectome
from src.connectome.builder import MushroomBodyBuilder
from src.encoding.spike_encoder import (
    image_to_rates,
    generate_poisson_spike_indices_and_times,
    make_horizontal_stripes,
    make_vertical_stripes,
)


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


class TestSpikeEncoder:
    def test_rate_encoding(self):
        pattern = make_horizontal_stripes(8)
        rates = image_to_rates(pattern, max_rate=100.0)
        assert len(rates) == 64
        assert rates.max() <= 100.0
        assert rates.min() >= 0.0

    def test_poisson_spikes(self):
        rates = np.full(10, 50.0)
        indices, times = generate_poisson_spike_indices_and_times(rates, 0.1, seed=42)
        assert len(indices) > 0
        assert len(indices) == len(times)
        assert np.all(times >= 0)
        assert np.all(times < 0.1)

    def test_patterns_differ(self):
        h = make_horizontal_stripes(8)
        v = make_vertical_stripes(8)
        assert not np.array_equal(h, v)
        assert h.sum() == v.sum()


class TestNetworkSimulation:
    def test_network_runs(self):
        config = load_config()
        config["connectome"]["num_pn"] = 64
        config["connectome"]["num_kc"] = 50
        config["connectome"]["num_mbon"] = 2
        config["connectome"]["kc_pn_connections"] = 6

        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        builder = MushroomBodyBuilder(config, connectome)
        net, components = builder.build_network(enable_stdp=False, enable_monitors=True)

        pattern = make_horizontal_stripes(8)
        rates = image_to_rates(pattern, max_rate=100.0)
        indices, times = generate_poisson_spike_indices_and_times(rates, 0.05, seed=42)

        input_group = SpikeGeneratorGroup(64, indices, times * second, name="test_input")
        from brian2 import Synapses, nS
        input_syn = Synapses(input_group, components["pn"], on_pre="g_exc_post += 5*nS", name="test_input_syn")
        input_syn.connect("i == j")
        net.add(input_group)
        net.add(input_syn)

        defaultclock.dt = 0.1 * ms
        net.run(0.05 * second)

        pn_spikes = components["pn_monitor"].num_spikes
        assert pn_spikes > 0, "PN neurons should fire with input"
