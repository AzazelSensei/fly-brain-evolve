import numpy as np
import yaml
from src.connectome.loader import generate_synthetic_connectome
from src.connectome.builder import MushroomBodyBuilder


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


class TestSyntheticConnectome:
    def test_connectome_shape(self):
        config = load_config()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        num_pn = config["connectome"]["num_pn"]
        num_kc = config["connectome"]["num_kc"]
        assert connectome["pn_kc"].shape == (num_pn, num_kc)

    def test_kc_input_count(self):
        config = load_config()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        pn_kc = connectome["pn_kc"]
        kc_inputs = np.count_nonzero(pn_kc, axis=0)
        expected = config["connectome"]["kc_pn_connections"]
        assert np.all(kc_inputs == expected)

    def test_kc_mbon_all_to_all(self):
        config = load_config()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        kc_mbon = connectome["kc_mbon"]
        num_kc = config["connectome"]["num_kc"]
        num_mbon = config["connectome"]["num_mbon"]
        assert kc_mbon.shape == (num_kc, num_mbon)
        assert np.all(kc_mbon > 0)

    def test_apl_connections(self):
        config = load_config()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        assert connectome["kc_apl"].shape[0] == config["connectome"]["num_kc"]
        assert connectome["apl_kc"].shape[0] == config["connectome"]["num_kc"]

    def test_reproducibility(self):
        config = load_config()
        c1 = generate_synthetic_connectome(config["connectome"], seed=42)
        c2 = generate_synthetic_connectome(config["connectome"], seed=42)
        assert np.array_equal(c1["pn_kc"], c2["pn_kc"])

    def test_different_seeds_differ(self):
        config = load_config()
        c1 = generate_synthetic_connectome(config["connectome"], seed=42)
        c2 = generate_synthetic_connectome(config["connectome"], seed=99)
        assert not np.array_equal(c1["pn_kc"], c2["pn_kc"])


class TestMushroomBodyBuilder:
    def test_builder_creates_groups(self):
        config = load_config()
        connectome = generate_synthetic_connectome(config["connectome"], seed=42)
        builder = MushroomBodyBuilder(config, connectome)
        network = builder.build()
        assert "pn" in network
        assert "kc" in network
        assert "mbon" in network
        assert "apl" in network
        assert "pn_kc_syn" in network
        assert "kc_mbon_syn" in network
