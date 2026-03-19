import numpy as np
from src.evolution.genome import ConnectomeGenome
from src.evolution.mutations import mutate
from src.evolution.crossover import crossover


class TestGenome:
    def test_create_genome(self):
        genome = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=42)
        assert genome.pn_kc.shape == (64, 50)
        assert genome.kc_mbon.shape == (50, 2)

    def test_genome_sparsity(self):
        genome = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=42)
        kc_inputs = np.count_nonzero(genome.pn_kc, axis=0)
        assert np.all(kc_inputs == 6)


class TestMutations:
    def test_mutate_weight(self):
        genome = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=42)
        original_weights = genome.kc_mbon.copy()
        mutated = mutate(genome, mutation_rate=1.0, seed=99)
        assert not np.array_equal(original_weights, mutated.kc_mbon)

    def test_mutate_preserves_shape(self):
        genome = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=42)
        mutated = mutate(genome, mutation_rate=0.5, seed=99)
        assert mutated.pn_kc.shape == genome.pn_kc.shape
        assert mutated.kc_mbon.shape == genome.kc_mbon.shape


class TestCrossover:
    def test_crossover_produces_offspring(self):
        parent_a = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=42)
        parent_b = ConnectomeGenome.random(num_pn=64, num_kc=50, num_mbon=2, kc_pn_k=6, seed=99)
        child = crossover(parent_a, parent_b, seed=0)
        assert child.pn_kc.shape == parent_a.pn_kc.shape
        assert child.kc_mbon.shape == parent_a.kc_mbon.shape
