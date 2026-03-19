import json
import time
import numpy as np
from tqdm import tqdm
from src.evolution.genome import ConnectomeGenome
from src.evolution.mutations import mutate
from src.evolution.crossover import crossover
from src.evolution.fitness import evaluate_fitness


def tournament_selection(population, fitnesses, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]


def create_initial_population(size, num_pn, num_kc, num_mbon, kc_pn_k, seed=42):
    rng = np.random.default_rng(seed)
    population = []
    for i in range(size):
        genome = ConnectomeGenome.random(
            num_pn=num_pn, num_kc=num_kc, num_mbon=num_mbon,
            kc_pn_k=kc_pn_k, seed=rng.integers(1e9),
        )
        population.append(genome)
    return population


def run_evolution(config, patterns, labels, log_path=None):
    evo_config = config["evolution"]
    conn_config = config["connectome"]
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    population_size = evo_config["population_size"]
    generations = evo_config["generations"]
    mutation_rate = evo_config["mutation_rate"]
    crossover_rate = evo_config["crossover_rate"]
    elitism = evo_config["elitism"]
    tournament_k = evo_config["tournament_k"]

    population = create_initial_population(
        size=population_size,
        num_pn=conn_config["num_pn"],
        num_kc=conn_config["num_kc"],
        num_mbon=conn_config["num_mbon"],
        kc_pn_k=conn_config["kc_pn_connections"],
        seed=rng.integers(1e9),
    )

    history = []

    for gen in tqdm(range(generations), desc="Evolution"):
        gen_start = time.time()

        fitnesses = []
        for genome in population:
            fitness = evaluate_fitness(
                genome, patterns, labels, config,
                seed=rng.integers(1e9),
            )
            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses)
        best_idx = np.argmax(fitnesses)
        gen_time = time.time() - gen_start

        gen_log = {
            "generation": gen,
            "best_fitness": float(fitnesses[best_idx]),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "worst_fitness": float(np.min(fitnesses)),
            "best_synapses": int(population[best_idx].num_synapses),
            "elapsed_seconds": round(gen_time, 2),
        }
        history.append(gen_log)

        if log_path:
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population = [population[i].copy() for i in sorted_indices[:elitism]]

        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, k=tournament_k, rng=rng)

            if rng.random() < crossover_rate:
                parent_b = tournament_selection(population, fitnesses, k=tournament_k, rng=rng)
                child = crossover(parent_a, parent_b, seed=rng.integers(1e9))
            else:
                child = parent_a.copy()

            child = mutate(child, mutation_rate=mutation_rate, seed=rng.integers(1e9))
            new_population.append(child)

        population = new_population

    final_fitnesses = []
    for genome in population:
        fitness = evaluate_fitness(genome, patterns, labels, config, seed=rng.integers(1e9))
        final_fitnesses.append(fitness)

    best_genome = population[np.argmax(final_fitnesses)]
    return best_genome, history
