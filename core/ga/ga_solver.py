from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from problem.fitness import repair_constraints, fitness_function
from core.ga.operators import get_selection, get_crossover, get_mutation
from utils.diversity import population_diversity


@dataclass
class GAConfig:
    pop_size: int = 30
    iterations: int = 100
    pc: float = 0.8          
    pm: float = 0.15         
    tournament_k: int = 3
    elite_count: int = 2
    selection: str = "tournament"
    crossover: str = "whole"
    mutation: str  = "non_uniform"


class GASolver:
    def __init__(self, scenario, config: Optional[GAConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.scenario = scenario
        self.config = config or GAConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        upper = scenario.demands.max() * 1.5
        self._pos_low = 0.0
        self._pos_high = float(upper)

        self._select = get_selection(self.config.selection)
        self._cross  = get_crossover(self.config.crossover)
        self._mutate = get_mutation(self.config.mutation)

    def _init_population(self) -> List[np.ndarray]:
        return [
            self.rng.uniform(self._pos_low, self._pos_high,
                             size=self.scenario.dimension)
            for _ in range(self.config.pop_size)
        ]

    def _next_generation(self, population, fitness, current_iter, max_iter):
        cfg = self.config
        n = len(population)
        elite_idx = np.argsort(fitness)[: cfg.elite_count]
        next_pop = [population[i].copy() for i in elite_idx]

        while len(next_pop) < n:
            p1 = self._select(population, fitness, self.rng, cfg.tournament_k)
            p2 = self._select(population, fitness, self.rng, cfg.tournament_k)

            if self.rng.random() < cfg.pc:
                c1, c2 = self._cross(p1, p2, self.rng, self.rng.uniform(0.3, 0.7))
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in [c1, c2]:
                if self.rng.random() < cfg.pm:
                    child = self._mutate(child, self._pos_low, self._pos_high,
                                         self.rng, current_iter, max_iter, rate=0.2)
                next_pop.append(child)
                if len(next_pop) >= n:
                    break

        return next_pop[:n]

    def run(self, verbose: bool = False
            ) -> Tuple[np.ndarray, float, List[float], List[float]]:
        population = self._init_population()
        population = [repair_constraints(g, self.scenario) for g in population]
        fitness = np.array([
            fitness_function(g, self.scenario) for g in population
        ])

        best_idx = int(np.argmin(fitness))
        best_pos = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        history: List[float] = []
        diversity_history: List[float] = []

        for it in range(self.config.iterations):
            population = self._next_generation(population, fitness, it,
                                               self.config.iterations)
            population = [repair_constraints(g, self.scenario) for g in population]
            fitness = np.array([
                fitness_function(g, self.scenario) for g in population
            ])

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_fit:
                best_fit = float(fitness[gen_best])
                best_pos = population[gen_best].copy()

            history.append(best_fit)
            diversity_history.append(population_diversity(population))

            if verbose and (it % 10 == 0 or it == self.config.iterations - 1):
                print(f"  [GA] iter {it:3d}  best={best_fit:.6f}"
                      f"  div={diversity_history[-1]:.4f}")

        return best_pos, best_fit, history, diversity_history
