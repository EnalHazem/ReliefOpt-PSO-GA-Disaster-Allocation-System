from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from problem.fitness import repair_constraints, fitness_function
from core.pso.pso_solver import PSOSolver, PSOConfig
from core.ga.operators import get_selection, get_crossover, get_mutation
from utils.diversity import population_diversity


@dataclass
class HybridConfig:
    # Shared budget
    pop_size: int = 30
    iterations: int = 100
    pso_fraction: float = 0.6       

    
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.5
    c2: float = 1.5

    
    pc: float = 0.8                
    pm: float = 0.15                
    tournament_k: int = 3
    elite_count: int = 2            

    
    selection: str = "tournament"   
    crossover: str = "whole"        
    mutation: str  = "non_uniform"  


class HybridPSOGASolver:
    def __init__(self, scenario, config: Optional[HybridConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.scenario = scenario
        self.config = config or HybridConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

       
        upper = scenario.demands.max() * 1.5
        self._pos_low = 0.0
        self._pos_high = float(upper)

        
        self._select = get_selection(self.config.selection)
        self._cross = get_crossover(self.config.crossover)
        self._mutate = get_mutation(self.config.mutation)

   

    def run(self, verbose: bool = False
            ) -> Tuple[np.ndarray, float, List[float], List[float]]:
        cfg = self.config
        pso_iters = max(1, int(cfg.iterations * cfg.pso_fraction))
        ga_iters = max(0, cfg.iterations - pso_iters)

        pso_cfg = PSOConfig(
            pop_size=cfg.pop_size,
            iterations=pso_iters,
            w_max=cfg.w_max, w_min=cfg.w_min,
            c1=cfg.c1, c2=cfg.c2,
        )
        pso = PSOSolver(self.scenario, pso_cfg, rng=self.rng)
        pso_best_pos, pso_best_fit, history, diversity_history = pso.run(verbose=verbose)

        if verbose:
            print(f"  [HYBRID] PSO stage done. fitness={pso_best_fit:.6f}")

        if ga_iters == 0:
            return pso_best_pos, pso_best_fit, history, diversity_history

       
        population: List[np.ndarray] = [p.pbest_position.copy() for p in pso.swarm]
        fitness = np.array([
            fitness_function(repair_constraints(g, self.scenario), self.scenario)
            for g in population
        ])
        best_idx = int(np.argmin(fitness))
        best_pos = population[best_idx].copy()
        best_fit = float(fitness[best_idx])
        if best_fit > pso_best_fit:
            best_pos, best_fit = pso_best_pos.copy(), pso_best_fit

        
        for it in range(ga_iters):
            new_pop = self._next_generation(population, fitness, it, ga_iters)

            new_fitness = np.empty(len(new_pop))
            for i, g in enumerate(new_pop):
                new_pop[i] = repair_constraints(g, self.scenario)
                new_fitness[i] = fitness_function(new_pop[i], self.scenario)

            population = new_pop
            fitness = new_fitness

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = float(fitness[gen_best_idx])
                best_pos = population[gen_best_idx].copy()

            history.append(best_fit)
            diversity_history.append(population_diversity(population))
            if verbose and (it % 10 == 0 or it == ga_iters - 1):
                print(f"  [GA]  iter {it:3d}  best = {best_fit:.6f}"
                      f"  div = {diversity_history[-1]:.4f}")

        return best_pos, best_fit, history, diversity_history


    def _next_generation(self, population: List[np.ndarray], fitness: np.ndarray,
                         current_iter: int, max_iter: int) -> List[np.ndarray]:
        cfg = self.config
        n = len(population)

        elite_idx = np.argsort(fitness)[: cfg.elite_count]
        next_pop: List[np.ndarray] = [population[i].copy() for i in elite_idx]

        while len(next_pop) < n:
            parent1 = self._select(population, fitness, self.rng, cfg.tournament_k)
            parent2 = self._select(population, fitness, self.rng, cfg.tournament_k)

            if self.rng.random() < cfg.pc:
                child1, child2 = self._cross(
                    parent1, parent2, self.rng, self.rng.uniform(0.3, 0.7)
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if self.rng.random() < cfg.pm:
                child1 = self._mutate(
                    child1, self._pos_low, self._pos_high, self.rng,
                    current_iter, max_iter, rate=0.2,
                )
            if self.rng.random() < cfg.pm:
                child2 = self._mutate(
                    child2, self._pos_low, self._pos_high, self.rng,
                    current_iter, max_iter, rate=0.2,
                )

            next_pop.append(child1)
            if len(next_pop) < n:
                next_pop.append(child2)

        return next_pop[:n]
