"""
Ant Lion Optimizer (ALO) — Standard Implementation
Based on: Mirjalili, S. (2015). The ant lion optimizer. Advances in Engineering Software.

Population matrices:
    Ants      : n x d  — candidate solutions (random walkers)
    Ant Lions : n x d  — elite traps (influence ants' trajectories)

Key operators:
    1. Random Walk of Ants
    2. Roulette Wheel Selection
    3. Adaptive Boundary Shrinking
    4. Elitism (global best preservation)
"""

import numpy as np


class ALO:
    """Standard Ant Lion Optimizer."""

    def __init__(
        self,
        n: int = 10,
        d: int = 1,
        max_iter: int = 30,
        lb: float = 0.0,
        ub: float = 1.0,
        fitness_fn=None,
    ):
        """
        Parameters
        ----------
        n         : population size (number of ants / ant lions)
        d         : dimensionality of the problem
        max_iter  : maximum number of iterations (= FL communication rounds)
        lb, ub    : scalar or array-like search space bounds
        fitness_fn: callable, fitness_fn(position) -> float (lower is better)
        """
        self.n = n
        self.d = d
        self.max_iter = max_iter
        self.lb = np.full(d, lb) if np.isscalar(lb) else np.array(lb, dtype=float)
        self.ub = np.full(d, ub) if np.isscalar(ub) else np.array(ub, dtype=float)
        self.fitness_fn = fitness_fn

        # State (populated at runtime)
        self.ants: np.ndarray = None
        self.ant_lions: np.ndarray = None
        self.ant_fitness: np.ndarray = None
        self.al_fitness: np.ndarray = None
        self.elite: np.ndarray = None
        self.elite_fitness: float = np.inf
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Randomly initialise ant and ant-lion positions within [lb, ub]."""
        self.ants = self._random_population()
        self.ant_lions = self._random_population()
        self.ant_fitness = self._evaluate_all(self.ants)
        self.al_fitness = self._evaluate_all(self.ant_lions)
        best_idx = np.argmin(self.al_fitness)
        self.elite = self.ant_lions[best_idx].copy()
        self.elite_fitness = self.al_fitness[best_idx]
        self.history = [self.elite_fitness]

    def step(self, t: int):
        """Execute one iteration of ALO (call after initialize())."""
        I = self._shrinkage_ratio(t)
        c = self.lb / I
        d = self.ub / I

        for i in range(self.n):
            # Roulette-wheel select an ant lion
            al_idx = self._roulette_wheel()
            A_sel = self.ant_lions[al_idx]

            # Adaptive bounds around selected ant lion and elite
            ct_al = A_sel + c
            dt_al = A_sel + d
            ct_el = self.elite + c
            dt_el = self.elite + d

            # Random walks (normalized cumsum)
            rw_al = self._random_walk(ct_al, dt_al)
            rw_el = self._random_walk(ct_el, dt_el)

            # Average both random walks → new ant position
            new_pos = (rw_al + rw_el) / 2.0
            new_pos = np.clip(new_pos, self.lb, self.ub)
            self.ants[i] = new_pos
            self.ant_fitness[i] = self._evaluate(new_pos)

            # Ant replaces ant lion if it is fitter
            if self.ant_fitness[i] < self.al_fitness[al_idx]:
                self.ant_lions[al_idx] = self.ants[i].copy()
                self.al_fitness[al_idx] = self.ant_fitness[i]

        # Update elite
        best_idx = np.argmin(self.al_fitness)
        if self.al_fitness[best_idx] < self.elite_fitness:
            self.elite = self.ant_lions[best_idx].copy()
            self.elite_fitness = self.al_fitness[best_idx]

        self.history.append(self.elite_fitness)
        return self.elite.copy(), self.elite_fitness

    def run(self, fitness_fn=None):
        """Convenience method: initialise and run for max_iter steps."""
        if fitness_fn is not None:
            self.fitness_fn = fitness_fn
        self.initialize()
        for t in range(1, self.max_iter + 1):
            self.step(t)
        return self.elite.copy(), self.elite_fitness

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_population(self) -> np.ndarray:
        return np.random.rand(self.n, self.d) * (self.ub - self.lb) + self.lb

    def _evaluate(self, x: np.ndarray) -> float:
        if self.fitness_fn is None:
            raise ValueError("fitness_fn must be set before evaluation.")
        return float(self.fitness_fn(x))

    def _evaluate_all(self, pop: np.ndarray) -> np.ndarray:
        return np.array([self._evaluate(pop[i]) for i in range(self.n)])

    def _shrinkage_ratio(self, t: int) -> float:
        """
        Adaptive shrinkage ratio I(t) = 10^w * (t / T)
        w increases with iteration stage to tighten boundaries.
        """
        ratio = t / self.max_iter
        if ratio < 0.1:
            w = 2
        elif ratio < 0.5:
            w = 3
        elif ratio < 0.75:
            w = 4
        else:
            w = 6
        I = (10 ** w) * ratio
        return max(I, 1e-10)

    def _roulette_wheel(self) -> int:
        """Select an ant lion index proportional to fitness (lower fitness → higher prob)."""
        # Invert fitness so that lower fitness gets higher weight
        inv = 1.0 / (self.al_fitness - self.al_fitness.min() + 1e-10)
        prob = inv / inv.sum()
        return int(np.random.choice(self.n, p=prob))

    def _random_walk(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        Cumulative-sum random walk normalised to [lower, upper].
        X(t) = cumsum(2*r(k) - 1), then linearly rescaled to bounds.
        """
        steps = np.random.randint(0, 2, size=self.d) * 2 - 1   # {-1, +1}
        walk = np.cumsum(steps).astype(float)

        w_min, w_max = walk.min(), walk.max()
        if w_max == w_min:
            normalised = np.zeros_like(walk)
        else:
            normalised = (walk - w_min) / (w_max - w_min)   # → [0, 1]

        return lower + normalised * (upper - lower)
