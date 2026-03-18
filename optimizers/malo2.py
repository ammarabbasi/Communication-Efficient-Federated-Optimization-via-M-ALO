"""
M-ALO2: Modified ALO with Entropy-Guided Shrinking Boundary + Momentum Update
==============================================================================
Builds on M-ALO1 (EGSB) and adds a momentum-based position update rule:

  v_i^{t+1} = γ * v_i^t + β * (X_i^temp - X_i^t)
  X_i^{t+1} = X_i^t + v_i^{t+1}

  γ (gamma) : momentum coefficient — how much of the previous velocity is retained
  β (beta)  : displacement weight — step size toward the intermediate target

This temporal smoothing reduces oscillatory behaviour and accelerates convergence
in FL environments with high data variability and intermittent communication delays.
"""

import numpy as np
from .malo1 import MALO1


class MALO2(MALO1):
    """M-ALO2 — M-ALO1 augmented with momentum-based position updates."""

    def __init__(self, gamma: float = 0.9, beta: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        gamma : momentum coefficient (γ), controls previous-velocity influence
        beta  : momentum weight (β), modulates current displacement
        **kwargs passed to MALO1 (and ALO) constructor
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta
        self.velocities: np.ndarray = None      # v_i — one row per ant

    # ------------------------------------------------------------------
    # Override initialize() to also reset velocities
    # ------------------------------------------------------------------

    def initialize(self):
        super().initialize()
        self.velocities = np.zeros((self.n, self.d))

    # ------------------------------------------------------------------
    # Override step() to apply momentum update after computing X_temp
    # ------------------------------------------------------------------

    def step(self, t: int):
        """One iteration of M-ALO2 (EGSB + momentum)."""
        # ---- Entropy / shrinkage (inherited logic) ----
        H_t = self._shannon_entropy(self.al_fitness)
        self.H_max = max(self.H_max, H_t)
        alpha = 1.0 - self.lam * (1.0 - H_t / (self.H_max + 1e-10))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        for i in range(self.n):
            al_idx = self._roulette_wheel()
            A_sel = self.ant_lions[al_idx]

            # Entropy-guided dynamic bounds (EGSB, same as M-ALO1)
            X_min = A_sel - alpha * np.abs(A_sel - self.lb)
            X_max = A_sel + alpha * np.abs(self.ub - A_sel)
            X_min = np.clip(X_min, self.lb, self.ub)
            X_max = np.clip(X_max, self.lb, self.ub)

            rw = self._random_walk(X_min, X_max)
            X_temp = (rw + self.elite) / 2.0   # intermediate position

            # ---- Momentum update (M-ALO2 specific) ----
            v_new = self.gamma * self.velocities[i] + self.beta * (X_temp - self.ants[i])
            new_pos = self.ants[i] + v_new
            new_pos = np.clip(new_pos, self.lb, self.ub)
            self.velocities[i] = v_new

            self.ants[i] = new_pos
            self.ant_fitness[i] = self._evaluate(new_pos)

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
