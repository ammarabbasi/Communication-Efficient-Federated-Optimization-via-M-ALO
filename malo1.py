"""
M-ALO1: Modified ALO with Entropy-Guided Shrinking Boundary (EGSB)
===================================================================
Enhancement over standard ALO:

  Instead of shrinking search boundaries purely based on iteration count,
  M-ALO1 regulates boundary contraction via the Shannon entropy of the
  population's normalised fitness distribution.

  Shannon entropy  H(t) = -Σ p_i * log(p_i)
  Shrinkage factor α(t) = 1 - λ * (1 - H(t) / H_max)

  High diversity (high H) → large α → broad exploration boundaries
  Low diversity  (low H)  → small α → tight exploitation boundaries

  Dynamic bounds:
    X_min^t = A_sel - α(t) * |A_sel - lb|
    X_max^t = A_sel + α(t) * |ub  - A_sel|
"""

import numpy as np
from .alo import ALO


class MALO1(ALO):
    """M-ALO1 — ALO with Entropy-Guided Shrinking Boundary (EGSB)."""

    def __init__(self, lam: float = 0.5, **kwargs):
        """
        Parameters
        ----------
        lam   : λ — scaling constant controlling boundary shrinkage sensitivity
        **kwargs are passed to the base ALO constructor
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.H_max: float = 0.0          # running max entropy across all iterations

    # ------------------------------------------------------------------
    # Override step() to use entropy-guided boundaries
    # ------------------------------------------------------------------

    def step(self, t: int):
        """One iteration of M-ALO1."""
        # --- Entropy of normalised fitness distribution ---
        H_t = self._shannon_entropy(self.al_fitness)
        self.H_max = max(self.H_max, H_t)

        # --- Shrinkage factor (Eq. in paper) ---
        alpha = 1.0 - self.lam * (1.0 - H_t / (self.H_max + 1e-10))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        for i in range(self.n):
            al_idx = self._roulette_wheel()
            A_sel = self.ant_lions[al_idx]

            # Entropy-guided dynamic bounds (EGSB)
            X_min = A_sel - alpha * np.abs(A_sel - self.lb)
            X_max = A_sel + alpha * np.abs(self.ub - A_sel)
            X_min = np.clip(X_min, self.lb, self.ub)
            X_max = np.clip(X_max, self.lb, self.ub)

            rw = self._random_walk(X_min, X_max)

            # Average random walk with elite (standard elitism)
            X_temp = (rw + self.elite) / 2.0
            new_pos = np.clip(X_temp, self.lb, self.ub)

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

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(fitness: np.ndarray) -> float:
        """
        Shannon entropy of the normalised fitness distribution.
          H = -Σ p_i * log(p_i),  p_i = f_i / Σ f_j   (after inversion)
        """
        inv = 1.0 / (fitness - fitness.min() + 1e-10)
        p = inv / inv.sum()
        # Avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))
