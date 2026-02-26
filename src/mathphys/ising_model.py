"""
2D Ising Model — Metropolis-Hastings Monte Carlo

System
------
Square N×N lattice with periodic boundary conditions.
Hamiltonian:  H = -J Σ_{<i,j>} s_i s_j
Spins:        s_i ∈ {-1, +1}
Coupling:     J > 0  (ferromagnetic)

Onsager's exact critical temperature (J = k_B = 1):
    T_c = 2 / ln(1 + √2) ≈ 2.2692
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

# Exact critical temperature (J = k_B = 1)
T_CRITICAL = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2692


class IsingModel2D:
    """2D Ising model on an N×N square lattice with periodic boundary conditions."""

    def __init__(self, n: int = 32, j: float = 1.0, seed: int = 42) -> None:
        """
        Parameters
        ----------
        n    : lattice size (N×N sites)
        j    : coupling constant (J > 0 for ferromagnet)
        seed : random-number seed for reproducibility
        """
        self.n = n
        self.j = j
        self._rng = default_rng(seed)
        # Random ±1 initialisation
        self.lattice: np.ndarray = self._rng.choice([-1, 1], size=(n, n)).astype(np.int8)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def energy(self) -> float:
        """Total energy E = -J Σ_{<i,j>} s_i s_j (sum over nearest-neighbour pairs)."""
        s = self.lattice
        # Sum right-neighbour and down-neighbour interactions (PBC via np.roll)
        return float(-self.j * (
            np.sum(s * np.roll(s, -1, axis=1)) +  # right
            np.sum(s * np.roll(s, -1, axis=0))    # down
        ))

    def magnetization(self) -> float:
        """Total magnetization M = Σ_i s_i."""
        return float(np.sum(self.lattice))

    # ------------------------------------------------------------------
    # Monte Carlo dynamics
    # ------------------------------------------------------------------

    def metropolis_step(self, temperature: float) -> None:
        """
        One full MC sweep: N² single-spin Metropolis-Hastings attempts.

        For each randomly chosen spin site (i, j):
          ΔE = 2 J s_{i,j} (s_{up} + s_{down} + s_{left} + s_{right})
          Accept flip if ΔE ≤ 0 or with probability exp(-ΔE / T).
        """
        n = self.n
        s = self.lattice
        beta = 1.0 / temperature

        # Pre-compute acceptance probabilities for positive ΔE values
        # ΔE ∈ {4J, 8J} for non-trivial cases; cache exp(-β·ΔE)
        exp_neg = {
            dE: float(np.exp(-beta * dE * self.j))
            for dE in (4, 8)
        }

        rows = self._rng.integers(0, n, size=n * n)
        cols = self._rng.integers(0, n, size=n * n)

        for i, j in zip(rows, cols):
            spin = int(s[i, j])
            neighbour_sum = (
                int(s[(i - 1) % n, j]) +
                int(s[(i + 1) % n, j]) +
                int(s[i, (j - 1) % n]) +
                int(s[i, (j + 1) % n])
            )
            dE_over_J = 2 * spin * neighbour_sum  # in units of J
            if dE_over_J <= 0 or self._rng.random() < exp_neg.get(dE_over_J, 0.0):
                s[i, j] = -spin

    def simulate(
        self,
        temperature: float,
        n_therm: int = 5_000,
        n_measure: int = 10_000,
    ) -> dict[str, float]:
        """
        Run MC simulation at fixed temperature.

        Parameters
        ----------
        temperature : temperature T (in units where k_B = 1)
        n_therm     : number of thermalisation sweeps (discarded)
        n_measure   : number of measurement sweeps

        Returns
        -------
        dict with keys:
            E_mean, E2_mean   — mean energy / site, mean (energy/site)²
            M_mean, M2_mean   — mean |magnetisation| / site, mean (mag/site)²
            Cv                — specific heat per site: Var(E/site) / T²
            chi               — susceptibility per site: Var(M/site) / T
        """
        n2 = self.n ** 2

        # Thermalisation
        for _ in range(n_therm):
            self.metropolis_step(temperature)

        # Measurement
        e_arr = np.empty(n_measure)
        m_arr = np.empty(n_measure)
        for k in range(n_measure):
            self.metropolis_step(temperature)
            e_arr[k] = self.energy() / n2
            m_arr[k] = abs(self.magnetization()) / n2

        e_mean = float(np.mean(e_arr))
        e2_mean = float(np.mean(e_arr ** 2))
        m_mean = float(np.mean(m_arr))
        m2_mean = float(np.mean(m_arr ** 2))

        cv = (e2_mean - e_mean ** 2) / (temperature ** 2)
        chi = (m2_mean - m_mean ** 2) / temperature

        return {
            "E_mean": e_mean,
            "E2_mean": e2_mean,
            "M_mean": m_mean,
            "M2_mean": m2_mean,
            "Cv": cv,
            "chi": chi,
        }
