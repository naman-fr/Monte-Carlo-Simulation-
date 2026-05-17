"""
2D Ising Model
=================

Square lattice Ising model with observables computation.
Inspired by NVIDIA/ising-gpu checkerboard decomposition.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple


class IsingModel:
    """2D square-lattice Ising model.

    H = -J * sum_{<ij>} s_i * s_j - h * sum_i s_i

    Args:
        L: Lattice size (L x L).
        J: Coupling constant (positive = ferromagnetic).
        h: External magnetic field.
        temperature: Temperature in units of J/k_B.
        seed: Random seed.
    """

    def __init__(
        self,
        L: int = 32,
        J: float = 1.0,
        h: float = 0.0,
        temperature: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.L = L
        self.J = J
        self.h = h
        self.temperature = temperature
        self.beta = 1.0 / temperature if temperature > 0 else float('inf')
        self.rng = np.random.default_rng(seed)
        # Initialize random spin configuration
        self.spins = self.rng.choice([-1, 1], size=(L, L))

    @property
    def N(self) -> int:
        """Total number of spins."""
        return self.L * self.L

    def energy(self) -> float:
        """Compute total energy of the current configuration."""
        E = 0.0
        s = self.spins
        # Nearest-neighbor interactions with periodic BC
        E -= self.J * np.sum(s * np.roll(s, 1, axis=0))
        E -= self.J * np.sum(s * np.roll(s, 1, axis=1))
        E -= self.h * np.sum(s)
        return float(E)

    def energy_per_spin(self) -> float:
        """Energy per spin."""
        return self.energy() / self.N

    def magnetization(self) -> float:
        """Total magnetization."""
        return float(np.sum(self.spins))

    def magnetization_per_spin(self) -> float:
        """Magnetization per spin."""
        return self.magnetization() / self.N

    def abs_magnetization_per_spin(self) -> float:
        """Absolute magnetization per spin (order parameter)."""
        return abs(self.magnetization()) / self.N

    def observables(self) -> Dict[str, float]:
        """Compute all observables for the current configuration."""
        return {
            "energy": self.energy(),
            "energy_per_spin": self.energy_per_spin(),
            "magnetization": self.magnetization(),
            "magnetization_per_spin": self.magnetization_per_spin(),
            "abs_magnetization_per_spin": self.abs_magnetization_per_spin(),
        }

    def initialize_cold(self):
        """All spins aligned (ground state)."""
        self.spins = np.ones((self.L, self.L), dtype=int)

    def initialize_hot(self):
        """Random spin configuration (high temperature)."""
        self.spins = self.rng.choice([-1, 1], size=(self.L, self.L))

    def set_temperature(self, T: float):
        """Update temperature and recalculate beta."""
        self.temperature = T
        self.beta = 1.0 / T if T > 0 else float('inf')

    def local_energy(self, i: int, j: int) -> float:
        """Energy contribution from spin at (i, j)."""
        L = self.L
        s = self.spins[i, j]
        neighbors = (
            self.spins[(i + 1) % L, j]
            + self.spins[(i - 1) % L, j]
            + self.spins[i, (j + 1) % L]
            + self.spins[i, (j - 1) % L]
        )
        return -self.J * s * neighbors - self.h * s

    def delta_energy(self, i: int, j: int) -> float:
        """Energy change if spin at (i, j) is flipped."""
        L = self.L
        s = self.spins[i, j]
        neighbors = (
            self.spins[(i + 1) % L, j]
            + self.spins[(i - 1) % L, j]
            + self.spins[i, (j + 1) % L]
            + self.spins[i, (j - 1) % L]
        )
        return 2.0 * self.J * s * neighbors + 2.0 * self.h * s

    @staticmethod
    def critical_temperature(J: float = 1.0) -> float:
        """Exact Onsager critical temperature for 2D Ising model."""
        return 2.0 * J / np.log(1.0 + np.sqrt(2.0))

    def correlation_function(self, max_r: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spin-spin correlation function C(r)."""
        if max_r is None:
            max_r = self.L // 2
        r_values = np.arange(max_r)
        corr = np.zeros(max_r)
        s = self.spins
        mean_s = np.mean(s)

        for r in range(max_r):
            c = np.mean(s * np.roll(s, r, axis=0)) - mean_s**2
            corr[r] = c

        # Normalize
        if corr[0] != 0:
            corr /= corr[0]
        return r_values, corr
