"""
Metropolis-Hastings Algorithm for 2D Ising Model
====================================================

Implements the single-spin-flip Metropolis algorithm with optional
checkerboard decomposition for parallelism, inspired by NVIDIA/ising-gpu.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from montecarlo.ising.model import IsingModel


class MetropolisSampler:
    """Metropolis-Hastings sampler for the 2D Ising model.

    Args:
        model: IsingModel instance.
        n_sweeps: Number of full lattice sweeps.
        n_thermalize: Thermalization sweeps before measurement.
        measure_interval: Measure observables every N sweeps.
    """

    def __init__(
        self,
        model: IsingModel,
        n_sweeps: int = 10000,
        n_thermalize: int = 1000,
        measure_interval: int = 10,
    ):
        self.model = model
        self.n_sweeps = n_sweeps
        self.n_thermalize = n_thermalize
        self.measure_interval = measure_interval
        self._history: Dict[str, List[float]] = {
            "energy": [], "magnetization": [], "abs_magnetization": [],
        }

    def sweep(self):
        """Perform one full Metropolis sweep (N spin flip attempts)."""
        L = self.model.L
        for _ in range(self.model.N):
            i = self.model.rng.integers(0, L)
            j = self.model.rng.integers(0, L)
            dE = self.model.delta_energy(i, j)

            if dE <= 0 or self.model.rng.random() < np.exp(-self.model.beta * dE):
                self.model.spins[i, j] *= -1

    def checkerboard_sweep(self):
        """Vectorized checkerboard Metropolis sweep (inspired by NVIDIA).

        Updates black and white sublattices separately for potential
        parallelization. All spins in a sublattice are independent.
        """
        L = self.model.L
        beta = self.model.beta

        for parity in [0, 1]:
            # Create checkerboard mask
            rows, cols = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
            mask = (rows + cols) % 2 == parity

            # Compute neighbor sums for all masked sites
            s = self.model.spins
            neighbors = (
                np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0)
                + np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1)
            )

            # Delta E for flipping each spin
            dE = 2.0 * self.model.J * s * neighbors + 2.0 * self.model.h * s

            # Accept/reject
            accept = (dE <= 0) | (self.model.rng.random((L, L)) < np.exp(-beta * dE))
            flip = mask & accept
            self.model.spins[flip] *= -1

    def run(self, use_checkerboard: bool = True) -> Dict[str, np.ndarray]:
        """Run the Metropolis simulation.

        Args:
            use_checkerboard: Use vectorized checkerboard update.

        Returns:
            Dictionary of observable time series.
        """
        sweep_fn = self.checkerboard_sweep if use_checkerboard else self.sweep

        # Thermalization
        for _ in range(self.n_thermalize):
            sweep_fn()

        # Production
        self._history = {"energy": [], "magnetization": [], "abs_magnetization": []}
        for sweep_idx in range(self.n_sweeps):
            sweep_fn()
            if sweep_idx % self.measure_interval == 0:
                self._history["energy"].append(self.model.energy_per_spin())
                self._history["magnetization"].append(self.model.magnetization_per_spin())
                self._history["abs_magnetization"].append(self.model.abs_magnetization_per_spin())

        return {k: np.array(v) for k, v in self._history.items()}

    def temperature_sweep(
        self,
        T_range: Tuple[float, float] = (1.0, 4.0),
        n_temps: int = 30,
        n_sweeps: int = 5000,
        n_thermalize: int = 2000,
    ) -> Dict[str, np.ndarray]:
        """Sweep over temperatures to map the phase transition.

        Returns:
            Dict with arrays: temperatures, energy, magnetization,
            specific_heat, susceptibility.
        """
        temps = np.linspace(T_range[0], T_range[1], n_temps)
        results = {
            "temperatures": temps,
            "energy": np.zeros(n_temps),
            "abs_magnetization": np.zeros(n_temps),
            "specific_heat": np.zeros(n_temps),
            "susceptibility": np.zeros(n_temps),
        }

        for idx, T in enumerate(temps):
            self.model.set_temperature(T)
            self.model.initialize_hot()
            self.n_sweeps = n_sweeps
            self.n_thermalize = n_thermalize
            history = self.run()

            E = history["energy"]
            M = history["abs_magnetization"]
            N = self.model.N

            results["energy"][idx] = np.mean(E)
            results["abs_magnetization"][idx] = np.mean(M)
            results["specific_heat"][idx] = N * np.var(E) / T**2
            results["susceptibility"][idx] = N * np.var(M) / T

        return results
