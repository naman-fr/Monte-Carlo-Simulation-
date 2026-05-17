"""
Wolff Cluster Algorithm for 2D Ising Model
=============================================

Cluster algorithm that flips correlated spin clusters, providing
much faster equilibration near the critical temperature.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from montecarlo.ising.model import IsingModel


class WolffSampler:
    """Wolff single-cluster algorithm for the Ising model.

    Near T_c, the Metropolis algorithm suffers from critical slowing down.
    The Wolff algorithm flips entire correlated clusters, dramatically
    reducing autocorrelation times.

    Args:
        model: IsingModel instance.
        n_clusters: Number of cluster flips to perform.
        n_thermalize: Thermalization steps.
        measure_interval: Measurement frequency.
    """

    def __init__(
        self,
        model: IsingModel,
        n_clusters: int = 10000,
        n_thermalize: int = 500,
        measure_interval: int = 1,
    ):
        self.model = model
        self.n_clusters = n_clusters
        self.n_thermalize = n_thermalize
        self.measure_interval = measure_interval

    def _add_probability(self) -> float:
        """Probability of adding a neighbor to the cluster."""
        return 1.0 - np.exp(-2.0 * self.model.beta * self.model.J)

    def flip_cluster(self) -> int:
        """Grow and flip a single Wolff cluster.

        Returns:
            Size of the flipped cluster.
        """
        L = self.model.L
        p_add = self._add_probability()

        # Pick random seed spin
        i0 = self.model.rng.integers(0, L)
        j0 = self.model.rng.integers(0, L)
        seed_spin = self.model.spins[i0, j0]

        # BFS/stack-based cluster growth
        cluster = set()
        stack = [(i0, j0)]
        cluster.add((i0, j0))

        while stack:
            i, j = stack.pop()
            # Check all 4 neighbors
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni = (i + di) % L
                nj = (j + dj) % L
                if (ni, nj) not in cluster:
                    if self.model.spins[ni, nj] == seed_spin:
                        if self.model.rng.random() < p_add:
                            cluster.add((ni, nj))
                            stack.append((ni, nj))

        # Flip entire cluster
        for (i, j) in cluster:
            self.model.spins[i, j] *= -1

        return len(cluster)

    def run(self) -> Dict[str, np.ndarray]:
        """Run the Wolff cluster simulation.

        Returns:
            Dictionary of observable time series.
        """
        # Thermalization
        for _ in range(self.n_thermalize):
            self.flip_cluster()

        # Production
        history = {"energy": [], "magnetization": [], "abs_magnetization": [], "cluster_size": []}
        for step in range(self.n_clusters):
            cluster_size = self.flip_cluster()
            if step % self.measure_interval == 0:
                history["energy"].append(self.model.energy_per_spin())
                history["magnetization"].append(self.model.magnetization_per_spin())
                history["abs_magnetization"].append(self.model.abs_magnetization_per_spin())
                history["cluster_size"].append(cluster_size / self.model.N)

        return {k: np.array(v) for k, v in history.items()}

    def temperature_sweep(
        self,
        T_range: Tuple[float, float] = (1.0, 4.0),
        n_temps: int = 30,
        n_clusters: int = 5000,
        n_thermalize: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Temperature sweep using Wolff algorithm."""
        temps = np.linspace(T_range[0], T_range[1], n_temps)
        results = {
            "temperatures": temps,
            "energy": np.zeros(n_temps),
            "abs_magnetization": np.zeros(n_temps),
            "specific_heat": np.zeros(n_temps),
            "susceptibility": np.zeros(n_temps),
            "mean_cluster_size": np.zeros(n_temps),
        }

        for idx, T in enumerate(temps):
            self.model.set_temperature(T)
            self.model.initialize_hot()
            self.n_clusters = n_clusters
            self.n_thermalize = n_thermalize
            history = self.run()

            E = history["energy"]
            M = history["abs_magnetization"]
            N = self.model.N

            results["energy"][idx] = np.mean(E)
            results["abs_magnetization"][idx] = np.mean(M)
            results["specific_heat"][idx] = N * np.var(E) / T**2
            results["susceptibility"][idx] = N * np.var(M) / T
            results["mean_cluster_size"][idx] = np.mean(history["cluster_size"])

        return results

    def estimate_critical_temperature(
        self,
        T_range: Tuple[float, float] = (2.0, 2.6),
        n_temps: int = 20,
    ) -> float:
        """Estimate T_c from the peak of the susceptibility."""
        results = self.temperature_sweep(T_range, n_temps, n_clusters=3000, n_thermalize=500)
        peak_idx = np.argmax(results["susceptibility"])
        return float(results["temperatures"][peak_idx])
