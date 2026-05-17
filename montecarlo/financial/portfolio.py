"""
Portfolio Monte Carlo Simulation
==================================

Geometric Brownian Motion (GBM) based portfolio simulation with
multi-asset correlation support. Inspired by pandas-montecarlo.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from montecarlo.core.engine import MonteCarloSimulation


class PortfolioSimulator(MonteCarloSimulation):
    """Monte Carlo simulator for portfolio returns and price paths.

    Args:
        initial_value: Starting portfolio value.
        expected_return: Annualized expected return (e.g., 0.08 for 8%).
        volatility: Annualized volatility (e.g., 0.20 for 20%).
        time_horizon: Simulation period in years.
        n_steps: Number of time steps (trading days).
        n_simulations: Number of MC paths.
        seed: Random seed.
    """

    def __init__(
        self,
        initial_value: float = 10000.0,
        expected_return: float = 0.08,
        volatility: float = 0.20,
        time_horizon: float = 1.0,
        n_steps: int = 252,
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__(n_simulations=n_simulations, seed=seed, name="PortfolioSimulator")
        self.initial_value = initial_value
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self.dt = time_horizon / n_steps

    def _simulate_single(self, rng: np.random.Generator) -> np.ndarray:
        """Simulate a single portfolio path using GBM."""
        path = np.zeros(self.n_steps + 1)
        path[0] = self.initial_value
        drift = (self.expected_return - 0.5 * self.volatility**2) * self.dt
        diffusion = self.volatility * np.sqrt(self.dt)

        for t in range(1, self.n_steps + 1):
            z = rng.standard_normal()
            path[t] = path[t - 1] * np.exp(drift + diffusion * z)

        return path

    def simulate_correlated(
        self,
        assets: Dict[str, Tuple[float, float, float]],
        correlation_matrix: np.ndarray,
        n_steps: int = 252,
    ) -> Dict[str, np.ndarray]:
        """Simulate correlated multi-asset portfolio paths.

        Args:
            assets: Dict of {name: (initial_value, expected_return, volatility)}.
            correlation_matrix: Correlation matrix between assets.
            n_steps: Number of time steps.

        Returns:
            Dict of {name: (n_simulations, n_steps+1) path array}.
        """
        names = list(assets.keys())
        n_assets = len(names)
        dt = self.time_horizon / n_steps
        L = np.linalg.cholesky(correlation_matrix)

        results = {name: np.zeros((self.n_simulations, n_steps + 1)) for name in names}
        for name in names:
            results[name][:, 0] = assets[name][0]

        for sim in range(self.n_simulations):
            for t in range(1, n_steps + 1):
                z = self._rng.standard_normal(n_assets)
                correlated_z = L @ z
                for i, name in enumerate(names):
                    S0, mu, sigma = assets[name]
                    drift = (mu - 0.5 * sigma**2) * dt
                    diffusion = sigma * np.sqrt(dt) * correlated_z[i]
                    results[name][sim, t] = results[name][sim, t - 1] * np.exp(drift + diffusion)

        return results

    def max_drawdown(self, path: np.ndarray) -> float:
        """Compute maximum drawdown for a price path."""
        peak = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        return float(np.min(drawdown))

    def terminal_wealth_distribution(self) -> Dict[str, float]:
        """Analyze the distribution of terminal portfolio values."""
        if self._results is None:
            raise ValueError("Run simulation first")
        terminal = self._results.samples[:, -1]
        return {
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "std": float(np.std(terminal)),
            "min": float(np.min(terminal)),
            "max": float(np.max(terminal)),
            "prob_profit": float(np.mean(terminal > self.initial_value)),
            "prob_double": float(np.mean(terminal > 2 * self.initial_value)),
            "prob_loss_50pct": float(np.mean(terminal < 0.5 * self.initial_value)),
            "cagr_mean": float((np.mean(terminal) / self.initial_value) ** (1 / self.time_horizon) - 1),
        }
