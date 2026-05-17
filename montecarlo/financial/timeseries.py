"""
Time Series Monte Carlo Simulation
=====================================

MC simulation on pandas Series data, similar to pandas-montecarlo API.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TimeSeriesMC:
    """Monte Carlo simulation on time series data.

    Simulates future paths based on historical return characteristics.

    Args:
        series: Pandas Series of prices or returns.
        is_returns: Whether the series contains returns (True) or prices (False).
    """

    def __init__(self, series: pd.Series, is_returns: bool = False):
        self.original = series.copy()
        if is_returns:
            self.returns = series.values
        else:
            self.returns = series.pct_change().dropna().values
        self.n_obs = len(self.returns)

    def simulate(
        self,
        n_simulations: int = 1000,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation on the time series.

        Args:
            n_simulations: Number of simulated paths.
            n_steps: Number of forward steps. Defaults to original length.
            seed: Random seed.

        Returns:
            DataFrame with original and simulated paths.
        """
        if n_steps is None:
            n_steps = self.n_obs

        rng = np.random.default_rng(seed)
        mu = np.mean(self.returns)
        sigma = np.std(self.returns, ddof=1)

        sims = np.zeros((n_steps, n_simulations + 1))
        sims[:min(n_steps, self.n_obs), 0] = self.returns[:min(n_steps, self.n_obs)]

        for i in range(n_simulations):
            sims[:, i + 1] = rng.normal(mu, sigma, size=n_steps)

        # Convert to cumulative returns
        cum_sims = np.cumprod(1 + sims, axis=0)

        columns = ["original"] + [f"sim_{i+1}" for i in range(n_simulations)]
        return pd.DataFrame(cum_sims, columns=columns)

    def stats(
        self,
        simulated_data: pd.DataFrame,
        bust: float = -0.1,
        goal: float = 1.0,
    ) -> Dict[str, float]:
        """Compute simulation statistics.

        Args:
            simulated_data: DataFrame from simulate().
            bust: Drawdown threshold for bust probability.
            goal: Return threshold for goal probability.

        Returns:
            Dictionary of statistics.
        """
        sim_cols = [c for c in simulated_data.columns if c != "original"]
        terminal = simulated_data[sim_cols].iloc[-1].values

        # Max drawdown per simulation
        max_drawdowns = []
        for col in sim_cols:
            path = simulated_data[col].values
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak
            max_drawdowns.append(np.min(dd))
        max_drawdowns = np.array(max_drawdowns)

        return {
            "min": float(np.min(terminal)),
            "max": float(np.max(terminal)),
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "std": float(np.std(terminal)),
            "maxdd_mean": float(np.mean(max_drawdowns)),
            "maxdd_min": float(np.min(max_drawdowns)),
            "maxdd_max": float(np.max(max_drawdowns)),
            "bust_probability": float(np.mean(max_drawdowns < bust)),
            "goal_probability": float(np.mean(terminal >= (1 + goal))),
        }

    def percentile_paths(
        self,
        simulated_data: pd.DataFrame,
        percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
    ) -> pd.DataFrame:
        """Extract percentile paths from simulation results."""
        sim_cols = [c for c in simulated_data.columns if c != "original"]
        sim_data = simulated_data[sim_cols].values

        result = {}
        for p in percentiles:
            result[f"p{p}"] = np.percentile(sim_data, p, axis=1)

        return pd.DataFrame(result, index=simulated_data.index)
