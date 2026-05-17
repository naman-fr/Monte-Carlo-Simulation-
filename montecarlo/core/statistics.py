"""
Statistical Analysis Utilities
================================

Comprehensive statistical analysis for Monte Carlo simulation outputs.
"""

from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from typing import Dict, Optional, Tuple


class StatisticalAnalyzer:
    """Statistical analysis toolkit for MC simulation results.

    Args:
        data: 1D array of simulation samples.
    """

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data).flatten()
        self.n = len(self.data)

    def descriptive_stats(self) -> Dict[str, float]:
        """Compute comprehensive descriptive statistics."""
        return {
            "n": self.n,
            "mean": float(np.mean(self.data)),
            "median": float(np.median(self.data)),
            "std": float(np.std(self.data, ddof=1)),
            "variance": float(np.var(self.data, ddof=1)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "range": float(np.ptp(self.data)),
            "skewness": float(sp_stats.skew(self.data)),
            "kurtosis": float(sp_stats.kurtosis(self.data)),
            "sem": float(sp_stats.sem(self.data)),
        }

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for the mean.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%).

        Returns:
            Tuple of (lower, upper) bounds.
        """
        se = sp_stats.sem(self.data)
        h = se * sp_stats.t.ppf((1 + confidence) / 2, self.n - 1)
        mean = np.mean(self.data)
        return (float(mean - h), float(mean + h))

    def percentiles(self, quantiles=(5, 10, 25, 50, 75, 90, 95)) -> Dict[str, float]:
        """Compute percentiles of the data."""
        return {f"p{q}": float(np.percentile(self.data, q)) for q in quantiles}

    def histogram(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram counts and bin edges."""
        counts, edges = np.histogram(self.data, bins=bins, density=True)
        return counts, edges

    def kde(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Kernel Density Estimation.

        Returns:
            Tuple of (x_grid, density) arrays.
        """
        kernel = sp_stats.gaussian_kde(self.data)
        x_min, x_max = self.data.min(), self.data.max()
        margin = 0.1 * (x_max - x_min)
        x_grid = np.linspace(x_min - margin, x_max + margin, n_points)
        return x_grid, kernel(x_grid)

    def normality_test(self) -> Dict[str, float]:
        """Shapiro-Wilk normality test (for n <= 5000)."""
        sample = self.data[:5000] if self.n > 5000 else self.data
        stat, pvalue = sp_stats.shapiro(sample)
        return {"statistic": float(stat), "p_value": float(pvalue)}

    def autocorrelation(self, max_lag: int = 50) -> np.ndarray:
        """Compute autocorrelation function up to max_lag."""
        x = self.data - np.mean(self.data)
        c0 = np.sum(x ** 2) / self.n
        if c0 == 0:
            return np.zeros(max_lag)
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            acf[lag] = np.sum(x[: self.n - lag] * x[lag:]) / (self.n * c0)
        return acf

    def effective_sample_size(self) -> float:
        """Estimate effective sample size accounting for autocorrelation."""
        acf = self.autocorrelation(max_lag=min(self.n // 2, 200))
        # Sum until first negative autocorrelation
        tau = 1.0
        for k in range(1, len(acf)):
            if acf[k] < 0:
                break
            tau += 2 * acf[k]
        return self.n / tau

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean.

        Args:
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level.
            seed: Random seed.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        rng = np.random.default_rng(seed)
        means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(self.data, size=self.n, replace=True)
            means[i] = np.mean(sample)
        alpha = 1 - confidence
        lower = np.percentile(means, 100 * alpha / 2)
        upper = np.percentile(means, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))
