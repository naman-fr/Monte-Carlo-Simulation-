"""
Convergence Diagnostics for MCMC and MC Simulations
=====================================================

Provides tools to assess whether a Monte Carlo simulation has converged.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


class ConvergenceDiagnostics:
    """Convergence diagnostic tools for Monte Carlo simulations.

    Args:
        chains: List of 1D arrays, each representing an independent chain.
    """

    def __init__(self, chains: Optional[List[np.ndarray]] = None):
        self.chains = [np.asarray(c) for c in chains] if chains else []

    def add_chain(self, chain: np.ndarray):
        """Add a chain for multi-chain diagnostics."""
        self.chains.append(np.asarray(chain))

    def gelman_rubin(self) -> float:
        """Compute the Gelman-Rubin R-hat statistic.

        Requires at least 2 chains. Values close to 1.0 indicate convergence.

        Returns:
            R-hat value. Should be < 1.05 for convergence.
        """
        if len(self.chains) < 2:
            raise ValueError("Gelman-Rubin requires at least 2 chains")

        m = len(self.chains)
        n = min(len(c) for c in self.chains)
        chains = [c[:n] for c in self.chains]

        chain_means = np.array([np.mean(c) for c in chains])
        chain_vars = np.array([np.var(c, ddof=1) for c in chains])
        overall_mean = np.mean(chain_means)

        # Between-chain variance
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
        # Within-chain variance
        W = np.mean(chain_vars)

        # Pooled variance estimate
        var_hat = ((n - 1) / n) * W + (1 / n) * B
        R_hat = np.sqrt(var_hat / W) if W > 0 else float('inf')
        return float(R_hat)

    def effective_sample_size(self, chain: Optional[np.ndarray] = None) -> float:
        """Estimate effective sample size for a single chain.

        Args:
            chain: 1D array. Uses first chain if None.

        Returns:
            Estimated effective sample size.
        """
        if chain is None:
            if not self.chains:
                raise ValueError("No chains available")
            chain = self.chains[0]

        n = len(chain)
        x = chain - np.mean(chain)
        c0 = np.sum(x ** 2) / n
        if c0 == 0:
            return float(n)

        # Compute autocorrelation
        max_lag = min(n // 2, 500)
        tau = 1.0
        for lag in range(1, max_lag):
            rho = np.sum(x[:n - lag] * x[lag:]) / (n * c0)
            if rho < 0.05:
                break
            tau += 2 * rho

        return n / tau

    def running_mean(self, data: np.ndarray) -> np.ndarray:
        """Compute cumulative running mean.

        Args:
            data: 1D array of samples.

        Returns:
            Array of running means.
        """
        return np.cumsum(data) / np.arange(1, len(data) + 1)

    def running_variance(self, data: np.ndarray) -> np.ndarray:
        """Compute cumulative running variance using Welford's algorithm.

        Args:
            data: 1D array of samples.

        Returns:
            Array of running variances.
        """
        n = len(data)
        running_var = np.zeros(n)
        mean = 0.0
        M2 = 0.0
        for i in range(n):
            delta = data[i] - mean
            mean += delta / (i + 1)
            delta2 = data[i] - mean
            M2 += delta * delta2
            running_var[i] = M2 / (i + 1) if i > 0 else 0.0
        return running_var

    def geweke_test(
        self,
        chain: Optional[np.ndarray] = None,
        first_frac: float = 0.1,
        last_frac: float = 0.5,
    ) -> Dict[str, float]:
        """Geweke convergence diagnostic.

        Compares the mean of the first fraction and last fraction of a chain.

        Args:
            chain: 1D array. Uses first chain if None.
            first_frac: Fraction of chain for the first segment.
            last_frac: Fraction of chain for the last segment.

        Returns:
            Dictionary with z-score and p-value.
        """
        if chain is None:
            chain = self.chains[0]

        n = len(chain)
        n_first = int(first_frac * n)
        n_last = int(last_frac * n)

        first = chain[:n_first]
        last = chain[-n_last:]

        mean_first = np.mean(first)
        mean_last = np.mean(last)
        var_first = np.var(first, ddof=1) / len(first)
        var_last = np.var(last, ddof=1) / len(last)

        denom = np.sqrt(var_first + var_last)
        z_score = (mean_first - mean_last) / denom if denom > 0 else 0.0

        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return {"z_score": float(z_score), "p_value": float(p_value)}

    def batch_means_se(
        self,
        chain: Optional[np.ndarray] = None,
        n_batches: int = 20,
    ) -> float:
        """Estimate standard error using batch means method.

        Args:
            chain: 1D array. Uses first chain if None.
            n_batches: Number of batches.

        Returns:
            Estimated standard error of the mean.
        """
        if chain is None:
            chain = self.chains[0]

        n = len(chain)
        batch_size = n // n_batches
        batch_means = np.array([
            np.mean(chain[i * batch_size:(i + 1) * batch_size])
            for i in range(n_batches)
        ])
        return float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))

    def summary(self) -> Dict[str, float]:
        """Generate a convergence summary report."""
        report = {}
        if len(self.chains) >= 2:
            report["r_hat"] = self.gelman_rubin()
        if self.chains:
            report["ess"] = self.effective_sample_size()
            geweke = self.geweke_test()
            report["geweke_z"] = geweke["z_score"]
            report["geweke_p"] = geweke["p_value"]
            report["batch_se"] = self.batch_means_se()
        return report
