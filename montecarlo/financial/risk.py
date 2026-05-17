"""
Risk Analysis and Metrics
============================

VaR, CVaR, maximum drawdown, and probability of ruin calculations.
Inspired by pandas-montecarlo risk statistics.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional


class RiskAnalyzer:
    """Financial risk analysis toolkit.

    Args:
        returns: Array of returns or simulation paths.
    """

    def __init__(self, returns: Optional[np.ndarray] = None):
        self.returns = returns

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk at given confidence level.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%).

        Returns:
            VaR as a positive loss number.
        """
        if self.returns is None:
            raise ValueError("No return data provided")
        data = self.returns[:, -1] if self.returns.ndim > 1 else self.returns
        return float(-np.percentile(data, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall).

        Average loss exceeding VaR threshold.
        """
        data = self.returns[:, -1] if self.returns.ndim > 1 else self.returns
        var_threshold = np.percentile(data, (1 - confidence) * 100)
        tail_losses = data[data <= var_threshold]
        return float(-np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0

    def max_drawdown(self, path: Optional[np.ndarray] = None) -> float:
        """Maximum drawdown for a price path."""
        if path is None:
            if self.returns is not None and self.returns.ndim > 1:
                path = self.returns[0]
            else:
                raise ValueError("Provide a price path")
        peak = np.maximum.accumulate(path)
        dd = (path - peak) / peak
        return float(np.min(dd))

    def max_drawdown_distribution(self, paths: np.ndarray) -> Dict[str, float]:
        """Compute max drawdown statistics across multiple paths."""
        drawdowns = np.array([self.max_drawdown(p) for p in paths])
        return {
            "min": float(np.min(drawdowns)),
            "max": float(np.max(drawdowns)),
            "mean": float(np.mean(drawdowns)),
            "median": float(np.median(drawdowns)),
            "std": float(np.std(drawdowns)),
        }

    def probability_of_ruin(self, threshold: float = 0.0) -> float:
        """Probability of portfolio value falling below threshold."""
        if self.returns is None:
            raise ValueError("No data")
        if self.returns.ndim > 1:
            min_vals = np.min(self.returns, axis=1)
        else:
            min_vals = self.returns
        return float(np.mean(min_vals < threshold))

    def sharpe_ratio(
        self,
        risk_free_rate: float = 0.02,
        annualization_factor: float = 252,
    ) -> float:
        """Compute annualized Sharpe ratio."""
        data = self.returns[:, -1] if self.returns.ndim > 1 else self.returns
        excess = data - risk_free_rate / annualization_factor
        return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(annualization_factor))

    def sortino_ratio(
        self,
        risk_free_rate: float = 0.02,
        annualization_factor: float = 252,
    ) -> float:
        """Compute annualized Sortino ratio (downside risk only)."""
        data = self.returns[:, -1] if self.returns.ndim > 1 else self.returns
        excess = data - risk_free_rate / annualization_factor
        downside = excess[excess < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
        return float(np.mean(excess) / downside_std * np.sqrt(annualization_factor))

    def full_report(self) -> Dict[str, float]:
        """Generate comprehensive risk report."""
        report = {
            "var_95": self.var(0.95),
            "var_99": self.var(0.99),
            "cvar_95": self.cvar(0.95),
            "cvar_99": self.cvar(0.99),
        }
        if self.returns is not None and self.returns.ndim > 1:
            dd_stats = self.max_drawdown_distribution(self.returns)
            report.update({f"mdd_{k}": v for k, v in dd_stats.items()})
            report["prob_ruin_50pct"] = self.probability_of_ruin(
                self.returns[0, 0] * 0.5 if self.returns.shape[1] > 0 else 0
            )
        return report
