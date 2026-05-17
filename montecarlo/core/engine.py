"""
Core Monte Carlo Simulation Engine
====================================

Provides the abstract base class and result container for all MC simulations.
Inspired by the architecture of GGEMS (GPU Monte Carlo) and OpenTOPAS
(particle therapy simulation), adapted to a Pythonic interface.

Design Principles:
    - Reproducibility via explicit seed management
    - Progress tracking with convergence monitoring
    - Structured results with statistical summaries
    - Extensibility through the template method pattern
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results.

    Attributes:
        samples: Raw simulation output samples (N_sims x N_steps or N_sims).
        statistics: Dictionary of computed statistics (mean, std, CI, etc.).
        metadata: Additional metadata about the simulation run.
        elapsed_time: Wall-clock time for the simulation in seconds.
        converged: Whether the simulation met convergence criteria.
    """

    samples: np.ndarray
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0
    converged: bool = False

    @property
    def mean(self) -> float:
        """Mean of terminal simulation values."""
        if self.samples.ndim == 1:
            return float(np.mean(self.samples))
        return float(np.mean(self.samples[:, -1]))

    @property
    def std(self) -> float:
        """Standard deviation of terminal simulation values."""
        if self.samples.ndim == 1:
            return float(np.std(self.samples, ddof=1))
        return float(np.std(self.samples[:, -1], ddof=1))

    @property
    def confidence_interval(self) -> tuple:
        """95% confidence interval for the mean."""
        n = len(self.samples) if self.samples.ndim == 1 else self.samples.shape[0]
        se = self.std / np.sqrt(n)
        return (self.mean - 1.96 * se, self.mean + 1.96 * se)

    def summary(self) -> str:
        """Return a formatted summary of the simulation results."""
        ci = self.confidence_interval
        lines = [
            "=" * 60,
            "Monte Carlo Simulation Results",
            "=" * 60,
            f"  Samples:            {self.samples.shape}",
            f"  Mean:               {self.mean:.6f}",
            f"  Std Dev:            {self.std:.6f}",
            f"  95% CI:             [{ci[0]:.6f}, {ci[1]:.6f}]",
            f"  Elapsed Time:       {self.elapsed_time:.3f}s",
            f"  Converged:          {self.converged}",
        ]
        if self.statistics:
            lines.append("  Additional Stats:")
            for key, val in self.statistics.items():
                lines.append(f"    {key}: {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


class MonteCarloSimulation(ABC):
    """Abstract base class for Monte Carlo simulations.

    All simulation modules (particle transport, financial, ising, nuclear)
    inherit from this class and implement the `_simulate_single` method.

    This follows the Template Method pattern: the `run()` method orchestrates
    the simulation loop while subclasses define the per-sample logic.

    Args:
        n_simulations: Number of independent MC samples to generate.
        seed: Random seed for reproducibility. None for random.
        name: Human-readable name for the simulation.

    Example:
        >>> class PiEstimator(MonteCarloSimulation):
        ...     def _simulate_single(self, rng):
        ...         x, y = rng.random(), rng.random()
        ...         return 1.0 if x**2 + y**2 <= 1.0 else 0.0
        ...
        ...     def _aggregate_results(self, samples):
        ...         return 4.0 * samples
        ...
        >>> sim = PiEstimator(n_simulations=100000, seed=42)
        >>> result = sim.run()
        >>> print(f"Pi ≈ {result.mean:.4f}")
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        seed: Optional[int] = None,
        name: str = "MonteCarloSimulation",
    ):
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")
        self.n_simulations = n_simulations
        self.seed = seed
        self.name = name
        self._rng = np.random.default_rng(seed)
        self._results: Optional[SimulationResult] = None

        logger.info(
            "Initialized %s with n_simulations=%d, seed=%s",
            name, n_simulations, seed
        )

    @abstractmethod
    def _simulate_single(self, rng: np.random.Generator) -> Union[float, np.ndarray]:
        """Run a single Monte Carlo sample.

        Args:
            rng: NumPy random number generator instance.

        Returns:
            A scalar value or 1D array representing the simulation outcome.
        """
        pass

    def _aggregate_results(self, samples: np.ndarray) -> np.ndarray:
        """Optional post-processing of raw samples.

        Override this method to apply transformations to the collected
        samples before statistical analysis. Default is identity.

        Args:
            samples: Array of shape (n_simulations,) or (n_simulations, n_steps).

        Returns:
            Transformed samples array.
        """
        return samples

    def _check_convergence(self, samples: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if simulation has converged using running mean stability.

        Args:
            samples: Collected samples so far.
            tolerance: Relative tolerance for convergence.

        Returns:
            True if the running mean has stabilized.
        """
        if len(samples) < 100:
            return False
        # Compare mean of last 10% to overall mean
        terminal = samples[:, -1] if samples.ndim > 1 else samples
        n = len(terminal)
        cutoff = max(int(0.9 * n), 1)
        overall_mean = np.mean(terminal)
        recent_mean = np.mean(terminal[cutoff:])
        if overall_mean == 0:
            return abs(recent_mean) < tolerance
        return abs((recent_mean - overall_mean) / overall_mean) < tolerance

    def run(self, show_progress: bool = True) -> SimulationResult:
        """Execute the Monte Carlo simulation.

        Args:
            show_progress: Whether to display a progress bar.

        Returns:
            SimulationResult containing samples and statistics.
        """
        logger.info("Starting simulation: %s", self.name)
        start_time = time.time()

        # Run all simulations
        results_list: List[Any] = []
        iterator = range(self.n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc=self.name, unit="sim")

        for i in iterator:
            # Create independent RNG streams for reproducibility
            child_rng = np.random.default_rng(
                self._rng.integers(0, 2**63)
            )
            result = self._simulate_single(child_rng)
            results_list.append(result)

        # Stack results
        samples = np.array(results_list)
        samples = self._aggregate_results(samples)

        elapsed = time.time() - start_time
        converged = self._check_convergence(samples)

        # Compute statistics
        if samples.ndim == 1:
            terminal = samples
        else:
            terminal = samples[:, -1]

        statistics = {
            "min": float(np.min(terminal)),
            "max": float(np.max(terminal)),
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "std": float(np.std(terminal, ddof=1)),
            "variance": float(np.var(terminal, ddof=1)),
            "skewness": float(self._skewness(terminal)),
            "kurtosis": float(self._kurtosis(terminal)),
            "percentile_5": float(np.percentile(terminal, 5)),
            "percentile_25": float(np.percentile(terminal, 25)),
            "percentile_75": float(np.percentile(terminal, 75)),
            "percentile_95": float(np.percentile(terminal, 95)),
        }

        self._results = SimulationResult(
            samples=samples,
            statistics=statistics,
            metadata={
                "name": self.name,
                "n_simulations": self.n_simulations,
                "seed": self.seed,
            },
            elapsed_time=elapsed,
            converged=converged,
        )

        logger.info(
            "Simulation complete in %.3fs. Mean=%.6f, Std=%.6f",
            elapsed, statistics["mean"], statistics["std"]
        )

        return self._results

    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        """Compute Fisher's skewness coefficient."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return float(n / ((n - 1) * (n - 2)) * np.sum(((data - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        return float(m4 / std**4 - 3.0)

    @property
    def results(self) -> Optional[SimulationResult]:
        """Access the last simulation results."""
        return self._results

    def reset(self, seed: Optional[int] = None):
        """Reset the simulation state.

        Args:
            seed: New random seed. If None, uses original seed.
        """
        self.seed = seed if seed is not None else self.seed
        self._rng = np.random.default_rng(self.seed)
        self._results = None
        logger.info("Reset simulation %s with seed=%s", self.name, self.seed)
