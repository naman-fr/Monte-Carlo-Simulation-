"""
Core Monte Carlo engine components.

Provides the foundational building blocks for all simulation modules:
- MonteCarloSimulation: Abstract base class for simulations
- Random number generation with multiple backends
- Statistical analysis and convergence diagnostics
"""

from montecarlo.core.engine import MonteCarloSimulation, SimulationResult
from montecarlo.core.random_gen import RandomGenerator
from montecarlo.core.statistics import StatisticalAnalyzer
from montecarlo.core.convergence import ConvergenceDiagnostics

__all__ = [
    "MonteCarloSimulation",
    "SimulationResult",
    "RandomGenerator",
    "StatisticalAnalyzer",
    "ConvergenceDiagnostics",
]
