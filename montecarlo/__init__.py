"""
MonteCarloX — Industry-Grade Monte Carlo Simulation Engine
==========================================================

A comprehensive simulation framework combining techniques from:
- Particle transport physics (inspired by GGEMS & OpenTOPAS)
- Financial engineering (inspired by pandas-montecarlo)
- Statistical mechanics (inspired by NVIDIA Ising GPU)
- Nuclear/astroparticle physics (inspired by NuRadioMC)

Modules:
    core: Base simulation engine, RNG, statistics, convergence
    particle_transport: Photon transport, materials, geometry, dosimetry
    financial: Portfolio simulation, options pricing, risk analysis
    ising: 2D Ising model, Metropolis & Wolff algorithms
    nuclear: Neutrino interactions, cross-sections, detector simulation
    visualization: Publication-quality plots and animations
"""

__version__ = "1.0.0"
__author__ = "Naman"

from montecarlo.core.engine import MonteCarloSimulation, SimulationResult

__all__ = [
    "MonteCarloSimulation",
    "SimulationResult",
    "__version__",
]
