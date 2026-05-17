"""
Performance Benchmarks for Monte Carlo Engine
================================================
"""

import time
import numpy as np
from montecarlo.core.engine import MonteCarloSimulation
from montecarlo.ising.model import IsingModel
from montecarlo.ising.metropolis import MetropolisSampler


class PiEstimator(MonteCarloSimulation):
    def _simulate_single(self, rng):
        x, y = rng.random(), rng.random()
        return 1.0 if x**2 + y**2 <= 1.0 else 0.0

    def _aggregate_results(self, samples):
        return 4.0 * samples


def benchmark_pi(n_list=(1000, 10000, 100000, 500000)):
    print("Pi Estimation Benchmark")
    print("-" * 50)
    for n in n_list:
        start = time.time()
        sim = PiEstimator(n_simulations=n, seed=42)
        result = sim.run(show_progress=False)
        elapsed = time.time() - start
        error = abs(result.mean - np.pi)
        print(f"  N={n:>8d} | Pi={result.mean:.6f} | Err={error:.6f} | Time={elapsed:.3f}s")


def benchmark_ising(sizes=(8, 16, 32, 64)):
    print("\nIsing Model Benchmark (1000 Metropolis sweeps)")
    print("-" * 50)
    for L in sizes:
        model = IsingModel(L=L, temperature=2.269, seed=42)
        sampler = MetropolisSampler(model, n_sweeps=1000, n_thermalize=500, measure_interval=10)
        start = time.time()
        sampler.run()
        elapsed = time.time() - start
        print(f"  L={L:>3d} (N={L*L:>5d}) | Time={elapsed:.3f}s | "
              f"Sweeps/s={1000/elapsed:.0f}")


if __name__ == "__main__":
    benchmark_pi()
    benchmark_ising()
