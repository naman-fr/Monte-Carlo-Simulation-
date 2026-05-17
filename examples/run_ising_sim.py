"""
Example: 2D Ising Model Simulation
=====================================

Demonstrates Metropolis and Wolff algorithms, phase transition analysis.
"""

import numpy as np
from montecarlo.ising import IsingModel, MetropolisSampler, WolffSampler


def main():
    print("=" * 60)
    print("2D Ising Model Simulation Example")
    print("=" * 60)

    T_c = IsingModel.critical_temperature()
    print(f"Exact critical temperature: T_c = {T_c:.4f}")

    # Low temperature simulation
    print("\n--- Metropolis at T = 1.5 (ordered phase) ---")
    model = IsingModel(L=32, temperature=1.5, seed=42)
    sampler = MetropolisSampler(model, n_sweeps=5000, n_thermalize=2000)
    history = sampler.run()
    print(f"  <E> = {np.mean(history['energy']):.4f}")
    print(f"  <|m|> = {np.mean(history['abs_magnetization']):.4f}")

    # High temperature
    print("\n--- Metropolis at T = 3.5 (disordered phase) ---")
    model = IsingModel(L=32, temperature=3.5, seed=42)
    sampler = MetropolisSampler(model, n_sweeps=5000, n_thermalize=2000)
    history = sampler.run()
    print(f"  <E> = {np.mean(history['energy']):.4f}")
    print(f"  <|m|> = {np.mean(history['abs_magnetization']):.4f}")

    # Wolff algorithm near T_c
    print(f"\n--- Wolff at T = T_c = {T_c:.4f} ---")
    model = IsingModel(L=32, temperature=T_c, seed=42)
    wolff = WolffSampler(model, n_clusters=5000, n_thermalize=1000)
    history = wolff.run()
    print(f"  <E> = {np.mean(history['energy']):.4f}")
    print(f"  <|m|> = {np.mean(history['abs_magnetization']):.4f}")
    print(f"  <cluster_size> = {np.mean(history['cluster_size']):.4f}")

    # Temperature sweep
    print("\n--- Temperature Sweep (Metropolis) ---")
    model = IsingModel(L=16, seed=42)
    sampler = MetropolisSampler(model)
    results = sampler.temperature_sweep(T_range=(1.5, 3.5), n_temps=10, n_sweeps=3000, n_thermalize=1000)

    print(f"  {'T':>6s} | {'<E>':>8s} | {'<|m|>':>8s} | {'C_v':>8s} | {'chi':>8s}")
    print("  " + "-" * 50)
    for i in range(len(results["temperatures"])):
        print(f"  {results['temperatures'][i]:6.3f} | "
              f"{results['energy'][i]:8.4f} | "
              f"{results['abs_magnetization'][i]:8.4f} | "
              f"{results['specific_heat'][i]:8.4f} | "
              f"{results['susceptibility'][i]:8.4f}")


if __name__ == "__main__":
    main()
