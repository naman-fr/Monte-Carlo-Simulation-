"""
Example: Neutrino Monte Carlo Simulation
============================================

Demonstrates neutrino event generation, cross-section calculations,
and detector response simulation.
"""

import numpy as np
from montecarlo.nuclear import NeutrinoSimulator, CrossSectionCalculator, DetectorSimulator
from montecarlo.nuclear.cross_sections import InteractionType
from montecarlo.nuclear.detector import DetectorConfig


def main():
    print("=" * 60)
    print("Neutrino Monte Carlo Simulation Example")
    print("=" * 60)

    # Cross-section calculation
    print("\n--- Neutrino Cross-Sections ---")
    xs = CrossSectionCalculator(model="connolly")
    energies = [1e14, 1e16, 1e18, 1e20]
    for E in energies:
        sigma = xs.total_cross_section(E)
        L = xs.interaction_length(E)
        print(f"  E = {E:.0e} eV: σ = {sigma:.3e} cm², L = {L:.3e} cm")

    # Neutrino simulation
    print("\n--- Neutrino Event Generation ---")
    sim = NeutrinoSimulator(
        energy_range=(1e15, 1e19),
        spectral_index=2.0,
        n_simulations=5000,
        seed=42,
    )
    result = sim.run(show_progress=True)

    interacting = [e for e in sim.events if e.interaction_happened]
    print(f"  Total events generated: {len(sim.events)}")
    print(f"  Interacting events: {len(interacting)}")

    if interacting:
        cc = sum(1 for e in interacting if e.interaction == InteractionType.CC)
        nc = len(interacting) - cc
        print(f"  CC interactions: {cc}")
        print(f"  NC interactions: {nc}")
        print(f"  Mean inelasticity: {np.mean([e.inelasticity for e in interacting]):.3f}")

    # Detector simulation
    print("\n--- Detector Response ---")
    config = DetectorConfig(
        shape="cylindrical",
        radius=500.0,
        height=1000.0,
        depth=2000.0,
        energy_threshold=1e16,
    )
    det = DetectorSimulator(config)
    print(f"  Effective volume: {det.effective_volume:.3e} m³")
    print(f"  Effective area: {det.effective_area:.3e} m²")

    # Expected events
    expected = det.expected_events(observation_years=10.0)
    print("\n  Expected events (10 years, cosmogenic flux):")
    for k, v in expected.items():
        print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
