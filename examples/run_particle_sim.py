"""
Example: Photon Transport Through Water Phantom
==================================================

Demonstrates photon transport simulation through a voxelized water phantom,
computing depth-dose profiles and energy deposition statistics.
"""

import numpy as np
from montecarlo.particle_transport import PhotonTransport, MaterialDatabase, VoxelGeometry
from montecarlo.particle_transport.dose import DoseCalculator


def main():
    print("=" * 60)
    print("Photon Transport Simulation Example")
    print("=" * 60)

    # Create a water phantom
    geom = VoxelGeometry(
        dimensions=(20, 20, 40),
        voxel_size=(0.5, 0.5, 0.5),
    )
    water_id = geom.assign_material(MaterialDatabase.get("water"))
    geom.material_grid[:, :, :] = water_id

    # Insert a bone region
    bone_id = geom.assign_material(MaterialDatabase.get("bone"))
    geom.fill_box((3.0, 3.0, 8.0), (7.0, 7.0, 12.0), bone_id)

    print(f"Phantom size: {geom.physical_size} cm")
    print(f"Voxels: {geom.nx} x {geom.ny} x {geom.nz}")

    # Run photon transport
    sim = PhotonTransport(
        geometry=geom,
        source_position=np.array([5.0, 5.0, -1.0]),
        source_energy=100.0,  # 100 keV
        n_simulations=5000,
        seed=42,
    )

    result = sim.run(show_progress=True)
    print(result.summary())

    # Depth-dose profile
    depths, dose = sim.depth_dose_profile(axis=2)
    print("\nDepth-Dose Profile (first 10 bins):")
    for d, dv in zip(depths[:10], dose[:10]):
        bar = "#" * int(dv / max(max(dose), 1) * 40)
        print(f"  {d:6.2f} cm: {bar} ({dv:.4f})")

    print(f"\nTotal tracks: {len(sim.tracks)}")
    avg_interactions = np.mean([len(t.interactions) for t in sim.tracks])
    print(f"Avg interactions per photon: {avg_interactions:.2f}")


if __name__ == "__main__":
    main()
