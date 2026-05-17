"""
Simulation Animations
========================

Animated visualizations for Ising model evolution and particle tracks.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from typing import List, Optional


class SimulationAnimator:
    """Create animations of Monte Carlo simulation processes."""

    @staticmethod
    def ising_evolution(
        frames: List[np.ndarray],
        interval: int = 100,
        title: str = "Ising Model Evolution",
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Animate Ising model spin evolution.

        Args:
            frames: List of 2D spin arrays at each time step.
            interval: Milliseconds between frames.
            title: Plot title.
            save_path: Optional path to save as gif/mp4.

        Returns:
            FuncAnimation object.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = mcolors.ListedColormap(["#1f77b4", "#d62728"])
        im = ax.imshow((frames[0] + 1) / 2, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=14)
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                       verticalalignment="top", color="white",
                       bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        def update(frame_idx):
            im.set_data((frames[frame_idx] + 1) / 2)
            m = np.mean(frames[frame_idx])
            text.set_text(f"Step {frame_idx} | <m> = {m:.3f}")
            return [im, text]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

        if save_path:
            anim.save(save_path, writer="pillow", fps=1000 // interval)

        return anim

    @staticmethod
    def path_evolution(
        paths: np.ndarray,
        interval: int = 50,
        max_paths: int = 50,
        title: str = "Simulation Path Evolution",
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Animate growing simulation paths.

        Args:
            paths: Array of shape (n_sims, n_steps).
            interval: Milliseconds between frames.
            max_paths: Maximum paths to show.
            title: Plot title.
            save_path: Optional save path.

        Returns:
            FuncAnimation object.
        """
        n_paths = min(paths.shape[0], max_paths)
        n_steps = paths.shape[1]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, n_steps)
        y_min, y_max = paths[:n_paths].min() * 0.95, paths[:n_paths].max() * 1.05
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        lines = [ax.plot([], [], alpha=0.3, linewidth=0.5, color="steelblue")[0]
                 for _ in range(n_paths)]

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(np.arange(frame + 1), paths[i, :frame + 1])
            return lines

        anim = FuncAnimation(fig, update, frames=range(1, n_steps, max(1, n_steps // 200)),
                             interval=interval, blit=True)

        if save_path:
            anim.save(save_path, writer="pillow", fps=1000 // interval)

        return anim
