"""
Publication-Quality Simulation Plots
========================================

Unified visualization for all MC simulation modules.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Dict, List, Optional, Tuple


# Global plot style
STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.facecolor": "#fafafa",
}


class SimulationPlotter:
    """Publication-quality plotting for Monte Carlo simulations."""

    def __init__(self):
        plt.rcParams.update(STYLE)

    @staticmethod
    def _apply_style():
        plt.rcParams.update(STYLE)

    def simulation_paths(
        self,
        paths: np.ndarray,
        title: str = "Monte Carlo Simulation Paths",
        xlabel: str = "Time Step",
        ylabel: str = "Value",
        max_paths: int = 100,
        percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ensemble of simulation paths with percentile bands."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(12, 7))

        n_sims = min(paths.shape[0], max_paths)
        x = np.arange(paths.shape[1])

        for i in range(n_sims):
            ax.plot(x, paths[i], alpha=0.05, color="steelblue", linewidth=0.5)

        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]
        labels = [f"{p}th percentile" for p in percentiles]
        for p, c, lbl in zip(percentiles, colors, labels):
            y = np.percentile(paths, p, axis=0)
            ls = "-" if p == 50 else "--"
            lw = 2.5 if p == 50 else 1.5
            ax.plot(x, y, color=c, linestyle=ls, linewidth=lw, label=lbl)

        ax.fill_between(x, np.percentile(paths, 5, axis=0),
                        np.percentile(paths, 95, axis=0), alpha=0.1, color="steelblue")
        ax.fill_between(x, np.percentile(paths, 25, axis=0),
                        np.percentile(paths, 75, axis=0), alpha=0.2, color="steelblue")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def distribution(
        self,
        data: np.ndarray,
        title: str = "Distribution",
        xlabel: str = "Value",
        bins: int = 50,
        show_kde: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot histogram with optional KDE overlay."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=bins, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", label="Histogram")

        if show_kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_grid, kde(x_grid), color="#d62728", linewidth=2, label="KDE")

        mean = np.mean(data)
        ax.axvline(mean, color="#2ca02c", linestyle="--", linewidth=2,
                   label=f"Mean = {mean:.4f}")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def ising_lattice(
        self,
        spins: np.ndarray,
        title: str = "Ising Model Configuration",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize 2D Ising model spin configuration."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(8, 8))

        cmap = mcolors.ListedColormap(["#1f77b4", "#d62728"])
        ax.imshow((spins + 1) / 2, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def phase_diagram(
        self,
        temperatures: np.ndarray,
        magnetization: np.ndarray,
        specific_heat: Optional[np.ndarray] = None,
        susceptibility: Optional[np.ndarray] = None,
        T_c: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Ising model phase diagram with observables vs temperature."""
        self._apply_style()
        n_plots = 1 + (specific_heat is not None) + (susceptibility is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        idx = 0
        axes[idx].plot(temperatures, magnetization, "o-", color="steelblue", markersize=4)
        axes[idx].set_ylabel("|m|")
        axes[idx].set_title("Phase Diagram", fontweight="bold")
        if T_c:
            axes[idx].axvline(T_c, color="red", linestyle="--", alpha=0.7, label=f"T_c = {T_c:.3f}")
            axes[idx].legend()
        idx += 1

        if specific_heat is not None:
            axes[idx].plot(temperatures, specific_heat, "s-", color="#d62728", markersize=4)
            axes[idx].set_ylabel("C_v")
            if T_c:
                axes[idx].axvline(T_c, color="red", linestyle="--", alpha=0.7)
            idx += 1

        if susceptibility is not None:
            axes[idx].plot(temperatures, susceptibility, "^-", color="#2ca02c", markersize=4)
            axes[idx].set_ylabel("χ")
            if T_c:
                axes[idx].axvline(T_c, color="red", linestyle="--", alpha=0.7)

        axes[-1].set_xlabel("Temperature (J/k_B)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def convergence_plot(
        self,
        data: np.ndarray,
        title: str = "Convergence",
        true_value: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot running mean convergence."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(10, 6))

        running_mean = np.cumsum(data) / np.arange(1, len(data) + 1)
        ax.plot(running_mean, color="steelblue", linewidth=1.5, label="Running Mean")

        if true_value is not None:
            ax.axhline(true_value, color="#d62728", linestyle="--", linewidth=2,
                       label=f"True Value = {true_value:.6f}")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Estimate")
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def dose_profile(
        self,
        depths: np.ndarray,
        dose: np.ndarray,
        title: str = "Depth-Dose Profile",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot radiation depth-dose profile."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(depths, dose / max(dose.max(), 1e-10) * 100, "o-",
                color="steelblue", markersize=3, linewidth=1.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Depth (cm)")
        ax.set_ylabel("Relative Dose (%)")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
