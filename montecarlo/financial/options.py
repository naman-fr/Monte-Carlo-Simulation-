"""
Options Pricing via Monte Carlo
==================================

Black-Scholes MC pricing for European, Asian, and barrier options.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict
from montecarlo.core.engine import MonteCarloSimulation


class OptionsPricer(MonteCarloSimulation):
    """Monte Carlo options pricing engine.

    Args:
        spot: Current asset price.
        strike: Strike price.
        risk_free_rate: Annualized risk-free rate.
        volatility: Annualized volatility.
        maturity: Time to expiration in years.
        option_type: 'call' or 'put'.
        n_steps: Time steps for path-dependent options.
        n_simulations: Number of MC paths.
        seed: Random seed.
    """

    def __init__(
        self,
        spot: float = 100.0,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        volatility: float = 0.20,
        maturity: float = 1.0,
        option_type: str = "call",
        n_steps: int = 252,
        n_simulations: int = 100000,
        seed: Optional[int] = None,
    ):
        super().__init__(n_simulations=n_simulations, seed=seed, name="OptionsPricer")
        self.spot = spot
        self.strike = strike
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = maturity
        self.option_type = option_type.lower()
        self.n_steps = n_steps
        self.dt = maturity / n_steps

    def _simulate_single(self, rng: np.random.Generator) -> float:
        """Simulate a single European option payoff (discounted)."""
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * rng.standard_normal()
        S_T = self.spot * np.exp(drift + diffusion)

        if self.option_type == "call":
            payoff = max(S_T - self.strike, 0.0)
        else:
            payoff = max(self.strike - S_T, 0.0)

        return payoff * np.exp(-self.r * self.T)

    def price_asian(self) -> float:
        """Price an Asian (arithmetic average) option."""
        payoffs = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            path = np.zeros(self.n_steps + 1)
            path[0] = self.spot
            for t in range(1, self.n_steps + 1):
                z = self._rng.standard_normal()
                drift = (self.r - 0.5 * self.sigma**2) * self.dt
                path[t] = path[t - 1] * np.exp(drift + self.sigma * np.sqrt(self.dt) * z)

            avg_price = np.mean(path[1:])
            if self.option_type == "call":
                payoffs[i] = max(avg_price - self.strike, 0.0)
            else:
                payoffs[i] = max(self.strike - avg_price, 0.0)

        return float(np.mean(payoffs) * np.exp(-self.r * self.T))

    def black_scholes_analytical(self) -> float:
        """Analytical Black-Scholes price for comparison."""
        from scipy.stats import norm
        d1 = (np.log(self.spot / self.strike) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            return float(self.spot * norm.cdf(d1) - self.strike * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            return float(self.strike * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.spot * norm.cdf(-d1))

    def greeks(self, bump: float = 0.01) -> Dict[str, float]:
        """Estimate option Greeks via finite differences.

        Returns:
            Dict with delta, gamma, vega, theta, rho.
        """
        base_price = self.run(show_progress=False).mean

        # Delta: dPrice/dSpot
        self.spot += bump
        self.reset(self.seed)
        up_price = self.run(show_progress=False).mean
        self.spot -= 2 * bump
        self.reset(self.seed)
        down_price = self.run(show_progress=False).mean
        self.spot += bump  # Restore

        delta = (up_price - down_price) / (2 * bump)
        gamma = (up_price - 2 * base_price + down_price) / (bump**2)

        # Vega: dPrice/dSigma
        self.sigma += bump
        self.reset(self.seed)
        vega_up = self.run(show_progress=False).mean
        self.sigma -= bump  # Restore
        vega = (vega_up - base_price) / bump

        # Theta: dPrice/dT
        self.T -= bump
        self.reset(self.seed)
        theta_val = self.run(show_progress=False).mean
        self.T += bump  # Restore
        theta = (theta_val - base_price) / (-bump)

        # Rho: dPrice/dR
        self.r += bump
        self.reset(self.seed)
        rho_up = self.run(show_progress=False).mean
        self.r -= bump  # Restore
        rho = (rho_up - base_price) / bump

        self.reset(self.seed)
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
