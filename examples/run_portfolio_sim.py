"""
Example: Portfolio Monte Carlo Simulation
============================================

Demonstrates portfolio simulation with risk analysis and options pricing.
"""

import numpy as np
from montecarlo.financial import PortfolioSimulator, OptionsPricer, RiskAnalyzer


def main():
    print("=" * 60)
    print("Financial Monte Carlo Simulation Example")
    print("=" * 60)

    # Portfolio simulation
    print("\n--- Portfolio Simulation ---")
    sim = PortfolioSimulator(
        initial_value=100000,
        expected_return=0.10,
        volatility=0.25,
        time_horizon=5.0,
        n_steps=252 * 5,
        n_simulations=10000,
        seed=42,
    )

    result = sim.run(show_progress=True)
    print(result.summary())

    wealth = sim.terminal_wealth_distribution()
    print("\nTerminal Wealth Distribution:")
    for k, v in wealth.items():
        print(f"  {k}: {v:.4f}")

    # Risk Analysis
    print("\n--- Risk Analysis ---")
    risk = RiskAnalyzer(result.samples)
    report = risk.full_report()
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")

    # Options Pricing
    print("\n--- European Call Option Pricing ---")
    pricer = OptionsPricer(
        spot=100, strike=105, risk_free_rate=0.05,
        volatility=0.20, maturity=1.0, option_type="call",
        n_simulations=100000, seed=42,
    )
    mc_result = pricer.run(show_progress=True)
    bs_price = pricer.black_scholes_analytical()

    print(f"  MC Price:   ${mc_result.mean:.4f}")
    print(f"  BS Price:   ${bs_price:.4f}")
    print(f"  Error:      {abs(mc_result.mean - bs_price) / bs_price * 100:.2f}%")

    # Asian option
    print("\n--- Asian Option Pricing ---")
    asian_price = pricer.price_asian()
    print(f"  Asian Call: ${asian_price:.4f}")


if __name__ == "__main__":
    main()
