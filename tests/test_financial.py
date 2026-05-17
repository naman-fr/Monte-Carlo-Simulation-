"""Tests for financial Monte Carlo module."""
import pytest
import numpy as np
from montecarlo.financial.portfolio import PortfolioSimulator
from montecarlo.financial.options import OptionsPricer
from montecarlo.financial.risk import RiskAnalyzer
from montecarlo.financial.timeseries import TimeSeriesMC
import pandas as pd


class TestPortfolio:
    def test_gbm_simulation(self):
        sim = PortfolioSimulator(
            initial_value=10000, expected_return=0.08,
            volatility=0.2, n_simulations=1000, seed=42,
        )
        result = sim.run(show_progress=False)
        assert result.samples.shape == (1000, 253)
        assert result.samples[0, 0] == 10000.0

    def test_positive_prices(self):
        """GBM should produce strictly positive prices."""
        sim = PortfolioSimulator(n_simulations=500, seed=42)
        result = sim.run(show_progress=False)
        assert np.all(result.samples > 0)


class TestOptions:
    def test_call_put_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        call = OptionsPricer(spot=100, strike=100, risk_free_rate=0.05,
                             volatility=0.2, maturity=1.0, option_type="call",
                             n_simulations=200000, seed=42)
        put = OptionsPricer(spot=100, strike=100, risk_free_rate=0.05,
                            volatility=0.2, maturity=1.0, option_type="put",
                            n_simulations=200000, seed=42)

        c = call.run(show_progress=False).mean
        p = put.run(show_progress=False).mean
        parity = c - p - (100 - 100 * np.exp(-0.05))
        assert abs(parity) < 1.0  # Should be close to 0

    def test_bs_comparison(self):
        """MC price should be within 2% of analytical Black-Scholes."""
        pricer = OptionsPricer(
            spot=100, strike=100, risk_free_rate=0.05,
            volatility=0.2, maturity=1.0, option_type="call",
            n_simulations=200000, seed=42,
        )
        mc_price = pricer.run(show_progress=False).mean
        bs_price = pricer.black_scholes_analytical()
        assert abs(mc_price - bs_price) / bs_price < 0.02


class TestRisk:
    def test_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 10000)
        analyzer = RiskAnalyzer(returns)
        var_95 = analyzer.var(0.95)
        assert var_95 > 0  # VaR should be positive (loss)

    def test_cvar(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 10000)
        analyzer = RiskAnalyzer(returns)
        cvar = analyzer.cvar(0.95)
        var = analyzer.var(0.95)
        assert cvar >= var  # CVaR >= VaR always

    def test_max_drawdown(self):
        path = np.array([100, 110, 105, 95, 90, 100, 85])
        analyzer = RiskAnalyzer()
        mdd = analyzer.max_drawdown(path)
        # Max drawdown from 110 to 85 = -22.7%
        assert mdd < 0


class TestTimeSeries:
    def test_simulation(self):
        prices = pd.Series(np.random.default_rng(42).lognormal(0, 0.02, 252))
        ts = TimeSeriesMC(prices)
        sim_df = ts.simulate(n_simulations=100, seed=42)
        assert sim_df.shape[1] == 101  # original + 100 sims
