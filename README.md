# 🎲 Monte Carlo Simulation Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A comprehensive, industry-grade Monte Carlo simulation framework combining techniques from **particle physics**, **financial engineering**, **statistical mechanics**, and **nuclear/astroparticle physics**.

---

## 🌟 Features

| Module | Description | Inspired By |
|--------|-------------|-------------|
| **Core Engine** | Abstract MC framework, advanced RNG, convergence diagnostics | — |
| **Particle Transport** | Photon transport, Compton/photoelectric, dose calculation | [GGEMS](https://github.com/GGEMS/ggems), [OpenTOPAS](https://github.com/OpenTOPAS/OpenTOPAS) |
| **Financial** | Portfolio GBM, options pricing (Black-Scholes MC), VaR/CVaR | [pandas-montecarlo](https://github.com/ranaroussi/pandas-montecarlo) |
| **Ising Model** | 2D Ising, Metropolis-Hastings, Wolff cluster, phase transitions | [NVIDIA/ising-gpu](https://github.com/NVIDIA/ising-gpu) |
| **Nuclear** | Neutrino interactions, cross-sections, detector simulation | [NuRadioMC](https://github.com/nu-radio/NuRadioMC) |
| **Visualization** | Publication-quality plots, lattice visualization, animations | — |

---

## 🏗️ Architecture

```
montecarlo/
├── core/                  # Base engine, RNG, statistics, convergence
├── particle_transport/    # Photon transport, materials, geometry, dosimetry
├── financial/             # Portfolio, options, risk, time-series MC
├── ising/                 # 2D Ising model, Metropolis & Wolff algorithms
├── nuclear/               # Neutrino simulation, cross-sections, detectors
└── visualization/         # Plots and animations
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/naman-fr/Monte-Carlo-Simulation-.git
cd Monte-Carlo-Simulation-
pip install -e ".[dev]"
```

### Estimate Pi (Core Engine Demo)

```python
from montecarlo.core.engine import MonteCarloSimulation
import numpy as np

class PiEstimator(MonteCarloSimulation):
    def _simulate_single(self, rng):
        x, y = rng.random(), rng.random()
        return 1.0 if x**2 + y**2 <= 1.0 else 0.0

    def _aggregate_results(self, samples):
        return 4.0 * samples

sim = PiEstimator(n_simulations=100000, seed=42)
result = sim.run()
print(f"Pi ≈ {result.mean:.5f}")  # Pi ≈ 3.14159
```

### Portfolio Simulation

```python
from montecarlo.financial import PortfolioSimulator

sim = PortfolioSimulator(
    initial_value=100000,
    expected_return=0.10,
    volatility=0.25,
    time_horizon=5.0,
    n_simulations=10000,
    seed=42,
)
result = sim.run()
print(result.summary())
```

### Options Pricing (MC vs Black-Scholes)

```python
from montecarlo.financial import OptionsPricer

pricer = OptionsPricer(
    spot=100, strike=105,
    risk_free_rate=0.05, volatility=0.20,
    maturity=1.0, option_type="call",
    n_simulations=200000, seed=42,
)
mc_price = pricer.run().mean
bs_price = pricer.black_scholes_analytical()
print(f"MC: ${mc_price:.4f} | BS: ${bs_price:.4f}")
```

### 2D Ising Model

```python
from montecarlo.ising import IsingModel, MetropolisSampler
import numpy as np

model = IsingModel(L=32, temperature=2.269, seed=42)  # At T_c
sampler = MetropolisSampler(model, n_sweeps=10000, n_thermalize=5000)
history = sampler.run()
print(f"<|m|> = {np.mean(history['abs_magnetization']):.4f}")
```

### Neutrino Simulation

```python
from montecarlo.nuclear import NeutrinoSimulator

sim = NeutrinoSimulator(
    energy_range=(1e15, 1e19),
    spectral_index=2.0,
    n_simulations=5000,
    seed=42,
)
result = sim.run()
print(result.summary())
```

---

## 📊 Modules in Detail

### Core Engine
- **`MonteCarloSimulation`**: Abstract base class using Template Method pattern
- **`RandomGenerator`**: Multiple RNG backends (MT19937, PCG64, Philox) + variance reduction (antithetic, stratified, Halton, Sobol)
- **`StatisticalAnalyzer`**: Descriptive stats, confidence intervals, KDE, bootstrap, normality tests
- **`ConvergenceDiagnostics`**: Gelman-Rubin R-hat, effective sample size, Geweke test, batch means

### Particle Transport
- Photon transport through voxelized geometry with Compton scattering and photoelectric absorption
- Material database with real mass attenuation coefficients (water, bone, tissue, lead, aluminum)
- 3D voxelized phantom with ray-box intersection and CT phantom generation
- Dose deposition, dose-volume histograms, and depth-dose profiles

### Financial
- Geometric Brownian Motion portfolio simulation with multi-asset Cholesky correlation
- European and Asian option pricing with analytical Black-Scholes comparison
- Greeks estimation via finite differences (Delta, Gamma, Vega, Theta, Rho)
- Risk metrics: VaR, CVaR, Sharpe/Sortino ratios, max drawdown, probability of ruin

### Ising Model
- 2D square-lattice Ising model with periodic boundary conditions
- **Metropolis-Hastings** with vectorized checkerboard decomposition (inspired by NVIDIA GPU approach)
- **Wolff cluster** algorithm for efficient sampling near T_c
- Temperature sweep with specific heat and susceptibility computation
- Exact Onsager critical temperature: T_c = 2J/ln(1+√2) ≈ 2.269

### Nuclear/Neutrino
- Neutrino-nucleon cross-sections (Connolly, Gandhi parametrizations)
- Power-law energy spectrum sampling with configurable spectral index
- Detector response simulation with trigger efficiency and resolution smearing
- Event rate estimation for cosmogenic neutrino flux models

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**Validation highlights:**
- Pi estimation within 0.05 of true value (50,000 samples)
- Black-Scholes MC within 2% of analytical solution
- Ising critical temperature matches Onsager's exact T_c = 2.269
- Beer-Lambert photon attenuation law verification
- Put-call parity verification for option pricing

---

## 📁 Examples

Run the example scripts:

```bash
python examples/run_particle_sim.py
python examples/run_portfolio_sim.py
python examples/run_ising_sim.py
python examples/run_nuclear_sim.py
```

---

## ⚡ Benchmarks

```bash
python benchmarks/benchmark_core.py
```

---

## 📚 References

1. **GGEMS**: Allison et al. — GPU GEant4-based Monte Carlo Simulation
2. **OpenTOPAS**: Faddegon et al. — Tool for Particle Simulation
3. **pandas-montecarlo**: Aroussi — MC simulation on Pandas Series
4. **NVIDIA/ising-gpu**: Block et al. — Multi-GPU Ising model
5. **NuRadioMC**: Glaser et al. — Radio neutrino Monte Carlo
6. **Onsager** (1944): Crystal statistics. Exact 2D Ising solution
7. **Black & Scholes** (1973): Options pricing model
8. **Connolly et al.** (2011): UHE neutrino cross-sections

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
