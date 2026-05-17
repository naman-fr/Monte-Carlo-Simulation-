# Mathematical Foundations

## 1. Monte Carlo Integration

The fundamental principle: estimate an expectation by averaging random samples.

```
E[f(X)] ≈ (1/N) Σ f(xᵢ),  xᵢ ~ p(x)
```

**Error convergence**: O(1/√N) regardless of dimensionality — the key advantage over deterministic quadrature in high dimensions.

---

## 2. Geometric Brownian Motion (Financial)

Stock price evolution under the risk-neutral measure:

```
dS = μS dt + σS dW
```

**Exact solution**: S(t) = S(0) exp[(μ - σ²/2)t + σW(t)]

**Discretization**: S(t+Δt) = S(t) exp[(r - σ²/2)Δt + σ√Δt · Z],  Z ~ N(0,1)

### Black-Scholes Formula
```
C = S·N(d₁) - K·e^{-rT}·N(d₂)

d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

---

## 3. Metropolis-Hastings Algorithm

For sampling from distribution π(x):

1. Propose x' from q(x'|x)
2. Accept with probability: α = min(1, π(x')q(x|x') / π(x)q(x'|x))
3. If rejected, keep current state

For the Ising model with single-spin flip:
```
α = min(1, exp(-βΔE))
ΔE = 2J·sᵢ·Σⱼ sⱼ
```

### Checkerboard Decomposition
Spins on the same sublattice are conditionally independent, enabling parallel updates.

---

## 4. Ising Model

**Hamiltonian**:
```
H = -J Σ_{<ij>} sᵢsⱼ - h Σᵢ sᵢ
```

**Onsager's exact critical temperature** (2D, h=0):
```
T_c = 2J / ln(1 + √2) ≈ 2.269 J/k_B
```

**Observables**:
- Energy per spin: ε = ⟨H⟩/N
- Magnetization: m = ⟨Σsᵢ⟩/N
- Specific heat: C_v = (⟨E²⟩ - ⟨E⟩²) / (k_BT²)
- Susceptibility: χ = N(⟨m²⟩ - ⟨|m|⟩²) / (k_BT)

### Wolff Cluster Algorithm
Build clusters by adding aligned neighbors with probability:
```
p_add = 1 - exp(-2βJ)
```
Reduces critical slowing down from z ≈ 2 to z ≈ 0.25.

---

## 5. Photon Transport Physics

### Beer-Lambert Law
```
I(x) = I₀ exp(-μx)
```

### Interaction Cross-Sections
- **Photoelectric absorption**: σ_pe ∝ Z^5 / E^{7/2}
- **Compton scattering**: Klein-Nishina formula
- **Pair production**: σ_pp ∝ Z² ln(E/1.022 MeV)  (threshold: 1.022 MeV)

### Dose Deposition
```
D = ΔE / m  [Gray = J/kg]
```

---

## 6. Neutrino Cross-Sections

### Deep Inelastic Scattering
```
dσ/dxdy = (G_F² M_N E_ν) / π · [xq(x,Q²) + x·q̄(x,Q²)(1-y)²]
```

### Connolly et al. (2011) Parametrization
For ultra-high energies (E > 10⁴ GeV):
```
log₁₀(σ) = c₀ + c₁·log₁₀(1 + exp(c₂·(log₁₀E + c₃)))
```

---

## 7. Variance Reduction Techniques

### Antithetic Variates
For each U ~ Uniform(0,1), use both U and 1-U:
```
θ̂ = [f(U) + f(1-U)] / 2
```
Reduces variance when f is monotonic: Var(θ̂) ≤ Var(f(U))/2

### Stratified Sampling
Divide [0,1) into K strata, sample from each:
```
Var(θ̂_strat) ≤ Var(θ̂_MC) / K
```

### Quasi-Random Sequences (Halton, Sobol)
Low-discrepancy sequences fill space more uniformly:
- Error: O(log(N)^d / N) vs O(1/√N) for pseudo-random
- Especially effective in moderate dimensions (d < 20)

---

## 8. Convergence Diagnostics

### Gelman-Rubin R̂
```
R̂ = √(V̂/W)

V̂ = ((n-1)/n)W + (1/n)B
B = (n/(m-1)) Σ(θ̄ⱼ - θ̄)²
W = (1/m) Σ sⱼ²
```
Convergence criterion: R̂ < 1.05

### Effective Sample Size
```
ESS = N / (1 + 2Σ ρ(k))
```
where ρ(k) is the autocorrelation at lag k.
