"""
Advanced Random Number Generation
===================================

Provides multiple RNG backends and variance reduction techniques.
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.stats.qmc import Halton, Sobol
    HAS_SCIPY_QMC = True
except ImportError:
    HAS_SCIPY_QMC = False

logger = logging.getLogger(__name__)


class RNGBackend(Enum):
    MERSENNE_TWISTER = "mt19937"
    PCG64 = "pcg64"
    PHILOX = "philox"


class RandomGenerator:
    """Advanced random number generator with multiple backends and variance reduction.

    Args:
        seed: Random seed for reproducibility.
        backend: RNG algorithm to use.
    """

    def __init__(self, seed: Optional[int] = None, backend: RNGBackend = RNGBackend.PCG64):
        self.seed = seed
        self.backend = backend
        self._generator = self._create_generator(seed, backend)

    @staticmethod
    def _create_generator(seed, backend):
        bg_map = {
            RNGBackend.MERSENNE_TWISTER: np.random.MT19937,
            RNGBackend.PCG64: np.random.PCG64,
            RNGBackend.PHILOX: np.random.Philox,
        }
        return np.random.Generator(bg_map[backend](seed))

    @property
    def generator(self) -> np.random.Generator:
        return self._generator

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._generator.uniform(low, high, size=size)

    def normal(self, mean=0.0, std=1.0, size=None):
        return self._generator.normal(mean, std, size=size)

    def exponential(self, scale=1.0, size=None):
        return self._generator.exponential(scale, size=size)

    def poisson(self, lam=1.0, size=None):
        return self._generator.poisson(lam, size=size)

    def integers(self, low, high, size=None):
        return self._generator.integers(low, high, size=size)

    def antithetic(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate antithetic variate pairs for variance reduction."""
        u = self._generator.uniform(0, 1, size=size)
        return u, 1.0 - u

    def stratified(self, n_strata: int, samples_per_stratum: int = 1) -> np.ndarray:
        """Generate stratified random samples in [0, 1)."""
        samples = np.zeros(n_strata * samples_per_stratum)
        width = 1.0 / n_strata
        idx = 0
        for i in range(n_strata):
            lo, hi = i * width, (i + 1) * width
            for _ in range(samples_per_stratum):
                samples[idx] = self._generator.uniform(lo, hi)
                idx += 1
        return samples

    def halton(self, n_samples: int, n_dims: int = 1) -> np.ndarray:
        """Generate Halton quasi-random sequence."""
        if HAS_SCIPY_QMC:
            return Halton(d=n_dims, scramble=True, seed=self.seed).random(n=n_samples)
        return self._van_der_corput(n_samples, n_dims)

    def sobol(self, n_samples: int, n_dims: int = 1) -> np.ndarray:
        """Generate Sobol quasi-random sequence."""
        if not HAS_SCIPY_QMC:
            raise ImportError("SciPy >= 1.7.0 required for Sobol sequences")
        m = int(np.ceil(np.log2(max(n_samples, 1))))
        return Sobol(d=n_dims, scramble=True, seed=self.seed).random_base2(m=m)[:n_samples]

    def _van_der_corput(self, n_samples, n_dims):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        result = np.zeros((n_samples, n_dims))
        for d in range(min(n_dims, len(primes))):
            base = primes[d]
            for i in range(n_samples):
                n, vdc, denom = i + 1, 0.0, 1.0
                while n > 0:
                    denom *= base
                    n, rem = divmod(n, base)
                    vdc += rem / denom
                result[i, d] = vdc
        return result

    def reset(self, seed=None):
        self.seed = seed if seed is not None else self.seed
        self._generator = self._create_generator(self.seed, self.backend)
