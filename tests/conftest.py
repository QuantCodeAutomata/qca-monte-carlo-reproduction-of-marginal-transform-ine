"""
Shared pytest fixtures for the LGC Monte Carlo test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nts import symmetric_nts, skewed_nts, NIGDistribution
from src.copulas import make_bicop, simulate_copula_with_margins
from src.lgc import make_eval_grid, global_bandwidth, lgc_surface
from src.transforms import gaussianize


# ---------------------------------------------------------------------------
# Random number generators
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def rng2() -> np.random.Generator:
    """Second seeded random generator (different seed)."""
    return np.random.default_rng(99)


# ---------------------------------------------------------------------------
# Marginal distributions
# ---------------------------------------------------------------------------

@pytest.fixture
def sym_marginal() -> NIGDistribution:
    """Symmetric unit-variance NIG marginal."""
    return symmetric_nts()


@pytest.fixture
def skew_marginal() -> NIGDistribution:
    """Skewed NIG marginal with target skewness ≈ -0.84."""
    return skewed_nts(target_skew=-0.84)


# ---------------------------------------------------------------------------
# Small bivariate samples (fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gaussian_sample(rng: np.random.Generator, sym_marginal: NIGDistribution) -> np.ndarray:
    """n=200 bivariate sample from Gaussian copula (tau=0.5) with NIG margins."""
    _, X = simulate_copula_with_margins("gaussian", 0.5, 200, sym_marginal, rng)
    return X


@pytest.fixture
def small_clayton_sample(rng: np.random.Generator, sym_marginal: NIGDistribution) -> np.ndarray:
    """n=200 bivariate sample from Clayton copula (tau=0.5) with NIG margins."""
    _, X = simulate_copula_with_margins("clayton", 0.5, 200, sym_marginal, rng)
    return X


@pytest.fixture
def small_independent_sample(rng: np.random.Generator) -> np.ndarray:
    """n=300 bivariate standard normal sample with zero correlation."""
    return rng.standard_normal((300, 2))


@pytest.fixture
def small_positive_sample(rng: np.random.Generator) -> np.ndarray:
    """n=300 bivariate normal sample with strong positive correlation."""
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    L = np.linalg.cholesky(cov)
    return rng.standard_normal((300, 2)) @ L.T


# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_grid() -> np.ndarray:
    """Standard 11×11 + diagonal LGC evaluation grid."""
    return make_eval_grid(n_grid=11)


@pytest.fixture
def small_grid() -> np.ndarray:
    """Small 5×5 grid for speed in expensive tests."""
    return make_eval_grid(n_grid=5)


# ---------------------------------------------------------------------------
# Gaussianized samples
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussianized_gaussian(small_gaussian_sample: np.ndarray) -> np.ndarray:
    """Empirically Gaussianized Gaussian-copula sample."""
    return gaussianize(small_gaussian_sample, transform_type="empirical")


@pytest.fixture
def gaussianized_independent(small_independent_sample: np.ndarray) -> np.ndarray:
    """Empirically Gaussianized independent bivariate normal sample."""
    return gaussianize(small_independent_sample, transform_type="empirical")
