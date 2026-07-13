"""
Tests for src/copulas.py — bivariate copula simulation via pyvinecopulib.

Coverage:
  - tau_to_param closed-form formulas for all four families
  - make_bicop constructs valid Bicop objects
  - simulate_copula_with_margins returns correct shapes and uniform marginals
  - ALL_FAMILIES constant contains the four required families
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
import pyvinecopulib as pv

from src.copulas import (
    tau_to_param,
    make_bicop,
    simulate_copula_with_margins,
    ALL_FAMILIES,
    CopulaFamily,
)
from src.nts import symmetric_nts


# ---------------------------------------------------------------------------
# tau_to_param formulas
# ---------------------------------------------------------------------------

class TestTauToParam:
    """Verify standard closed-form tau → parameter mappings."""

    def test_gaussian_at_half(self) -> None:
        """Gaussian: rho = sin(pi*tau/2)."""
        rho = tau_to_param("gaussian", 0.5)
        expected = np.sin(np.pi * 0.5 / 2.0)
        assert rho == pytest.approx(expected, rel=1e-10)

    def test_gaussian_at_zero(self) -> None:
        """Gaussian at tau=0 gives rho=0."""
        assert tau_to_param("gaussian", 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_clayton_at_half(self) -> None:
        """Clayton: theta = 2*tau/(1-tau)."""
        theta = tau_to_param("clayton", 0.5)
        expected = 2.0 * 0.5 / (1.0 - 0.5)
        assert theta == pytest.approx(expected, rel=1e-10)

    def test_clayton_at_two_thirds(self) -> None:
        """Clayton at tau=2/3 gives theta=4.0."""
        theta = tau_to_param("clayton", 2.0 / 3.0)
        assert theta == pytest.approx(4.0, rel=1e-6)

    def test_gumbel_at_half(self) -> None:
        """Gumbel: theta = 1/(1-tau)."""
        theta = tau_to_param("gumbel", 0.5)
        expected = 1.0 / (1.0 - 0.5)
        assert theta == pytest.approx(expected, rel=1e-10)

    def test_gumbel_at_zero_is_one(self) -> None:
        """Gumbel independence case: theta=1.0 at tau=0."""
        theta = tau_to_param("gumbel", 0.0)
        assert theta == pytest.approx(1.0, abs=1e-10)

    def test_t4_returns_rho_and_nu(self) -> None:
        """t-copula returns (rho, nu=4) tuple."""
        result = tau_to_param("t4", 0.5)
        assert isinstance(result, tuple)
        rho, nu = result
        assert nu == pytest.approx(4.0, abs=1e-10)
        expected_rho = np.sin(np.pi * 0.5 / 2.0)
        assert rho == pytest.approx(expected_rho, rel=1e-10)

    def test_unknown_family_raises(self) -> None:
        """Unknown family raises ValueError."""
        with pytest.raises((ValueError, KeyError, TypeError)):
            tau_to_param("unknown_family", 0.5)  # type: ignore[arg-type]

    def test_clayton_requires_positive_tau(self) -> None:
        """Clayton with tau=0 raises AssertionError."""
        with pytest.raises(AssertionError):
            tau_to_param("clayton", 0.0)


# ---------------------------------------------------------------------------
# ALL_FAMILIES constant
# ---------------------------------------------------------------------------

def test_all_families_contains_required() -> None:
    """ALL_FAMILIES must contain all four required families."""
    required = {"gaussian", "clayton", "t4", "gumbel"}
    assert required.issubset(set(ALL_FAMILIES))


def test_all_families_length() -> None:
    """ALL_FAMILIES has at least 4 entries."""
    assert len(ALL_FAMILIES) >= 4


# ---------------------------------------------------------------------------
# make_bicop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["gaussian", "clayton", "t4", "gumbel"])
def test_make_bicop_returns_bicop(family: CopulaFamily) -> None:
    """make_bicop returns a pyvinecopulib Bicop for each family at tau=0.5."""
    cop = make_bicop(family, 0.5)
    assert isinstance(cop, pv.Bicop)


@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_make_bicop_gaussian_various_tau(tau: float) -> None:
    """make_bicop works for Gaussian at multiple tau values."""
    cop = make_bicop("gaussian", tau)
    assert isinstance(cop, pv.Bicop)


def test_make_bicop_t4_has_4_dof() -> None:
    """t4 bicop has 4 degrees of freedom."""
    cop = make_bicop("t4", 0.5)
    # Parameter stored as (rho, nu); nu should be ~4
    params = cop.parameters
    assert params[1] == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# simulate_copula_with_margins
# ---------------------------------------------------------------------------

@pytest.fixture
def marginal():
    return symmetric_nts()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize("family", ["gaussian", "clayton", "t4", "gumbel"])
def test_simulate_returns_u_and_x(family: CopulaFamily, marginal, rng) -> None:
    """simulate_copula_with_margins returns (U, X) tuple of correct shape."""
    n = 200
    U, X = simulate_copula_with_margins(family, 0.5, n, marginal, rng)
    assert U.shape == (n, 2)
    assert X.shape == (n, 2)


def test_uniform_marginals_gaussian(marginal, rng) -> None:
    """Copula draws U are uniformly distributed in (0,1)."""
    n = 1000
    U, _ = simulate_copula_with_margins("gaussian", 0.5, n, marginal, rng)
    # Both margins should be approximately Uniform(0,1)
    _, p1 = stats.kstest(U[:, 0], "uniform")
    _, p2 = stats.kstest(U[:, 1], "uniform")
    assert p1 > 0.01, f"Margin 1 KS p-value too small: {p1}"
    assert p2 > 0.01, f"Margin 2 KS p-value too small: {p2}"


@pytest.mark.parametrize("family", ["gaussian", "clayton", "t4", "gumbel"])
def test_simulate_shape_various_n(family: CopulaFamily, marginal, rng) -> None:
    """simulate_copula_with_margins respects different sample sizes."""
    for n in [50, 100, 500]:
        U, X = simulate_copula_with_margins(family, 0.5, n, marginal, rng)
        assert U.shape[0] == n
        assert X.shape[0] == n


def test_simulate_positive_correlation_gaussian(marginal, rng) -> None:
    """Gaussian copula at tau=0.5 produces positive Pearson correlation in X."""
    n = 2000
    _, X = simulate_copula_with_margins("gaussian", 0.5, n, marginal, rng)
    r = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert r > 0.3, f"Expected positive correlation, got {r:.3f}"


def test_simulate_nts_marginals_heavy_tails(marginal, rng) -> None:
    """NIG-marginal X samples should have kurtosis > 3 (excess kurtosis > 0)."""
    n = 5000
    _, X = simulate_copula_with_margins("gaussian", 0.5, n, marginal, rng)
    kurt_0 = float(stats.kurtosis(X[:, 0]))  # excess kurtosis
    kurt_1 = float(stats.kurtosis(X[:, 1]))
    assert kurt_0 > 0.0, f"Expected heavy tails, excess kurtosis={kurt_0:.3f}"
    assert kurt_1 > 0.0


def test_simulate_reproducible_with_same_seed(marginal) -> None:
    """Same seed → same output."""
    U1, X1 = simulate_copula_with_margins("gaussian", 0.5, 100, marginal,
                                          np.random.default_rng(0))
    U2, X2 = simulate_copula_with_margins("gaussian", 0.5, 100, marginal,
                                          np.random.default_rng(0))
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(X1, X2)


def test_simulate_different_seeds_differ(marginal) -> None:
    """Different seeds → different output with very high probability."""
    _, X1 = simulate_copula_with_margins("gaussian", 0.5, 100, marginal,
                                         np.random.default_rng(0))
    _, X2 = simulate_copula_with_margins("gaussian", 0.5, 100, marginal,
                                         np.random.default_rng(1))
    assert not np.allclose(X1, X2), "Different seeds should produce different samples"


def test_simulate_finite_values(marginal, rng) -> None:
    """All simulated values should be finite."""
    n = 300
    U, X = simulate_copula_with_margins("gaussian", 0.5, n, marginal, rng)
    assert np.all(np.isfinite(U)), "U must be finite"
    assert np.all(np.isfinite(X)), "X must be finite"
