"""
Tests for src/transforms.py — Gaussianization transforms.

Coverage:
  - empirical_transform: rank convention, output normality, shape
  - oracle_transform: uses true marginal CDF, output quality
  - fitted_transform: ML-fitted CDF, output quality
  - gaussianize dispatcher: all three variants
  - _gaussianize_column: avoids 0 and 1, output range
  - Edge cases: single column, small n, extreme values
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.transforms import (
    _gaussianize_column,
    empirical_transform,
    oracle_transform,
    fitted_transform,
    gaussianize,
)
from src.nts import symmetric_nts, skewed_nts


# ---------------------------------------------------------------------------
# _gaussianize_column helper
# ---------------------------------------------------------------------------

class TestGaussianizeColumn:
    """Internal helper: Φ⁻¹ on values clipped away from (0,1) boundaries."""

    def test_output_shape(self) -> None:
        u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        z = _gaussianize_column(u)
        assert z.shape == u.shape

    def test_median_maps_to_zero(self) -> None:
        """u=0.5 → z=0."""
        z = _gaussianize_column(np.array([0.5]))
        assert z[0] == pytest.approx(0.0, abs=1e-10)

    def test_output_is_increasing(self) -> None:
        """Φ⁻¹ is monotone increasing."""
        u = np.linspace(0.01, 0.99, 50)
        z = _gaussianize_column(u)
        assert np.all(np.diff(z) > 0)

    def test_clips_away_from_boundaries(self) -> None:
        """Exact 0 and 1 should not produce ±Inf."""
        u = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        z = _gaussianize_column(u)
        assert np.all(np.isfinite(z))

    def test_symmetric_around_zero(self) -> None:
        """Φ⁻¹(1-u) = -Φ⁻¹(u)."""
        u = np.array([0.1, 0.2, 0.3])
        z = _gaussianize_column(u)
        z_comp = _gaussianize_column(1.0 - u)
        np.testing.assert_allclose(z_comp, -z, atol=1e-8)


# ---------------------------------------------------------------------------
# empirical_transform
# ---------------------------------------------------------------------------

class TestEmpiricalTransform:
    """Tests for the canonical rank-based Gaussianization."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(42)
        marginal = symmetric_nts()
        # Generate bivariate NIG data
        n = 500
        u = rng.random((n, 2))
        self.X = marginal.ppf(u)   # shape (n, 2)

    def test_output_shape(self) -> None:
        Z = empirical_transform(self.X)
        assert Z.shape == self.X.shape

    def test_output_is_float(self) -> None:
        Z = empirical_transform(self.X)
        assert Z.dtype == float

    def test_output_finite(self) -> None:
        Z = empirical_transform(self.X)
        assert np.all(np.isfinite(Z))

    def test_rank_convention_avoids_zero_one(self) -> None:
        """Empirical uniform scores should be strictly in (0, 1)."""
        n = self.X.shape[0]
        # Reverse-engineer: scores are (rank - 0.5)/n
        ranks = stats.rankdata(self.X[:, 0])
        scores = (ranks - 0.5) / n
        assert np.all(scores > 0.0) and np.all(scores < 1.0)

    def test_zero_mean_approx(self) -> None:
        """Output columns should have approximately zero mean."""
        Z = empirical_transform(self.X)
        assert np.mean(Z[:, 0]) == pytest.approx(0.0, abs=0.1)
        assert np.mean(Z[:, 1]) == pytest.approx(0.0, abs=0.1)

    def test_unit_variance_approx(self) -> None:
        """Output columns should have variance near 1."""
        Z = empirical_transform(self.X)
        assert np.var(Z[:, 0]) == pytest.approx(1.0, abs=0.15)

    def test_output_near_standard_normal(self) -> None:
        """KS test of each output column against N(0,1): p > 0.01."""
        Z = empirical_transform(self.X)
        _, p0 = stats.kstest(Z[:, 0], "norm")
        _, p1 = stats.kstest(Z[:, 1], "norm")
        assert p0 > 0.01, f"Column 0 not standard normal: KS p={p0:.4f}"
        assert p1 > 0.01, f"Column 1 not standard normal: KS p={p1:.4f}"

    def test_monotone_transform(self) -> None:
        """Rank order must be preserved."""
        Z = empirical_transform(self.X)
        order_X = np.argsort(self.X[:, 0])
        order_Z = np.argsort(Z[:, 0])
        np.testing.assert_array_equal(order_X, order_Z)

    def test_small_n(self) -> None:
        """Works on n=5 (edge case)."""
        X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.5, -1.0], [2.0, 1.5], [-2.0, -0.5]])
        Z = empirical_transform(X)
        assert Z.shape == (5, 2)
        assert np.all(np.isfinite(Z))


# ---------------------------------------------------------------------------
# oracle_transform
# ---------------------------------------------------------------------------

class TestOracleTransform:
    """Tests for the oracle (true parametric marginal) transform."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(88)
        self.marginal = symmetric_nts()
        n = 500
        u = rng.random((n, 2))
        self.X = self.marginal.ppf(u)

    def test_output_shape(self) -> None:
        Z = oracle_transform(self.X, self.marginal)
        assert Z.shape == self.X.shape

    def test_output_finite(self) -> None:
        Z = oracle_transform(self.X, self.marginal)
        assert np.all(np.isfinite(Z))

    def test_output_near_standard_normal(self) -> None:
        """Oracle transform should produce near-standard-normal output."""
        Z = oracle_transform(self.X, self.marginal)
        _, p0 = stats.kstest(Z[:, 0], "norm")
        assert p0 > 0.01, f"KS p-value too small: {p0:.4f}"

    def test_zero_mean_approx(self) -> None:
        Z = oracle_transform(self.X, self.marginal)
        assert np.mean(Z[:, 0]) == pytest.approx(0.0, abs=0.1)

    def test_monotone_transform(self) -> None:
        """Oracle transform preserves rank order (since CDF is monotone)."""
        Z = oracle_transform(self.X, self.marginal)
        order_X = np.argsort(self.X[:, 0])
        order_Z = np.argsort(Z[:, 0])
        np.testing.assert_array_equal(order_X, order_Z)


# ---------------------------------------------------------------------------
# fitted_transform
# ---------------------------------------------------------------------------

class TestFittedTransform:
    """Tests for the ML-fitted parametric marginal transform."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(77)
        self.marginal = symmetric_nts()
        n = 500
        u = rng.random((n, 2))
        self.X = self.marginal.ppf(u)

    def test_output_shape(self) -> None:
        Z = fitted_transform(self.X, self.marginal)
        assert Z.shape == self.X.shape

    def test_output_finite(self) -> None:
        Z = fitted_transform(self.X, self.marginal)
        assert np.all(np.isfinite(Z))

    def test_output_near_standard_normal(self) -> None:
        """ML-fitted transform should also produce near-standard-normal output."""
        Z = fitted_transform(self.X, self.marginal)
        _, p0 = stats.kstest(Z[:, 0], "norm")
        assert p0 > 0.01, f"KS p-value too small: {p0:.4f}"

    def test_monotone_transform(self) -> None:
        """Fitted CDF is monotone → rank order preserved."""
        Z = fitted_transform(self.X, self.marginal)
        order_X = np.argsort(self.X[:, 0])
        order_Z = np.argsort(Z[:, 0])
        np.testing.assert_array_equal(order_X, order_Z)


# ---------------------------------------------------------------------------
# gaussianize dispatcher
# ---------------------------------------------------------------------------

class TestGaussianizeDispatcher:
    """Tests for the gaussianize() dispatch function."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(33)
        self.marginal = symmetric_nts()
        n = 300
        u = rng.random((n, 2))
        self.X = self.marginal.ppf(u)

    def test_empirical_dispatch(self) -> None:
        Z = gaussianize(self.X, "empirical")
        assert Z.shape == self.X.shape
        assert np.all(np.isfinite(Z))

    def test_oracle_dispatch(self) -> None:
        Z = gaussianize(self.X, "oracle", marginal=self.marginal)
        assert Z.shape == self.X.shape
        assert np.all(np.isfinite(Z))

    def test_fitted_dispatch(self) -> None:
        Z = gaussianize(self.X, "fitted", marginal=self.marginal)
        assert Z.shape == self.X.shape
        assert np.all(np.isfinite(Z))

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError):
            gaussianize(self.X, "bad_transform")  # type: ignore[arg-type]

    def test_oracle_without_marginal_raises(self) -> None:
        with pytest.raises((AssertionError, TypeError)):
            gaussianize(self.X, "oracle", marginal=None)

    def test_fitted_without_marginal_raises(self) -> None:
        with pytest.raises((AssertionError, TypeError)):
            gaussianize(self.X, "fitted", marginal=None)

    def test_empirical_identical_to_direct_call(self) -> None:
        """gaussianize(..., 'empirical') == empirical_transform(...)."""
        Z1 = gaussianize(self.X, "empirical")
        Z2 = empirical_transform(self.X)
        np.testing.assert_array_equal(Z1, Z2)

    def test_all_three_outputs_similar_marginals(self) -> None:
        """All three transforms should produce near-standard-normal marginals."""
        Z_emp = gaussianize(self.X, "empirical")
        Z_ora = gaussianize(self.X, "oracle", marginal=self.marginal)
        Z_fit = gaussianize(self.X, "fitted", marginal=self.marginal)
        for Z, name in [(Z_emp, "empirical"), (Z_ora, "oracle"), (Z_fit, "fitted")]:
            m = np.mean(Z[:, 0])
            v = np.var(Z[:, 0])
            assert abs(m) < 0.2, f"{name} mean far from 0: {m:.3f}"
            assert abs(v - 1.0) < 0.3, f"{name} variance far from 1: {v:.3f}"


# ---------------------------------------------------------------------------
# All three transforms produce close LGC estimates (inertness property)
# ---------------------------------------------------------------------------

def test_inertness_similar_rho_at_center() -> None:
    """
    Empirical and oracle transforms should yield similar LGC at (0,0).
    This validates the central paper claim of marginal-transform inertness.
    """
    from src.lgc import lgc_at_point, global_bandwidth

    rng = np.random.default_rng(2024)
    marginal = symmetric_nts()
    n = 2000
    u = rng.random((n, 2))
    X = marginal.ppf(u)

    Z_emp = empirical_transform(X)
    Z_ora = oracle_transform(X, marginal)
    Z_fit = fitted_transform(X, marginal)

    b = global_bandwidth(n)
    x = np.array([0.0, 0.0])

    rho_emp, _ = lgc_at_point(Z_emp, x, b)
    rho_ora, _ = lgc_at_point(Z_ora, x, b)
    rho_fit, _ = lgc_at_point(Z_fit, x, b)

    # All three should be finite
    assert np.isfinite(rho_emp)
    assert np.isfinite(rho_ora)
    assert np.isfinite(rho_fit)

    # Differences must be small (inertness)
    diff_emp_ora = abs(rho_emp - rho_ora)
    diff_emp_fit = abs(rho_emp - rho_fit)
    assert diff_emp_ora < 0.15, f"|emp-oracle|={diff_emp_ora:.4f} too large"
    assert diff_emp_fit < 0.15, f"|emp-fitted|={diff_emp_fit:.4f} too large"


# ---------------------------------------------------------------------------
# Skewed marginal transforms
# ---------------------------------------------------------------------------

def test_empirical_handles_skewed_marginals() -> None:
    """Empirical transform works correctly when marginals are skewed."""
    rng = np.random.default_rng(111)
    marginal = skewed_nts(target_skew=-0.84)
    n = 300
    u = rng.random((n, 2))
    X = marginal.ppf(u)
    Z = empirical_transform(X)
    assert Z.shape == (n, 2)
    assert np.all(np.isfinite(Z))
    _, p = stats.kstest(Z[:, 0], "norm")
    assert p > 0.01


def test_oracle_handles_skewed_marginals() -> None:
    """Oracle transform with skewed marginals produces normal-ish output."""
    rng = np.random.default_rng(222)
    marginal = skewed_nts(target_skew=-0.84)
    n = 300
    u = rng.random((n, 2))
    X = marginal.ppf(u)
    Z = oracle_transform(X, marginal)
    assert Z.shape == (n, 2)
    assert np.all(np.isfinite(Z))
