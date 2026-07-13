"""
Tests for src/nts.py — Normal Inverse Gaussian (NIG) distribution
used as proxy for Normal Tempered Stable (NTS) marginals.

Coverage:
  - Moments of symmetric and skewed NIG (mean, variance, skewness)
  - CDF/PPF consistency and monotonicity
  - PDF integrates to ~1 and is non-negative
  - ML fitting recovers generating parameters
  - Factory functions and edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.nts import (
    NIGDistribution,
    symmetric_nts,
    skewed_nts,
    _nig_raw_skewness,
    _calibrate_skewed_params,
)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

def test_nig_raw_skewness_zero_for_symmetric() -> None:
    """b=0 gives zero skewness."""
    assert _nig_raw_skewness(a=1.0, b=0.0) == pytest.approx(0.0, abs=1e-12)


def test_nig_raw_skewness_negative_for_negative_b() -> None:
    """Negative b gives negative skewness."""
    skew = _nig_raw_skewness(a=2.0, b=-0.5)
    assert skew < 0.0


def test_calibrate_skewed_params_achieves_target() -> None:
    """Calibrated b should produce skewness within 0.01 of target."""
    target = -0.84
    b = _calibrate_skewed_params(target_skew=target, a=2.0)
    achieved = _nig_raw_skewness(a=2.0, b=b)
    assert achieved == pytest.approx(target, abs=0.01)


# ---------------------------------------------------------------------------
# NIGDistribution construction
# ---------------------------------------------------------------------------

def test_nig_symmetric_construction() -> None:
    """Symmetric NIG (b=0) constructs without error."""
    d = NIGDistribution(a=1.0, b=0.0)
    assert d.a == 1.0
    assert d.b == 0.0


def test_nig_invalid_a_raises() -> None:
    """Non-positive a raises AssertionError."""
    with pytest.raises(AssertionError):
        NIGDistribution(a=0.0, b=0.0)


def test_nig_invalid_b_raises() -> None:
    """|b| >= a raises AssertionError."""
    with pytest.raises(AssertionError):
        NIGDistribution(a=1.0, b=1.5)


# ---------------------------------------------------------------------------
# Moments of standardized NIG
# ---------------------------------------------------------------------------

class TestSymmetricNIGMoments:
    """Symmetric unit-variance NIG should have mean≈0, var≈1, skewness≈0."""

    n_samples: int = 100_000
    rng: np.random.Generator = np.random.default_rng(0)

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        self.d = symmetric_nts()
        self.samples = self.d.rvs(size=self.n_samples, rng=self.rng)

    def test_mean_near_zero(self) -> None:
        assert np.mean(self.samples) == pytest.approx(0.0, abs=0.02)

    def test_variance_near_one(self) -> None:
        assert np.var(self.samples) == pytest.approx(1.0, abs=0.05)

    def test_skewness_near_zero(self) -> None:
        skew = float(stats.skew(self.samples))
        assert abs(skew) < 0.1


class TestSkewedNIGMoments:
    """Skewed NIG with target_skew=-0.84 should have mean≈0, var≈1, skew≈-0.84."""

    n_samples: int = 100_000
    rng: np.random.Generator = np.random.default_rng(1)

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        self.d = skewed_nts(target_skew=-0.84)
        self.samples = self.d.rvs(size=self.n_samples, rng=self.rng)

    def test_mean_near_zero(self) -> None:
        assert np.mean(self.samples) == pytest.approx(0.0, abs=0.02)

    def test_variance_near_one(self) -> None:
        assert np.var(self.samples) == pytest.approx(1.0, abs=0.05)

    def test_skewness_near_target(self) -> None:
        skew = float(stats.skew(self.samples))
        assert skew == pytest.approx(-0.84, abs=0.15)


# ---------------------------------------------------------------------------
# CDF / PPF consistency
# ---------------------------------------------------------------------------

def test_cdf_ppf_roundtrip_symmetric() -> None:
    """CDF(PPF(q)) ≈ q for symmetric NIG."""
    d = symmetric_nts()
    q = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    x = d.ppf(q)
    q_recovered = d.cdf(x)
    np.testing.assert_allclose(q_recovered, q, atol=5e-4)


def test_cdf_ppf_roundtrip_skewed() -> None:
    """CDF(PPF(q)) ≈ q for skewed NIG."""
    d = skewed_nts()
    q = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    x = d.ppf(q)
    q_recovered = d.cdf(x)
    np.testing.assert_allclose(q_recovered, q, atol=5e-4)


def test_cdf_monotone_symmetric() -> None:
    """CDF is non-decreasing for symmetric NIG."""
    d = symmetric_nts()
    x = np.linspace(-5.0, 5.0, 200)
    cdf_vals = d.cdf(x)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-10), "CDF must be non-decreasing"


def test_cdf_bounds_symmetric() -> None:
    """CDF values lie in [0, 1]."""
    d = symmetric_nts()
    x = np.linspace(-10.0, 10.0, 500)
    cdf_vals = d.cdf(x)
    assert np.all(cdf_vals >= 0.0) and np.all(cdf_vals <= 1.0)


def test_cdf_boundary_values_symmetric() -> None:
    """CDF → 0 at far left, → 1 at far right."""
    d = symmetric_nts()
    assert d.cdf(np.array([-25.0])) < 0.001
    assert d.cdf(np.array([25.0])) > 0.999


# ---------------------------------------------------------------------------
# PDF properties
# ---------------------------------------------------------------------------

def test_pdf_non_negative_symmetric() -> None:
    """PDF is non-negative everywhere."""
    d = symmetric_nts()
    x = np.linspace(-8.0, 8.0, 400)
    assert np.all(d.pdf(x) >= 0.0)


def test_pdf_integrates_to_one_symmetric() -> None:
    """PDF integrates to ≈ 1 over a wide range."""
    d = symmetric_nts()
    x = np.linspace(-20.0, 20.0, 10_000)
    integral = np.trapz(d.pdf(x), x)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_logpdf_equals_log_pdf() -> None:
    """log(PDF(x)) equals logPDF(x) for finite values."""
    d = symmetric_nts()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(d.logpdf(x), np.log(d.pdf(x)), atol=1e-8)


# ---------------------------------------------------------------------------
# RVS shape
# ---------------------------------------------------------------------------

def test_rvs_int_size(sym_marginal: NIGDistribution) -> None:
    """rvs with int size returns 1-D array of that length."""
    samples = sym_marginal.rvs(size=50, rng=np.random.default_rng(7))
    assert samples.shape == (50,)


def test_rvs_tuple_size(sym_marginal: NIGDistribution) -> None:
    """rvs with tuple size returns correctly shaped array."""
    samples = sym_marginal.rvs(size=(10, 2), rng=np.random.default_rng(8))
    assert samples.shape == (10, 2)


def test_rvs_reproducible_with_rng(sym_marginal: NIGDistribution) -> None:
    """Same seed produces same samples."""
    s1 = sym_marginal.rvs(size=20, rng=np.random.default_rng(11))
    s2 = sym_marginal.rvs(size=20, rng=np.random.default_rng(11))
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# ML fitting
# ---------------------------------------------------------------------------

def test_fit_ml_recovers_symmetric_params() -> None:
    """ML fitting on large symmetric NIG sample recovers a≈1, b≈0."""
    rng = np.random.default_rng(123)
    d_true = symmetric_nts()
    data = d_true.rvs(size=3000, rng=rng)
    d_fit = d_true.fit_ml(data)
    # b should be small (symmetric)
    assert abs(d_fit.b) < 0.3
    # Fitted distribution should have near-zero mean and unit variance
    assert d_fit.raw_mean / d_fit.raw_std == pytest.approx(0.0, abs=0.3)


def test_fit_ml_returns_nig_distribution() -> None:
    """fit_ml returns a NIGDistribution instance."""
    d = symmetric_nts()
    data = d.rvs(size=500, rng=np.random.default_rng(77))
    d_fit = d.fit_ml(data)
    assert isinstance(d_fit, NIGDistribution)


def test_fit_ml_preserves_skew_sign() -> None:
    """ML fit on negatively skewed data returns b < 0."""
    d_true = skewed_nts(target_skew=-0.84)
    data = d_true.rvs(size=5000, rng=np.random.default_rng(55))
    d_fit = d_true.fit_ml(data)
    assert d_fit.b < 0.0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def test_symmetric_nts_returns_nig() -> None:
    assert isinstance(symmetric_nts(), NIGDistribution)


def test_skewed_nts_returns_nig() -> None:
    assert isinstance(skewed_nts(), NIGDistribution)


def test_symmetric_nts_b_is_zero() -> None:
    d = symmetric_nts()
    assert d.b == 0.0


def test_skewed_nts_b_negative_for_negative_skew() -> None:
    d = skewed_nts(target_skew=-0.84)
    assert d.b < 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cdf_scalar_input() -> None:
    """CDF accepts scalar-like input and returns a 0-dim or 1-dim array."""
    d = symmetric_nts()
    result = d.cdf(np.array(0.0))
    assert 0.0 < float(result) < 1.0


def test_ppf_extreme_quantiles() -> None:
    """PPF handles extreme quantiles without NaN."""
    d = symmetric_nts()
    result = d.ppf(np.array([0.001, 0.999]))
    assert np.all(np.isfinite(result))
    assert result[0] < result[1]


def test_pdf_at_zero_symmetric() -> None:
    """PDF at x=0 is the mode for symmetric NIG (positive value)."""
    d = symmetric_nts()
    assert d.pdf(np.array([0.0])) > 0.0
