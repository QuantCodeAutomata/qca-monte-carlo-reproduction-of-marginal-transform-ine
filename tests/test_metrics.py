"""
Tests for src/metrics.py — pointwise diagnostics, ISE aggregation,
and scarcity regression.

Coverage:
  - pointwise_diagnostics: shape, correctness of bias/sd/rmse/fisher_sd
  - aggregate_ise: mean and sd computation, NaN handling
  - scarcity_regression: slope sign, R², regression with known inputs
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import pointwise_diagnostics, aggregate_ise, scarcity_regression


# ---------------------------------------------------------------------------
# pointwise_diagnostics
# ---------------------------------------------------------------------------

class TestPointwiseDiagnostics:
    """Tests for the per-grid-point bias/SD/RMSE/Fisher-SD computation."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(21)
        self.R = 30
        self.M = 40
        self.rho_gt = rng.uniform(-0.8, 0.8, size=self.M)
        self.rho_hat_reps = self.rho_gt[np.newaxis, :] + rng.standard_normal(
            (self.R, self.M)) * 0.1
        self.eff_n_reps = np.abs(rng.standard_normal((self.R, self.M))) * 50.0 + 10.0

    def test_output_is_dataframe(self) -> None:
        from pandas import DataFrame
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        assert isinstance(df, DataFrame)

    def test_output_length(self) -> None:
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        assert len(df) == self.M

    def test_required_columns(self) -> None:
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        for col in ("bias", "sd", "rmse", "fisher_sd", "mean_eff_n"):
            assert col in df.columns, f"Missing column: {col}"

    def test_bias_formula(self) -> None:
        """bias = mean(rho_hat) - rho_gt."""
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        expected_bias = np.nanmean(self.rho_hat_reps, axis=0) - self.rho_gt
        np.testing.assert_allclose(df["bias"].values, expected_bias, atol=1e-10)

    def test_sd_formula(self) -> None:
        """sd = std(rho_hat, ddof=0) across replications."""
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        expected_sd = np.nanstd(self.rho_hat_reps, axis=0, ddof=0)
        np.testing.assert_allclose(df["sd"].values, expected_sd, atol=1e-10)

    def test_rmse_formula(self) -> None:
        """rmse = sqrt(bias² + sd²)."""
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        expected_rmse = np.sqrt(df["bias"].values**2 + df["sd"].values**2)
        np.testing.assert_allclose(df["rmse"].values, expected_rmse, atol=1e-10)

    def test_rmse_non_negative(self) -> None:
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        assert np.all(df["rmse"].values >= 0.0)

    def test_sd_non_negative(self) -> None:
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        assert np.all(df["sd"].values >= 0.0)

    def test_fisher_sd_formula(self) -> None:
        """fisher_sd = sd / (1 - rho_gt²)."""
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        denom = np.maximum(1.0 - self.rho_gt**2, 1e-6)
        expected_fisher = np.nanstd(self.rho_hat_reps, axis=0, ddof=0) / denom
        np.testing.assert_allclose(df["fisher_sd"].values, expected_fisher, atol=1e-8)

    def test_mean_eff_n_formula(self) -> None:
        df = pointwise_diagnostics(self.rho_hat_reps, self.rho_gt, self.eff_n_reps)
        expected_mean_eff_n = np.nanmean(self.eff_n_reps, axis=0)
        np.testing.assert_allclose(df["mean_eff_n"].values, expected_mean_eff_n, atol=1e-10)

    def test_nan_in_rho_hat(self) -> None:
        """NaN in rho_hat_reps is handled gracefully."""
        rho_hat = self.rho_hat_reps.copy()
        rho_hat[0, 5] = np.nan
        df = pointwise_diagnostics(rho_hat, self.rho_gt, self.eff_n_reps)
        assert np.all(np.isfinite(df["bias"].values))

    def test_zero_bias_when_unbiased(self) -> None:
        """If rho_hat = rho_gt everywhere every rep, bias = 0."""
        rho_hat_unbiased = np.tile(self.rho_gt, (self.R, 1))
        df = pointwise_diagnostics(rho_hat_unbiased, self.rho_gt, self.eff_n_reps)
        np.testing.assert_allclose(df["bias"].values, 0.0, atol=1e-12)

    def test_zero_sd_when_constant(self) -> None:
        """If rho_hat is identical across reps, sd = 0."""
        rho_hat_const = np.tile(self.rho_gt + 0.1, (self.R, 1))
        df = pointwise_diagnostics(rho_hat_const, self.rho_gt, self.eff_n_reps)
        np.testing.assert_allclose(df["sd"].values, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# aggregate_ise
# ---------------------------------------------------------------------------

class TestAggregateISE:
    """Tests for replication-level ISE aggregation."""

    def test_mean_ise_formula(self) -> None:
        ise = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = aggregate_ise(ise)
        assert result["mean_ise"] == pytest.approx(0.03, abs=1e-10)

    def test_sd_ise_formula(self) -> None:
        ise = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = aggregate_ise(ise)
        expected_sd = np.std(ise, ddof=1)
        assert result["sd_ise"] == pytest.approx(expected_sd, rel=1e-8)

    def test_nan_excluded(self) -> None:
        """NaN values are excluded from aggregation."""
        ise = np.array([0.01, np.nan, 0.03])
        result = aggregate_ise(ise)
        assert np.isfinite(result["mean_ise"])
        assert result["mean_ise"] == pytest.approx(0.02, abs=1e-10)

    def test_all_nan_returns_nan(self) -> None:
        ise = np.array([np.nan, np.nan])
        result = aggregate_ise(ise)
        assert np.isnan(result["mean_ise"])

    def test_single_value_sd_nan(self) -> None:
        """Single-value ISE gives NaN for sd (ddof=1)."""
        result = aggregate_ise(np.array([0.05]))
        assert np.isnan(result["sd_ise"])

    def test_positive_ise(self) -> None:
        """mean_ise must be positive for positive ISE values."""
        ise = np.array([0.1, 0.2, 0.15])
        result = aggregate_ise(ise)
        assert result["mean_ise"] > 0.0


# ---------------------------------------------------------------------------
# scarcity_regression
# ---------------------------------------------------------------------------

class TestScarcityRegression:
    """Tests for the log-log scarcity regression."""

    def _make_regression_data(
        self, slope: float = -0.5, n_pts: int = 100, seed: int = 0
    ):
        """Create synthetic data satisfying log[sd/(1-rho^2)] = a + slope*log(eff_n)."""
        rng = np.random.default_rng(seed)
        eff_n = np.exp(rng.uniform(np.log(5), np.log(500), n_pts))
        rho_gt = rng.uniform(-0.5, 0.5, n_pts)
        # fisher_sd = exp(intercept) * eff_n^slope (+ noise)
        log_fisher_sd = -1.0 + slope * np.log(eff_n) + rng.standard_normal(n_pts) * 0.05
        fisher_sd = np.exp(log_fisher_sd)
        denom = np.maximum(1.0 - rho_gt**2, 1e-6)
        sd_vals = fisher_sd * denom
        return sd_vals, rho_gt, eff_n

    def test_returns_required_keys(self) -> None:
        sd, rho, eff = self._make_regression_data()
        result = scarcity_regression(sd, rho, eff)
        for key in ("slope", "intercept", "r_squared", "n_obs"):
            assert key in result, f"Missing key: {key}"

    def test_slope_negative(self) -> None:
        """Larger eff_n → smaller dispersion → negative slope."""
        sd, rho, eff = self._make_regression_data(slope=-0.5)
        result = scarcity_regression(sd, rho, eff)
        assert result["slope"] < 0.0

    def test_r_squared_in_unit_interval(self) -> None:
        sd, rho, eff = self._make_regression_data()
        result = scarcity_regression(sd, rho, eff)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_recovers_known_slope(self) -> None:
        """With low noise, recovered slope should be near -0.61."""
        rng = np.random.default_rng(99)
        n_pts = 500
        eff_n = np.exp(rng.uniform(np.log(5), np.log(500), n_pts))
        rho_gt = rng.uniform(-0.5, 0.5, n_pts)
        true_slope = -0.61
        log_fisher_sd = -0.5 + true_slope * np.log(eff_n) + rng.standard_normal(n_pts) * 0.02
        fisher_sd = np.exp(log_fisher_sd)
        denom = np.maximum(1.0 - rho_gt**2, 1e-6)
        sd_vals = fisher_sd * denom

        result = scarcity_regression(sd_vals, rho_gt, eff_n)
        assert result["slope"] == pytest.approx(true_slope, abs=0.05)

    def test_high_r_squared_for_clean_data(self) -> None:
        """Perfect log-linear data should give R² ≈ 1."""
        rng = np.random.default_rng(3)
        n_pts = 200
        eff_n = np.exp(rng.uniform(np.log(5), np.log(500), n_pts))
        rho_gt = np.zeros(n_pts)
        # Exact log-linear relationship (zero noise)
        log_fisher_sd = -1.0 + (-0.5) * np.log(eff_n)
        sd_vals = np.exp(log_fisher_sd)

        result = scarcity_regression(sd_vals, rho_gt, eff_n)
        assert result["r_squared"] > 0.99

    def test_too_few_points_returns_nan(self) -> None:
        """Fewer than 5 supported points returns NaN."""
        sd = np.array([0.1, 0.2])
        rho = np.array([0.0, 0.1])
        eff = np.array([10.0, 20.0])
        result = scarcity_regression(sd, rho, eff)
        assert np.isnan(result["slope"])

    def test_filters_low_eff_n(self) -> None:
        """Points with eff_n < min_eff_n are excluded."""
        sd, rho, eff = self._make_regression_data(n_pts=100)
        eff[: 50] = 1.0  # below threshold
        result = scarcity_regression(sd, rho, eff, min_eff_n=5.0)
        assert result["n_obs"] <= 50

    def test_n_obs_positive(self) -> None:
        sd, rho, eff = self._make_regression_data()
        result = scarcity_regression(sd, rho, eff)
        assert result["n_obs"] > 0
