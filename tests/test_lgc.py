"""
Tests for src/lgc.py — Local Gaussian Correlation estimator.

Coverage:
  - make_eval_grid geometry and shape
  - global_bandwidth formula (b0 = 1.75 * n^(-1/6))
  - kernel_weights: positivity, shape, decay with distance
  - lgc_at_point: bounds, behavior under independence and correlation
  - lgc_surface: shape, NaN handling
  - density_weighted_ise: zero for exact match, positive for different surfaces
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.lgc import (
    make_eval_grid,
    global_bandwidth,
    kernel_weights,
    lgc_at_point,
    lgc_surface,
    density_weighted_ise,
)


# ---------------------------------------------------------------------------
# make_eval_grid
# ---------------------------------------------------------------------------

class TestMakeEvalGrid:
    """Tests for the standard LGC evaluation grid."""

    def test_default_shape(self) -> None:
        """Default 11×11 grid + 5 diagonal points = 126 total."""
        grid = make_eval_grid()
        # 11*11 = 121 + 5 diagonal = 126
        assert grid.shape[1] == 2
        assert grid.shape[0] >= 121

    def test_custom_grid_size(self) -> None:
        """5×5 grid + 5 diagonal = 30 total."""
        grid = make_eval_grid(n_grid=5)
        assert grid.shape[0] >= 25
        assert grid.shape[1] == 2

    def test_grid_within_range(self) -> None:
        """All regular grid points lie in [-3, 3]^2."""
        grid = make_eval_grid(n_grid=11, lo=-3.0, hi=3.0)
        assert np.all(np.isfinite(grid))
        # Regular grid portion
        assert np.all(grid[:121, 0] >= -3.0 - 1e-10)
        assert np.all(grid[:121, 0] <= 3.0 + 1e-10)
        assert np.all(grid[:121, 1] >= -3.0 - 1e-10)
        assert np.all(grid[:121, 1] <= 3.0 + 1e-10)

    def test_diagonal_points_included(self) -> None:
        """Grid must include z_q = (Φ⁻¹(q), Φ⁻¹(q)) for q ∈ {0.01,0.05,0.5,0.95,0.99}."""
        grid = make_eval_grid()
        qs = [0.01, 0.05, 0.5, 0.95, 0.99]
        expected = stats.norm.ppf(qs)
        for z in expected:
            diag_pt = np.array([z, z])
            dists = np.linalg.norm(grid - diag_pt, axis=1)
            assert np.min(dists) < 1e-8, f"Diagonal point ({z:.2f},{z:.2f}) missing"

    def test_custom_range(self) -> None:
        """Custom lo/hi is respected."""
        grid = make_eval_grid(n_grid=7, lo=-2.0, hi=2.0)
        assert np.all(grid[:49, :] >= -2.0 - 1e-10)
        assert np.all(grid[:49, :] <= 2.0 + 1e-10)


# ---------------------------------------------------------------------------
# global_bandwidth
# ---------------------------------------------------------------------------

class TestGlobalBandwidth:
    """Tests for the plug-in bandwidth formula b0 = c * n^(-1/6)."""

    def test_formula_at_n1000(self) -> None:
        """b0 = 1.75 * 1000^(-1/6)."""
        b = global_bandwidth(1000)
        expected = 1.75 * (1000 ** (-1.0 / 6.0))
        assert b == pytest.approx(expected, rel=1e-10)

    def test_bandwidth_decreases_with_n(self) -> None:
        """Larger n gives smaller bandwidth."""
        b1 = global_bandwidth(500)
        b2 = global_bandwidth(5000)
        assert b1 > b2

    def test_custom_constant(self) -> None:
        """Custom c parameter is respected."""
        b = global_bandwidth(1000, c=2.0)
        expected = 2.0 * (1000 ** (-1.0 / 6.0))
        assert b == pytest.approx(expected, rel=1e-10)

    def test_formula_scaling_exponent(self) -> None:
        """Doubling n multiplies b by 2^(-1/6)."""
        b1 = global_bandwidth(1000)
        b2 = global_bandwidth(2000)
        ratio = b2 / b1
        expected_ratio = (2.0) ** (-1.0 / 6.0)
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    @pytest.mark.parametrize("n", [500, 1000, 2500, 5000])
    def test_reasonable_magnitude(self, n: int) -> None:
        """b0 should be in (0.2, 1.0) for typical n values."""
        b = global_bandwidth(n)
        assert 0.2 < b < 1.0, f"Unexpected bandwidth {b:.4f} for n={n}"


# ---------------------------------------------------------------------------
# kernel_weights
# ---------------------------------------------------------------------------

class TestKernelWeights:
    """Tests for the Gaussian product kernel."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(5)
        self.Z = rng.standard_normal((200, 2))
        self.b = 0.5

    def test_output_shape(self) -> None:
        """Kernel weights have shape (n,)."""
        x = np.array([0.0, 0.0])
        w = kernel_weights(self.Z, x, self.b)
        assert w.shape == (self.Z.shape[0],)

    def test_all_positive(self) -> None:
        """All kernel weights are strictly positive (Gaussian kernel)."""
        x = np.array([0.0, 0.0])
        w = kernel_weights(self.Z, x, self.b)
        assert np.all(w > 0.0)

    def test_max_at_closest_point(self) -> None:
        """Weight is highest at the origin when data is centered."""
        x = np.array([0.0, 0.0])
        w_center = kernel_weights(self.Z, x, self.b)
        x_far = np.array([10.0, 10.0])
        w_far = kernel_weights(self.Z, x_far, self.b)
        assert w_center.sum() > w_far.sum()

    def test_weight_at_exact_point_is_one(self) -> None:
        """If Z has a row equal to x, that row's weight = 1.0."""
        Z = np.array([[1.0, 2.0], [0.0, 0.0]])
        x = np.array([1.0, 2.0])
        w = kernel_weights(Z, x, b=1.0)
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_bandwidth_controls_spread(self) -> None:
        """Larger bandwidth gives more weight to distant points."""
        x = np.array([3.0, 3.0])
        w_small = kernel_weights(self.Z, x, b=0.3)
        w_large = kernel_weights(self.Z, x, b=2.0)
        assert w_large.sum() > w_small.sum()

    def test_all_finite(self) -> None:
        """No NaN or Inf in kernel weights."""
        x = np.array([0.0, 0.0])
        w = kernel_weights(self.Z, x, self.b)
        assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# lgc_at_point
# ---------------------------------------------------------------------------

class TestLGCAtPoint:
    """Tests for local Gaussian correlation at a single evaluation point."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(7)
        self.b = 0.6
        # Independent standard normal data → rho ≈ 0
        self.Z_indep = rng.standard_normal((500, 2))
        # Positively correlated → rho > 0
        cov = np.array([[1.0, 0.7], [0.7, 1.0]])
        L = np.linalg.cholesky(cov)
        self.Z_pos = rng.standard_normal((500, 2)) @ L.T

    def test_returns_tuple_of_two(self) -> None:
        """lgc_at_point returns a (rho_hat, eff_n) tuple."""
        result = lgc_at_point(self.Z_indep, np.array([0.0, 0.0]), self.b)
        assert len(result) == 2

    def test_rho_in_valid_range(self) -> None:
        """rho_hat must be in (-1, 1) or NaN."""
        rho, _ = lgc_at_point(self.Z_indep, np.array([0.0, 0.0]), self.b)
        if np.isfinite(rho):
            assert -1.0 <= rho <= 1.0

    def test_eff_n_positive_at_center(self) -> None:
        """Effective sample size is positive at the grid center."""
        _, eff = lgc_at_point(self.Z_indep, np.array([0.0, 0.0]), self.b)
        assert eff > 0.0

    def test_independent_data_rho_near_zero(self) -> None:
        """Independent data should yield |rho| < 0.5 at center (stochastic)."""
        rng = np.random.default_rng(999)
        Z = rng.standard_normal((2000, 2))
        rho, _ = lgc_at_point(Z, np.array([0.0, 0.0]), b=0.7)
        assert np.isfinite(rho)
        assert abs(rho) < 0.4, f"Expected near-zero rho, got {rho:.3f}"

    def test_correlated_data_rho_positive(self) -> None:
        """Positively correlated data should yield rho > 0 at center."""
        rho, _ = lgc_at_point(self.Z_pos, np.array([0.0, 0.0]), self.b)
        assert np.isfinite(rho)
        assert rho > 0.2, f"Expected positive rho, got {rho:.3f}"

    def test_far_point_nan_or_low_eff(self) -> None:
        """Evaluation far from data returns NaN or very low eff_n."""
        rho, eff = lgc_at_point(self.Z_indep, np.array([20.0, 20.0]), b=0.3)
        # Either NaN or very small eff_n
        assert np.isnan(rho) or eff < 2.0

    def test_rho_clipped_to_open_unit_interval(self) -> None:
        """rho_hat should never be exactly ±1."""
        rho, _ = lgc_at_point(self.Z_pos, np.array([0.0, 0.0]), self.b)
        if np.isfinite(rho):
            assert rho < 1.0
            assert rho > -1.0


# ---------------------------------------------------------------------------
# lgc_surface
# ---------------------------------------------------------------------------

class TestLGCSurface:
    """Tests for the full LGC surface over the evaluation grid."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(13)
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        L = np.linalg.cholesky(cov)
        self.Z = rng.standard_normal((300, 2)) @ L.T
        self.grid = make_eval_grid(n_grid=5)   # small 5×5 for speed
        self.b = global_bandwidth(300)

    def test_output_shapes(self) -> None:
        """lgc_surface returns (M,) arrays for rho and eff_n."""
        rho, eff = lgc_surface(self.Z, self.grid, self.b)
        M = self.grid.shape[0]
        assert rho.shape == (M,)
        assert eff.shape == (M,)

    def test_center_rho_finite_and_positive(self) -> None:
        """At the grid center (0,0), rho should be finite and positive."""
        center = np.array([[0.0, 0.0]])
        rho, eff = lgc_surface(self.Z, center, self.b)
        assert np.isfinite(rho[0])
        assert rho[0] > 0.0

    def test_eff_n_non_negative(self) -> None:
        """Effective sample sizes are non-negative."""
        _, eff = lgc_surface(self.Z, self.grid, self.b)
        assert np.all(eff >= 0.0)

    def test_center_has_larger_eff_n_than_corner(self) -> None:
        """Center point has higher eff_n than a far corner."""
        center = np.array([[0.0, 0.0]])
        corner = np.array([[3.0, 3.0]])
        _, eff_c = lgc_surface(self.Z, center, self.b)
        _, eff_e = lgc_surface(self.Z, corner, self.b)
        assert eff_c[0] >= eff_e[0]

    def test_all_rho_in_valid_range(self) -> None:
        """All non-NaN rho values are in (-1, 1)."""
        rho, _ = lgc_surface(self.Z, self.grid, self.b)
        finite_rho = rho[np.isfinite(rho)]
        assert np.all(finite_rho >= -1.0)
        assert np.all(finite_rho <= 1.0)


# ---------------------------------------------------------------------------
# density_weighted_ise
# ---------------------------------------------------------------------------

class TestDensityWeightedISE:
    """Tests for the ISE metric used in all simulation experiments."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(17)
        M = 50
        self.rho_gt = np.clip(rng.standard_normal(M) * 0.3, -0.9, 0.9)
        self.eff_n = np.abs(rng.standard_normal(M)) * 20.0 + 5.0

    def test_ise_zero_for_exact_match(self) -> None:
        """ISE = 0 when rho_hat equals rho_gt exactly."""
        ise = density_weighted_ise(self.rho_gt.copy(), self.rho_gt, self.eff_n)
        assert ise == pytest.approx(0.0, abs=1e-12)

    def test_ise_positive_for_different_surfaces(self) -> None:
        """ISE > 0 when rho_hat differs from rho_gt."""
        rho_hat = self.rho_gt + 0.2
        rho_hat = np.clip(rho_hat, -1.0, 1.0)
        ise = density_weighted_ise(rho_hat, self.rho_gt, self.eff_n)
        assert ise > 0.0

    def test_ise_increases_with_error(self) -> None:
        """Larger error → larger ISE."""
        ise_small = density_weighted_ise(
            self.rho_gt + 0.05, self.rho_gt, self.eff_n)
        ise_large = density_weighted_ise(
            self.rho_gt + 0.2, self.rho_gt, self.eff_n)
        assert ise_large > ise_small

    def test_nan_rho_hat_excluded(self) -> None:
        """Points where rho_hat is NaN are excluded without error."""
        rho_hat = self.rho_gt + 0.1
        rho_hat[5] = np.nan
        ise = density_weighted_ise(rho_hat, self.rho_gt, self.eff_n)
        assert np.isfinite(ise)

    def test_low_eff_n_excluded(self) -> None:
        """Points with eff_n < min_eff_n are excluded."""
        eff_n = self.eff_n.copy()
        eff_n[0] = 0.0  # unsupported
        rho_hat = self.rho_gt + 0.5  # big error at excluded point
        rho_hat[0] = 100.0  # would dominate if not excluded
        ise = density_weighted_ise(rho_hat, self.rho_gt, eff_n, min_eff_n=1.0)
        assert np.isfinite(ise)
        assert ise < 1.0  # dominated point is excluded

    def test_all_unsupported_returns_nan(self) -> None:
        """ISE is NaN when all points are unsupported."""
        eff_n = np.zeros(self.rho_gt.shape)
        ise = density_weighted_ise(self.rho_gt, self.rho_gt, eff_n)
        assert np.isnan(ise)

    def test_ise_symmetric_in_hat_vs_gt(self) -> None:
        """ISE(hat, gt) = ISE(gt, hat) given same weights."""
        rho_hat = self.rho_gt + 0.1
        ise1 = density_weighted_ise(rho_hat, self.rho_gt, self.eff_n)
        ise2 = density_weighted_ise(self.rho_gt, rho_hat, self.eff_n)
        assert ise1 == pytest.approx(ise2, rel=1e-10)

    def test_ise_weights_proportional_to_eff_n(self) -> None:
        """Error at high-eff_n point contributes more to ISE."""
        M = 10
        rho_gt = np.zeros(M)
        rho_hat = np.zeros(M)
        rho_hat[0] = 0.5  # error only at point 0

        eff_low = np.ones(M) * 10.0
        eff_high = eff_low.copy()
        eff_high[0] = 100.0  # high weight at the error point

        ise_low = density_weighted_ise(rho_hat, rho_gt, eff_low)
        ise_high = density_weighted_ise(rho_hat, rho_gt, eff_high)
        assert ise_high > ise_low
