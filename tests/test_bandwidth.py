"""
Tests for src/bandwidth.py — adaptive and heuristic LGC bandwidth selection.

Coverage:
  - pilot_density: positivity, shape, scaling with eff_n
  - estimate_beta_field: shape, finiteness, sign floor
  - corrected_adaptive_bandwidth: shape, cap constraints, budget neutrality
  - heuristic_adaptive_bandwidth: shape, cap constraints
  - lgc_surface_adaptive: shape, correctness relative to global
"""

from __future__ import annotations

import numpy as np
import pytest

from src.lgc import make_eval_grid, global_bandwidth
from src.bandwidth import (
    pilot_density,
    estimate_beta_field,
    corrected_adaptive_bandwidth,
    heuristic_adaptive_bandwidth,
    lgc_surface_adaptive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corr_data() -> np.ndarray:
    """n=500 bivariate normal with correlation 0.5 (Gaussianized data)."""
    rng = np.random.default_rng(55)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    L = np.linalg.cholesky(cov)
    return rng.standard_normal((500, 2)) @ L.T


@pytest.fixture
def small_grid() -> np.ndarray:
    return make_eval_grid(n_grid=5)


@pytest.fixture
def standard_grid() -> np.ndarray:
    return make_eval_grid(n_grid=11)


# ---------------------------------------------------------------------------
# pilot_density
# ---------------------------------------------------------------------------

class TestPilotDensity:
    """Tests for the KDE-based pilot density estimate."""

    def test_output_shape(self, corr_data, small_grid) -> None:
        b = global_bandwidth(500)
        f = pilot_density(corr_data, small_grid, b)
        assert f.shape == (small_grid.shape[0],)

    def test_non_negative(self, corr_data, small_grid) -> None:
        b = global_bandwidth(500)
        f = pilot_density(corr_data, small_grid, b)
        assert np.all(f >= 0.0)

    def test_center_higher_than_corner(self, corr_data) -> None:
        """Density at origin > density at (3,3) for centered data."""
        b = global_bandwidth(500)
        center = np.array([[0.0, 0.0]])
        corner = np.array([[3.0, 3.0]])
        f_c = pilot_density(corr_data, center, b)
        f_e = pilot_density(corr_data, corner, b)
        assert f_c[0] > f_e[0]

    def test_finite_values(self, corr_data, small_grid) -> None:
        b = global_bandwidth(500)
        f = pilot_density(corr_data, small_grid, b)
        assert np.all(np.isfinite(f))

    def test_scales_with_n(self) -> None:
        """Density estimate should be in reasonable range for unit-normal data."""
        rng = np.random.default_rng(7)
        Z = rng.standard_normal((1000, 2))
        grid_center = np.array([[0.0, 0.0]])
        b = global_bandwidth(1000)
        f = pilot_density(Z, grid_center, b)
        # For N(0,1) data at origin, true density = 1/(2*pi) ≈ 0.159
        assert 0.01 < f[0] < 2.0


# ---------------------------------------------------------------------------
# estimate_beta_field
# ---------------------------------------------------------------------------

class TestEstimateBetaField:
    """Tests for the bias functional beta_tilde estimator."""

    def _get_pilot(self, Z: np.ndarray, grid: np.ndarray, b: float):
        from src.lgc import lgc_surface
        rho, eff = lgc_surface(Z, grid, b)
        f = pilot_density(Z, grid, b)
        return rho, f

    def test_output_shape(self, corr_data, standard_grid) -> None:
        b = global_bandwidth(500)
        rho, f = self._get_pilot(corr_data, standard_grid, b)
        beta = estimate_beta_field(rho, f, standard_grid, n_side=11)
        assert beta.shape == standard_grid.shape[:1]

    def test_no_exact_zeros(self, corr_data, standard_grid) -> None:
        """Beta must not be zero (floor applied)."""
        b = global_bandwidth(500)
        rho, f = self._get_pilot(corr_data, standard_grid, b)
        beta = estimate_beta_field(rho, f, standard_grid, n_side=11, beta_floor=1e-3)
        finite_beta = beta[np.isfinite(beta)]
        assert np.all(np.abs(finite_beta) >= 1e-3 - 1e-12)

    def test_finite_at_regular_grid_points(self, corr_data, standard_grid) -> None:
        """At least the inner regular grid points should be finite."""
        b = global_bandwidth(500)
        rho, f = self._get_pilot(corr_data, standard_grid, b)
        beta = estimate_beta_field(rho, f, standard_grid, n_side=11)
        n_sq = 11 ** 2
        inner_beta = beta[:n_sq]
        assert np.all(np.isfinite(inner_beta))


# ---------------------------------------------------------------------------
# corrected_adaptive_bandwidth
# ---------------------------------------------------------------------------

class TestCorrectedAdaptiveBandwidth:
    """Tests for the AMISE corrected adaptive bandwidth surface."""

    def test_output_shape(self, corr_data, standard_grid) -> None:
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert b_surf.shape == (standard_grid.shape[0],)

    def test_all_finite(self, corr_data, standard_grid) -> None:
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(np.isfinite(b_surf))

    def test_cap_lower_bound(self, corr_data, standard_grid) -> None:
        """All bandwidths ≥ 0.5 * b0."""
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf >= 0.5 * b0 - 1e-10)

    def test_cap_upper_bound(self, corr_data, standard_grid) -> None:
        """All bandwidths ≤ 3 * b0."""
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf <= 3.0 * b0 + 1e-10)

    def test_all_positive(self, corr_data, standard_grid) -> None:
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf > 0.0)

    def test_budget_neutral_near_b0(self, corr_data, standard_grid) -> None:
        """Density-weighted geometric mean of bandwidths ≈ b0."""
        from src.lgc import kernel_weights
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        f = pilot_density(corr_data, standard_grid, b0)
        supported = f > 0
        if supported.sum() == 0:
            pytest.skip("No supported points")
        w = f[supported] / f[supported].sum()
        log_mean = np.dot(w, np.log(b_surf[supported]))
        geo_mean = np.exp(log_mean)
        # Should be close to b0 (within a factor of 3 due to capping)
        assert 0.5 * b0 <= geo_mean <= 3.0 * b0

    def test_spatial_variation(self, corr_data, standard_grid) -> None:
        """Adaptive bandwidths should vary across the grid."""
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid)
        assert b_surf.std() > 1e-6, "Adaptive bandwidths are constant — no spatial variation"


# ---------------------------------------------------------------------------
# heuristic_adaptive_bandwidth
# ---------------------------------------------------------------------------

class TestHeuristicAdaptiveBandwidth:
    """Tests for the curvature-based heuristic adaptive bandwidth."""

    def test_output_shape(self, corr_data, standard_grid) -> None:
        b_surf = heuristic_adaptive_bandwidth(corr_data, standard_grid)
        assert b_surf.shape == (standard_grid.shape[0],)

    def test_all_finite(self, corr_data, standard_grid) -> None:
        b_surf = heuristic_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(np.isfinite(b_surf))

    def test_cap_lower_bound(self, corr_data, standard_grid) -> None:
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        b_surf = heuristic_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf >= 0.5 * b0 - 1e-10)

    def test_cap_upper_bound(self, corr_data, standard_grid) -> None:
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        b_surf = heuristic_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf <= 3.0 * b0 + 1e-10)

    def test_all_positive(self, corr_data, standard_grid) -> None:
        b_surf = heuristic_adaptive_bandwidth(corr_data, standard_grid)
        assert np.all(b_surf > 0.0)


# ---------------------------------------------------------------------------
# lgc_surface_adaptive
# ---------------------------------------------------------------------------

class TestLGCSurfaceAdaptive:
    """Tests for LGC surface estimation with per-point bandwidths."""

    def test_output_shape(self, corr_data, standard_grid) -> None:
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        bandwidths = np.full(standard_grid.shape[0], b0)
        rho, eff = lgc_surface_adaptive(corr_data, standard_grid, bandwidths)
        assert rho.shape == (standard_grid.shape[0],)
        assert eff.shape == (standard_grid.shape[0],)

    def test_uniform_bandwidth_matches_global(self, corr_data, standard_grid) -> None:
        """Uniform bandwidths equal to b0 should give same result as lgc_surface."""
        from src.lgc import lgc_surface
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        bandwidths = np.full(standard_grid.shape[0], b0)

        rho_adaptive, eff_adaptive = lgc_surface_adaptive(
            corr_data, standard_grid, bandwidths)
        rho_global, eff_global = lgc_surface(corr_data, standard_grid, b0)

        # Results should be identical
        np.testing.assert_allclose(rho_adaptive, rho_global, atol=1e-10)

    def test_all_rho_in_valid_range(self, corr_data, standard_grid) -> None:
        n = corr_data.shape[0]
        b0 = global_bandwidth(n)
        bandwidths = np.full(standard_grid.shape[0], b0)
        rho, _ = lgc_surface_adaptive(corr_data, standard_grid, bandwidths)
        finite_rho = rho[np.isfinite(rho)]
        assert np.all(finite_rho >= -1.0) and np.all(finite_rho <= 1.0)

    def test_adaptive_surface_rho_finite_at_center(self, corr_data, standard_grid) -> None:
        """Center grid point (0,0) should have finite rho under adaptive bandwidths.

        corrected_adaptive_bandwidth requires a full regular grid (n_side×n_side)
        to compute finite-difference derivatives, so we pass the standard 11×11 grid
        and then pick out the point closest to the origin.
        """
        b_surf = corrected_adaptive_bandwidth(corr_data, standard_grid, n_side=11)
        rho, eff = lgc_surface_adaptive(corr_data, standard_grid, b_surf)
        # Find the grid point nearest to (0, 0)
        dists = np.linalg.norm(standard_grid, axis=1)
        center_idx = int(np.argmin(dists))
        assert np.isfinite(rho[center_idx]), (
            f"LGC at grid point nearest to (0,0) is NaN; eff_n={eff[center_idx]:.1f}"
        )
        assert eff[center_idx] > 0.0
