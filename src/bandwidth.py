"""
Bandwidth selection for Local Gaussian Correlation estimation.
# Custom — Context7 found no library equivalent (adaptive LGC bandwidth)

Three bandwidth strategies:
  1. Global plug-in:    b0 = 1.75 * n^(-1/6)
  2. Corrected AMISE adaptive: pointwise adaptive using pilot rho, density,
     and bias functional beta(x)
  3. Heuristic adaptive: based on curvature of pilot rho surface

The adaptive bandwidth follows the AMISE-optimal shape:
  b_raw(x) ∝ [(1-rho(x)^2)^2 / (f(x) * beta(x)^2)]^(1/6) * n^(-1/6)

then budget-neutrally normalized and capped to [0.5, 3]*b0.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple

from src.lgc import lgc_surface, global_bandwidth, kernel_weights


# ---------------------------------------------------------------------------
# Pilot surface utilities
# ---------------------------------------------------------------------------

def pilot_density(Z: np.ndarray, grid: np.ndarray, b: float) -> np.ndarray:
    """
    Estimate marginal density at grid points via Gaussian KDE approximation.

    f_tilde(x) ≈ eff_n(x) / (2*pi*b^2*n)

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    b : float
        Bandwidth.

    Returns
    -------
    np.ndarray, shape (M,)
        Density estimates at grid points.
    """
    n = Z.shape[0]
    M = grid.shape[0]
    f_tilde = np.zeros(M)
    for j in range(M):
        w = kernel_weights(Z, grid[j], b)
        eff_n = w.sum() / (w.max() + 1e-300)
        f_tilde[j] = eff_n / (2.0 * np.pi * b**2 * n)
    return f_tilde


def finite_diff_gradient(
    values: np.ndarray,
    grid_1d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute finite-difference gradient of a scalar field on an nxn grid.

    Parameters
    ----------
    values : np.ndarray, shape (n_side, n_side)
        Field values on regular 2D grid.
    grid_1d : np.ndarray, shape (n_side,)
        Grid coordinates (same for both dimensions).

    Returns
    -------
    (grad_x, grad_y) : each shape (n_side, n_side)
    """
    h = grid_1d[1] - grid_1d[0]
    grad_x = np.gradient(values, h, axis=1)
    grad_y = np.gradient(values, h, axis=0)
    return grad_x, grad_y


def finite_diff_laplacian(values: np.ndarray, grid_1d: np.ndarray) -> np.ndarray:
    """
    Compute finite-difference Laplacian of a scalar field on a regular grid.

    Parameters
    ----------
    values : np.ndarray, shape (n_side, n_side)
        Field values.
    grid_1d : np.ndarray
        1D coordinates.

    Returns
    -------
    np.ndarray, shape (n_side, n_side)
    """
    h = grid_1d[1] - grid_1d[0]
    d2x = np.gradient(np.gradient(values, h, axis=1), h, axis=1)
    d2y = np.gradient(np.gradient(values, h, axis=0), h, axis=0)
    return d2x + d2y


def estimate_beta_field(
    rho_pilot: np.ndarray,
    f_pilot: np.ndarray,
    grid: np.ndarray,
    n_side: int,
    smooth_sigma: float = 1.0,
    beta_floor: float = 1e-3,
) -> np.ndarray:
    """
    Estimate the bias functional beta(x) for the AMISE adaptive bandwidth.

    beta_tilde(x) = Delta rho(x) + 2 * grad[log f(x)] . grad[rho(x)]

    Uses finite differences on the regular grid portion.

    Parameters
    ----------
    rho_pilot : np.ndarray, shape (n_side^2 + 5,)
        Pilot LGC estimates at full evaluation grid (11x11 + diagonal pts).
    f_pilot : np.ndarray, shape (n_side^2 + 5,)
        Pilot density estimates at full grid.
    grid : np.ndarray, shape (M, 2)
        Full evaluation grid.
    n_side : int
        Side length of regular grid (e.g. 11).
    smooth_sigma : float
        Gaussian smoothing sigma for beta stabilization (in grid units).
    beta_floor : float
        Minimum |beta| to avoid division by zero.

    Returns
    -------
    np.ndarray, shape (M,)
        Beta functional estimates at full grid.
    """
    # Work on the regular grid portion only for finite differences
    n_sq = n_side ** 2
    rho_grid = rho_pilot[:n_sq].reshape(n_side, n_side)
    f_grid = f_pilot[:n_sq].reshape(n_side, n_side)

    # Determine 1D grid coordinates from global grid
    x_coords = grid[:n_sq:n_side, 0]  # first column of grid, stride n_side
    # Fallback: use evenly spaced
    lo, hi = grid[:n_sq, 0].min(), grid[:n_sq, 0].max()
    x1d = np.linspace(lo, hi, n_side)

    # Replace NaN pilot values with local smoothed value
    rho_filled = np.where(np.isfinite(rho_grid), rho_grid, 0.0)
    f_filled = np.where(f_grid > 0, f_grid, 1e-8)

    # Gradient of log f
    log_f = np.log(f_filled + 1e-12)
    grad_logf_x, grad_logf_y = finite_diff_gradient(log_f, x1d)

    # Gradient of rho
    grad_rho_x, grad_rho_y = finite_diff_gradient(rho_filled, x1d)

    # Laplacian of rho
    lap_rho = finite_diff_laplacian(rho_filled, x1d)

    # Beta functional on grid
    beta_grid = lap_rho + 2.0 * (grad_logf_x * grad_rho_x + grad_logf_y * grad_rho_y)

    # Smooth to stabilize
    beta_smoothed = gaussian_filter(beta_grid, sigma=smooth_sigma)

    # Floor
    sign_beta = np.sign(beta_smoothed)
    sign_beta[sign_beta == 0] = 1
    beta_floored = sign_beta * np.maximum(np.abs(beta_smoothed), beta_floor)

    # Build full result including diagonal quantile points
    beta_full = np.full(rho_pilot.shape[0], np.nan)
    beta_flat = beta_floored.ravel()
    beta_full[:n_sq] = beta_flat

    # For diagonal quantile points: interpolate from nearest grid points
    for j in range(n_sq, len(rho_pilot)):
        xj = grid[j]
        dists = np.sum((grid[:n_sq] - xj) ** 2, axis=1)
        nearest = np.argmin(dists)
        beta_full[j] = beta_flat[nearest]

    return beta_full


# ---------------------------------------------------------------------------
# Corrected AMISE adaptive bandwidth
# ---------------------------------------------------------------------------

def corrected_adaptive_bandwidth(
    Z: np.ndarray,
    grid: np.ndarray,
    n_side: int = 11,
    c_global: float = 1.75,
    b_lo_frac: float = 0.5,
    b_hi_frac: float = 3.0,
) -> np.ndarray:
    """
    Compute pointwise corrected AMISE adaptive bandwidths.

    Steps:
      1. Global pilot estimate at b0.
      2. Extract pilot rho, eff_n, density.
      3. Estimate beta_tilde via finite differences.
      4. Compute raw adaptive shape.
      5. Budget-neutral normalization.
      6. Cap to [0.5, 3]*b0.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    n_side : int
        Side length of regular grid sub-block.
    c_global : float
        Bandwidth constant for global plug-in.
    b_lo_frac, b_hi_frac : float
        Cap fractions relative to b0.

    Returns
    -------
    np.ndarray, shape (M,)
        Pointwise adaptive bandwidths.
    """
    n = Z.shape[0]
    b0 = global_bandwidth(n, c_global)

    # Step 1: pilot LGC surface
    rho_pilot, eff_n_pilot = lgc_surface(Z, grid, b0)

    # Step 2: pilot density
    f_pilot = pilot_density(Z, grid, b0)

    # Step 3: bias functional beta_tilde
    beta_tilde = estimate_beta_field(
        rho_pilot, f_pilot, grid, n_side=n_side
    )

    # Step 4: raw adaptive shape
    # b_raw(x) ∝ [(1-rho^2)^2 / (f * beta^2)]^(1/6) * n^(-1/6)
    rho2 = np.where(np.isfinite(rho_pilot), rho_pilot ** 2, 0.0)
    numerator = (1.0 - rho2) ** 2
    denom = np.abs(f_pilot + 1e-12) * (beta_tilde ** 2 + 1e-12)
    shape_raw = (numerator / (denom + 1e-12)) ** (1.0 / 6.0)

    # Step 5: budget-neutral normalization — density-weighted geometric mean = b0
    supported = np.isfinite(rho_pilot) & (f_pilot > 0) & np.isfinite(shape_raw)
    if supported.sum() == 0:
        return np.full(grid.shape[0], b0)

    log_shape = np.log(shape_raw[supported] + 1e-12)
    w_norm = f_pilot[supported] / f_pilot[supported].sum()
    log_mean_shape = np.dot(w_norm, log_shape)
    mean_shape = np.exp(log_mean_shape)

    scale = b0 / (mean_shape + 1e-12)
    b_adaptive = shape_raw * scale

    # Step 6: cap
    b_lo = b_lo_frac * b0
    b_hi = b_hi_frac * b0
    b_adaptive = np.clip(b_adaptive, b_lo, b_hi)
    b_adaptive[~np.isfinite(b_adaptive)] = b0

    return b_adaptive


def heuristic_adaptive_bandwidth(
    Z: np.ndarray,
    grid: np.ndarray,
    n_side: int = 11,
    c_global: float = 1.75,
    b_lo_frac: float = 0.5,
    b_hi_frac: float = 3.0,
) -> np.ndarray:
    """
    Heuristic adaptive bandwidth based on local curvature proxy.

    kappa(x) = || Hessian rho_tilde(x) || (Frobenius norm of curvature)
    b_heur(x) ∝ (1-rho^2)^(1/6) * f^(-1/6) * kappa^(-1/3)
    then budget-neutrally normalized and capped.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    n_side : int
        Side length of regular grid block.
    c_global : float
        Global bandwidth constant.
    b_lo_frac, b_hi_frac : float
        Cap fractions.

    Returns
    -------
    np.ndarray, shape (M,)
        Pointwise heuristic adaptive bandwidths.
    """
    n = Z.shape[0]
    b0 = global_bandwidth(n, c_global)

    rho_pilot, eff_n_pilot = lgc_surface(Z, grid, b0)
    f_pilot = pilot_density(Z, grid, b0)

    n_sq = n_side ** 2
    rho_grid = np.where(np.isfinite(rho_pilot[:n_sq]), rho_pilot[:n_sq], 0.0)
    rho_grid = rho_grid.reshape(n_side, n_side)

    lo = grid[:n_sq, 0].min()
    hi = grid[:n_sq, 0].max()
    x1d = np.linspace(lo, hi, n_side)
    h = x1d[1] - x1d[0]

    # Hessian components via finite differences
    d2xx = np.gradient(np.gradient(rho_grid, h, axis=1), h, axis=1)
    d2yy = np.gradient(np.gradient(rho_grid, h, axis=0), h, axis=0)
    d2xy = np.gradient(np.gradient(rho_grid, h, axis=1), h, axis=0)

    # Frobenius norm of Hessian
    kappa_grid = np.sqrt(d2xx**2 + 2*d2xy**2 + d2yy**2)
    kappa_grid = np.maximum(kappa_grid, 1e-4)  # floor

    kappa_full = np.full(rho_pilot.shape[0], 1e-4)
    kappa_full[:n_sq] = kappa_grid.ravel()
    for j in range(n_sq, len(rho_pilot)):
        xj = grid[j]
        dists = np.sum((grid[:n_sq] - xj)**2, axis=1)
        kappa_full[j] = kappa_grid.ravel()[np.argmin(dists)]

    # Heuristic shape
    rho2 = np.where(np.isfinite(rho_pilot), rho_pilot**2, 0.0)
    f_safe = np.maximum(f_pilot, 1e-12)
    shape_raw = ((1 - rho2)**(1/6)) * (f_safe**(-1/6)) * (kappa_full**(-1/3))

    # Budget-neutral normalization
    supported = np.isfinite(rho_pilot) & (f_pilot > 0) & np.isfinite(shape_raw)
    if supported.sum() == 0:
        return np.full(grid.shape[0], b0)

    log_shape = np.log(shape_raw[supported] + 1e-12)
    w_norm = f_pilot[supported] / f_pilot[supported].sum()
    log_mean_shape = np.dot(w_norm, log_shape)
    mean_shape = np.exp(log_mean_shape)

    scale = b0 / (mean_shape + 1e-12)
    b_heur = np.clip(shape_raw * scale, b_lo_frac * b0, b_hi_frac * b0)
    b_heur[~np.isfinite(b_heur)] = b0

    return b_heur


# ---------------------------------------------------------------------------
# LGC surface with per-point bandwidth
# ---------------------------------------------------------------------------

def lgc_surface_adaptive(
    Z: np.ndarray,
    grid: np.ndarray,
    bandwidths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate LGC surface using per-point bandwidths.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    bandwidths : np.ndarray, shape (M,)
        Per-point bandwidth values.

    Returns
    -------
    (rho_hat, eff_n) : each shape (M,)
    """
    from src.lgc import lgc_at_point
    M = grid.shape[0]
    rho_hat = np.full(M, np.nan)
    eff_n = np.zeros(M)

    for j in range(M):
        rho_hat[j], eff_n[j] = lgc_at_point(Z, grid[j], bandwidths[j])

    return rho_hat, eff_n
