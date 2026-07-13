"""
Local Gaussian Correlation (LGC) estimator.
# Custom — Context7 found no library equivalent (paper Eq. for local Gaussian likelihood)

The LGC at evaluation point x is estimated as the correlation parameter
of the locally-weighted bivariate Gaussian MLE:

  rho_hat(x) = locally-weighted Pearson correlation with kernel weights
               K_b(Z_i - x) = exp(-||Z_i - x||^2 / (2*b^2))

This corresponds to maximizing the local log pseudo-likelihood
  sum_i K_b(Z_i - x) * log phi_2(Z_i; mu, Sigma)
over (mu, Sigma), which has the closed-form MLE equal to the
weighted sample mean and covariance matrix.

Bandwidth convention: product Gaussian kernel,
  K_b(u1, u2) = exp(-(u1^2 + u2^2)/(2*b^2))

Global plug-in bandwidth:  b0 = 1.75 * n^(-1/6)    (paper default)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple


# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------

def make_eval_grid(n_grid: int = 11, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """
    Build the standard evaluation grid for LGC experiments.

    Returns an (M, 2) array of evaluation points: 11x11 regular grid over
    [-3,3]^2 augmented with 5 diagonal quantile points.

    Parameters
    ----------
    n_grid : int
        Grid side length (default 11).
    lo, hi : float
        Grid range (default -3 to 3).

    Returns
    -------
    np.ndarray of shape (M, 2)
    """
    # 11x11 regular grid
    g = np.linspace(lo, hi, n_grid)
    gx, gy = np.meshgrid(g, g)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    # Diagonal quantile points z_q = (Phi^{-1}(q), Phi^{-1}(q))
    q_vals = [0.01, 0.05, 0.5, 0.95, 0.99]
    z_diag = stats.norm.ppf(q_vals)
    diag_points = np.column_stack([z_diag, z_diag])

    # Combine and deduplicate (approximately)
    all_points = np.vstack([grid_points, diag_points])
    return all_points  # shape (126, 2) or similar


# ---------------------------------------------------------------------------
# Global plug-in bandwidth
# ---------------------------------------------------------------------------

def global_bandwidth(n: int, c: float = 1.75) -> float:
    """
    Global plug-in bandwidth: b0 = c * n^(-1/6).

    Parameters
    ----------
    n : int
        Sample size.
    c : float
        Bandwidth constant (default 1.75 per paper).

    Returns
    -------
    float
        Bandwidth value.
    """
    return c * (n ** (-1.0 / 6.0))


# ---------------------------------------------------------------------------
# Kernel weights
# ---------------------------------------------------------------------------

def kernel_weights(Z: np.ndarray, x: np.ndarray, b: float) -> np.ndarray:
    """
    Compute unnormalized Gaussian product kernel weights K_b(Z_i - x).

    K_b(u1, u2) = exp(-(u1^2 + u2^2) / (2*b^2))

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    x : np.ndarray, shape (2,)
        Evaluation point.
    b : float
        Bandwidth.

    Returns
    -------
    np.ndarray, shape (n,)
        Kernel weights (unnormalized).
    """
    diff = Z - x[np.newaxis, :]          # (n, 2)
    sq_dist = np.sum(diff ** 2, axis=1)  # (n,)
    return np.exp(-sq_dist / (2.0 * b ** 2))


# ---------------------------------------------------------------------------
# LGC at a single evaluation point
# ---------------------------------------------------------------------------

def lgc_at_point(
    Z: np.ndarray,
    x: np.ndarray,
    b: float,
    min_weight_sum: float = 1e-10,
) -> Tuple[float, float]:
    """
    Estimate Local Gaussian Correlation and effective sample size at point x.

    Uses the closed-form MLE for the locally-weighted bivariate Gaussian,
    equivalent to maximizing the local pseudo-log-likelihood.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data matrix.
    x : np.ndarray, shape (2,)
        Evaluation point.
    b : float
        Kernel bandwidth.
    min_weight_sum : float
        Minimum total weight to consider the point supported.

    Returns
    -------
    (rho_hat, eff_n) : tuple
        rho_hat in [-1, 1]: local Gaussian correlation estimate.
        eff_n > 0: local effective sample size.
    """
    w = kernel_weights(Z, x, b)          # (n,)
    w_sum = w.sum()

    if w_sum < min_weight_sum:
        return np.nan, 0.0

    # Effective sample size: w_sum / max(w_i)   (paper definition)
    eff_n = w_sum / (w.max() + 1e-300)

    # Weighted means
    mu1 = np.dot(w, Z[:, 0]) / w_sum
    mu2 = np.dot(w, Z[:, 1]) / w_sum

    # Weighted centered deviations
    dz1 = Z[:, 0] - mu1
    dz2 = Z[:, 1] - mu2

    # Weighted variances
    var1 = np.dot(w, dz1 ** 2) / w_sum
    var2 = np.dot(w, dz2 ** 2) / w_sum

    if var1 < 1e-14 or var2 < 1e-14:
        return np.nan, eff_n

    # Weighted covariance → correlation
    cov12 = np.dot(w, dz1 * dz2) / w_sum
    rho = cov12 / np.sqrt(var1 * var2)
    rho = float(np.clip(rho, -1.0 + 1e-8, 1.0 - 1e-8))

    return rho, eff_n


# ---------------------------------------------------------------------------
# LGC surface over a grid
# ---------------------------------------------------------------------------

def lgc_surface(
    Z: np.ndarray,
    grid: np.ndarray,
    b: float,
    min_eff_n: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate LGC and effective sample size at all grid points.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    b : float
        Kernel bandwidth.
    min_eff_n : float
        Minimum eff_n to mark a point as supported (informational only).

    Returns
    -------
    (rho_hat, eff_n) : tuple of np.ndarray, each shape (M,)
    """
    M = grid.shape[0]
    rho_hat = np.full(M, np.nan)
    eff_n = np.zeros(M)

    for j in range(M):
        rho_hat[j], eff_n[j] = lgc_at_point(Z, grid[j], b)

    return rho_hat, eff_n


# ---------------------------------------------------------------------------
# Density-weighted ISE
# ---------------------------------------------------------------------------

def density_weighted_ise(
    rho_hat: np.ndarray,
    rho_gt: np.ndarray,
    eff_n: np.ndarray,
    min_eff_n: float = 1.0,
) -> float:
    """
    Compute density-weighted integrated squared error (ISE).

    ISE = sum_{x in G, supported} w(x) * (rho_hat(x) - rho_gt(x))^2

    Weights w(x) are proportional to eff_n(x) (proxy for local density).
    Unsupported or NaN points are excluded.

    Parameters
    ----------
    rho_hat : np.ndarray, shape (M,)
        Estimated LGC surface.
    rho_gt : np.ndarray, shape (M,)
        Ground-truth LGC surface.
    eff_n : np.ndarray, shape (M,)
        Effective sample sizes.
    min_eff_n : float
        Minimum eff_n for a point to be included.

    Returns
    -------
    float
        Density-weighted ISE.
    """
    supported = (eff_n >= min_eff_n) & np.isfinite(rho_hat) & np.isfinite(rho_gt)
    if supported.sum() == 0:
        return np.nan

    sq_err = (rho_hat[supported] - rho_gt[supported]) ** 2
    weights = eff_n[supported]
    weights = weights / weights.sum()
    return float(np.dot(weights, sq_err))
