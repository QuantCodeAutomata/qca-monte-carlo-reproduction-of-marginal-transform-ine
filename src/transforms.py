"""
Gaussianization transforms for LGC estimation.
# Custom — Context7 found no library equivalent (rank-based and parametric transforms)

Three transform variants:
  1. 'empirical': rank-based empirical CDF + Phi^{-1}
                  rank convention: (rank - 0.5) / n  (avoids 0 and 1)
  2. 'oracle':    true NTS marginal CDF + Phi^{-1}
  3. 'fitted':    ML-fitted NTS marginal CDF + Phi^{-1}

All transforms output bivariate data in approximately N(0,1) marginals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Literal

TransformType = Literal["empirical", "oracle", "fitted"]


# ---------------------------------------------------------------------------
# Helper: apply Gaussianization to a single margin
# ---------------------------------------------------------------------------

def _gaussianize_column(u: np.ndarray) -> np.ndarray:
    """
    Transform uniform [0,1] values to standard normal via Phi^{-1}.

    Parameters
    ----------
    u : np.ndarray, shape (n,)
        Values in (0, 1).

    Returns
    -------
    np.ndarray, shape (n,)
        Standard normal scores.
    """
    u_clipped = np.clip(u, 1e-10, 1.0 - 1e-10)
    return stats.norm.ppf(u_clipped)


def empirical_transform(X: np.ndarray) -> np.ndarray:
    """
    Canonical empirical normal-score transform.

    For each column: compute rank-based empirical CDF (rank-0.5)/n
    then apply Phi^{-1}.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Bivariate data with NTS marginals.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Gaussianized data.
    """
    n = X.shape[0]
    Z = np.empty_like(X, dtype=float)
    for k in range(X.shape[1]):
        ranks = stats.rankdata(X[:, k])          # ranks from 1 to n
        uniform_scores = (ranks - 0.5) / n        # avoids 0 and 1
        Z[:, k] = _gaussianize_column(uniform_scores)
    return Z


def oracle_transform(X: np.ndarray, marginal) -> np.ndarray:
    """
    Oracle transform using the true NTS marginal CDF + Phi^{-1}.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Bivariate data with NTS marginals.
    marginal : NIGDistribution
        The true marginal distribution (same for both margins by symmetry).

    Returns
    -------
    np.ndarray, shape (n, 2)
        Gaussianized data.
    """
    Z = np.empty_like(X, dtype=float)
    for k in range(X.shape[1]):
        u = marginal.cdf(X[:, k])
        Z[:, k] = _gaussianize_column(u)
    return Z


def fitted_transform(X: np.ndarray, marginal) -> np.ndarray:
    """
    Fitted parametric transform: ML-fit NTS marginal on each margin separately,
    then apply fitted CDF + Phi^{-1}.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Bivariate data with NTS marginals.
    marginal : NIGDistribution
        Base NTS distribution used for initialization of ML fitting.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Gaussianized data with fitted parametric marginal transforms.
    """
    Z = np.empty_like(X, dtype=float)
    for k in range(X.shape[1]):
        fitted_marginal = marginal.fit_ml(X[:, k])
        u = fitted_marginal.cdf(X[:, k])
        Z[:, k] = _gaussianize_column(u)
    return Z


def gaussianize(
    X: np.ndarray,
    transform_type: TransformType,
    marginal=None,
) -> np.ndarray:
    """
    Dispatch to the appropriate Gaussianization transform.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Raw bivariate sample with NTS marginals.
    transform_type : str
        One of 'empirical', 'oracle', 'fitted'.
    marginal : NIGDistribution or None
        Required for 'oracle' and 'fitted' transforms.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Gaussianized bivariate data.
    """
    if transform_type == "empirical":
        return empirical_transform(X)
    elif transform_type == "oracle":
        assert marginal is not None, "marginal required for oracle transform"
        return oracle_transform(X, marginal)
    elif transform_type == "fitted":
        assert marginal is not None, "marginal required for fitted transform"
        return fitted_transform(X, marginal)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
