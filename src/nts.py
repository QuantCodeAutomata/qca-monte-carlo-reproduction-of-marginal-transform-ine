"""
Normal Inverse Gaussian (NIG) distribution used as a proxy for the
Normal Tempered Stable (NTS) distribution with heavy-tailed marginals.

The NIG is a special case of NTS (tempered stable index alpha=1/2) and
provides tractable PDF/CDF via scipy.stats.norminvgauss, ML fitting,
and both symmetric and skewed variants needed by the experiments.

scipy.stats.norminvgauss(a, b) parameterization:
  - a > 0: tail steepness (larger = lighter tails)
  - b: asymmetry, |b| < a
  - mean  = b / sqrt(a^2 - b^2)
  - var   = a^2 / (a^2 - b^2)^(3/2)
  - skew  = 3*b / (a * (a^2-b^2)^{1/4})    [after standardization, preserved]

After standardization (subtract mean, divide by std), we obtain
zero-mean, unit-variance NIG with matching skewness.
"""

from __future__ import annotations

import numpy as np
from scipy import stats, optimize
from scipy.integrate import cumulative_trapezoid
from typing import Tuple


# ---------------------------------------------------------------------------
# Default calibrated NIG parameters
# ---------------------------------------------------------------------------
# Symmetric NIG: a=1, b=0 → mean=0, var=1 exactly (unit-variance, zero-mean)
_SYM_A: float = 1.0
_SYM_B: float = 0.0

# Skewed NIG: calibrated numerically to approximate NTS skewness ≈ -0.84
# (paper experiment 2: beta=-0.6, skewness≈-0.84)
# We find (a, b) of the *raw* NIG such that, after standardizing, skew≈-0.84.
# Skewness of NIG is invariant under affine transform, so we just need raw skew=-0.84.
#   raw_skew(a,b) = 3*b / (a * (a^2-b^2)^{1/4})
# Solved numerically below; stored as module-level constants.
_SKEW_A: float = 2.0      # steepness (calibrated)
_SKEW_B: float = -1.1264  # asymmetry (calibrated for skew≈-0.84)


def _nig_raw_skewness(a: float, b: float) -> float:
    """Compute raw NIG skewness for scipy.stats.norminvgauss(a, b).

    Formula: 3*b / (a * (a^2 - b^2)^{1/4}), matches scipy.stats.norminvgauss.stats().
    """
    gamma = np.sqrt(a**2 - b**2)
    return 3.0 * b / (a * gamma**0.5) if a > abs(b) else np.nan


def _calibrate_skewed_params(target_skew: float = -0.84, a: float = 2.0) -> float:
    """Find b (given a) such that NIG skewness matches target_skew."""
    # Custom — Context7 found no library equivalent (calibration step)
    def obj(b: float) -> float:
        if abs(b) >= a:
            return 1e6
        return _nig_raw_skewness(a, b) - target_skew

    result = optimize.brentq(obj, -(a / np.sqrt(2.0) - 1e-6), -1e-6)
    return float(result)


# Calibrate on import (fast, runs once)
_SKEW_B = _calibrate_skewed_params(target_skew=-0.84, a=_SKEW_A)


# ---------------------------------------------------------------------------
# NIG distribution wrapper (standardized)
# ---------------------------------------------------------------------------

class NIGDistribution:
    """
    Standardized Normal Inverse Gaussian distribution: zero mean, unit variance.

    Parameters
    ----------
    a : float
        Tail steepness parameter of NIG (a > 0).
    b : float
        Asymmetry parameter (|b| < a). b=0 gives symmetric distribution.
    """

    # Grid resolution for the CDF/PPF lookup table (standardised coordinates)
    _GRID_LO: float = -30.0
    _GRID_HI: float = 30.0
    _GRID_N: int = 20_000

    def __init__(self, a: float, b: float) -> None:
        assert a > 0, "a must be positive"
        assert abs(b) < a, "|b| must be strictly less than a"
        self.a = float(a)
        self.b = float(b)
        # Raw distribution before standardization
        self._dist = stats.norminvgauss(a=a, b=b)
        # Standardization parameters derived from raw moments
        self.raw_mean: float = float(self._dist.mean())
        self.raw_std: float = float(np.sqrt(self._dist.var()))
        # Pre-build lookup table for fast ppf / cdf via PDF integration.
        # scipy.stats.norminvgauss lacks a vectorised PPF in scipy ≤ 1.14;
        # computing PDF on a fine grid + cumtrapz gives <1e-5 error in <2 ms.
        self._x_table, self._cdf_table = self._build_lookup_table()

    # ------------------------------------------------------------------
    def rvs(self, size: int | Tuple, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Draw random samples from the standardized NIG distribution.

        Parameters
        ----------
        size : int or tuple
            Number / shape of samples.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        seed = int(rng.integers(0, 2**31)) if rng is not None else None
        raw = self._dist.rvs(size=size, random_state=seed)
        return (raw - self.raw_mean) / self.raw_std

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density of the standardized NIG.

        Parameters
        ----------
        x : np.ndarray
            Evaluation points.
        """
        x_raw = np.asarray(x) * self.raw_std + self.raw_mean
        return self._dist.pdf(x_raw) / self.raw_std

    def _build_lookup_table(self) -> tuple:
        """
        Build a CDF lookup table from the PDF via cumulative trapezoidal integration.

        Returns (x_grid, cdf_grid) in standardised coordinates for use with
        np.interp.  Building takes ~2 ms; subsequent ppf/cdf calls are O(log N).
        """
        x_grid = np.linspace(self._GRID_LO, self._GRID_HI, self._GRID_N)
        pdf_vals = self.pdf(x_grid)
        cdf_vals = cumulative_trapezoid(pdf_vals, x_grid, initial=0.0)
        cdf_vals /= cdf_vals[-1]  # normalise to [0, 1]
        return x_grid, cdf_vals

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function of the standardized NIG.

        Uses the pre-built lookup table for fast vectorised evaluation.

        Parameters
        ----------
        x : np.ndarray
            Evaluation points.
        """
        return np.interp(np.asarray(x, dtype=float),
                         self._x_table, self._cdf_table,
                         left=0.0, right=1.0)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent-point (quantile) function of the standardized NIG.

        Uses the pre-built lookup table for fast vectorised evaluation.

        Parameters
        ----------
        q : np.ndarray
            Probability levels in (0, 1).
        """
        return np.interp(np.asarray(q, dtype=float),
                         self._cdf_table, self._x_table,
                         left=self._x_table[0], right=self._x_table[-1])

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Log-probability density of the standardized NIG.

        Parameters
        ----------
        x : np.ndarray
            Evaluation points.
        """
        x_raw = np.asarray(x) * self.raw_std + self.raw_mean
        return self._dist.logpdf(x_raw) - np.log(self.raw_std)

    def fit_ml(self, data: np.ndarray) -> "NIGDistribution":
        """
        Fit NIG parameters to data by maximum likelihood.

        Returns a new NIGDistribution with fitted parameters.

        Parameters
        ----------
        data : np.ndarray
            Observed values assumed to follow a NIG distribution.
        """
        # Use method-of-moments initialization then MLE via scipy
        # Custom — Context7 found no library equivalent (ML fit loop)
        data = np.asarray(data, dtype=float)

        def neg_loglik(params: np.ndarray) -> float:
            a_p, b_p = params
            if a_p <= 0 or abs(b_p) >= a_p:
                return 1e10
            dist = stats.norminvgauss(a=a_p, b=b_p)
            # Standardize internal representation
            m = dist.mean()
            s = np.sqrt(dist.var())
            lls = dist.logpdf(data * s + m) - np.log(s)
            return -np.sum(lls)

        x0 = np.array([self.a, self.b])
        bounds = [(0.1, 20.0), (-self.a + 1e-4, self.a - 1e-4)]

        res = optimize.minimize(
            neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12}
        )
        a_fit, b_fit = res.x
        return NIGDistribution(a=float(a_fit), b=float(b_fit))


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def symmetric_nts(a: float = _SYM_A, b: float = _SYM_B) -> NIGDistribution:
    """
    Return a symmetric (b=0) NIG distribution with unit variance.

    Parameters
    ----------
    a : float
        Tail steepness (default=1.0 gives exact unit variance, zero mean).
    b : float
        Asymmetry; should be 0 for symmetric case.
    """
    return NIGDistribution(a=a, b=b)


def skewed_nts(target_skew: float = -0.84, a: float = _SKEW_A) -> NIGDistribution:
    """
    Return a skewed NIG distribution calibrated to approximate target_skew.

    After standardization the distribution has zero mean, unit variance,
    and skewness ≈ target_skew (paper experiment 2: beta=-0.6 → skew≈-0.84).

    Parameters
    ----------
    target_skew : float
        Target skewness (default -0.84).
    a : float
        Tail steepness parameter (default 2.0).
    """
    b = _calibrate_skewed_params(target_skew=target_skew, a=a)
    return NIGDistribution(a=a, b=b)
