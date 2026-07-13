"""
Bivariate copula simulation using pyvinecopulib.
# Using pyvinecopulib Bicop — Context7 confirmed

Supported families: Gaussian, Clayton, t (nu=4), Gumbel.
All families parameterized via Kendall's tau mapping.

Tau-to-parameter mappings (standard formulas):
  - Gaussian:  rho = sin(pi * tau / 2)
  - Clayton:   theta = 2*tau/(1-tau)         (lower-tail dependence)
  - t(nu):     rho = sin(pi * tau / 2)       (same as Gaussian)
  - Gumbel:    theta = 1/(1-tau)             (upper-tail dependence)
"""

from __future__ import annotations

import numpy as np
import pyvinecopulib as pv
from typing import Literal

CopulaFamily = Literal["gaussian", "clayton", "t4", "gumbel"]


# ---------------------------------------------------------------------------
# Tau → parameter mapping (standard closed-form formulas)
# ---------------------------------------------------------------------------

def tau_to_param(family: CopulaFamily, tau: float) -> float | tuple[float, float]:
    """
    Convert Kendall's tau to the canonical copula parameter.

    Parameters
    ----------
    family : str
        Copula family: 'gaussian', 'clayton', 't4', 'gumbel'.
    tau : float
        Kendall's rank correlation, in (0, 1).

    Returns
    -------
    float or (float, float)
        Copula parameter(s). t-copula returns (rho, nu).
    """
    if family == "gaussian":
        return float(np.sin(np.pi * tau / 2.0))
    elif family == "clayton":
        assert tau > 0, "Clayton requires tau > 0"
        return float(2.0 * tau / (1.0 - tau))
    elif family == "t4":
        rho = float(np.sin(np.pi * tau / 2.0))
        return (rho, 4.0)
    elif family == "gumbel":
        assert tau >= 0, "Gumbel requires tau >= 0"
        return float(1.0 / (1.0 - tau))
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Copula factory
# ---------------------------------------------------------------------------

def make_bicop(family: CopulaFamily, tau: float) -> pv.Bicop:
    """
    Construct a pyvinecopulib Bicop for the given family and Kendall's tau.

    Parameters
    ----------
    family : str
        Copula family identifier.
    tau : float
        Kendall's rank correlation in (0, 1).

    Returns
    -------
    pv.Bicop
        Configured bivariate copula object.
    """
    # Using pyvinecopulib Bicop — Context7 confirmed
    param = tau_to_param(family, tau)
    if family == "gaussian":
        return pv.Bicop(family=pv.BicopFamily.gaussian,
                        parameters=np.array([[param]]))
    elif family == "clayton":
        return pv.Bicop(family=pv.BicopFamily.clayton,
                        parameters=np.array([[param]]))
    elif family == "t4":
        rho, nu = param
        return pv.Bicop(family=pv.BicopFamily.student,
                        parameters=np.array([[rho], [nu]]))
    elif family == "gumbel":
        return pv.Bicop(family=pv.BicopFamily.gumbel,
                        parameters=np.array([[param]]))
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Simulation from copula with NTS marginals
# ---------------------------------------------------------------------------

def simulate_copula_with_margins(
    family: CopulaFamily,
    tau: float,
    n: int,
    marginal,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n bivariate samples from a copula with NTS marginals.

    Steps:
      1. Generate n uniform draws from the copula (U_1, U_2).
      2. Apply the marginal inverse CDF: X_k = F^{-1}(U_k).

    Parameters
    ----------
    family : str
        Copula family.
    tau : float
        Kendall's tau.
    n : int
        Sample size.
    marginal : NIGDistribution
        Marginal distribution with ppf() method.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    (U, X) : tuple of np.ndarray
        U: uniform copula draws (n, 2).
        X: transformed samples with NTS marginals (n, 2).
    """
    cop = make_bicop(family, tau)
    seed_int = int(rng.integers(0, 2**31))
    U = cop.simulate(n=n, seeds=[seed_int, seed_int + 1])

    # Clip to avoid PPF at exactly 0 or 1
    eps = 1e-12
    U_clipped = np.clip(U, eps, 1.0 - eps)
    X = marginal.ppf(U_clipped)
    return U, X


# ---------------------------------------------------------------------------
# Available copula families (for iteration in experiments)
# ---------------------------------------------------------------------------

ALL_FAMILIES: list[CopulaFamily] = ["gaussian", "clayton", "t4", "gumbel"]
