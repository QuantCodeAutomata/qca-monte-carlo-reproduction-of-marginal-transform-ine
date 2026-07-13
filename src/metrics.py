"""
Metrics and diagnostic utilities for LGC simulation experiments.
# Custom — Context7 found no library equivalent (LGC-specific ISE metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def pointwise_diagnostics(
    rho_hat_reps: np.ndarray,
    rho_gt: np.ndarray,
    eff_n_reps: np.ndarray,
) -> pd.DataFrame:
    """
    Compute pointwise bias, SD, RMSE, and Fisher-normalized SD across replications.

    Parameters
    ----------
    rho_hat_reps : np.ndarray, shape (R, M)
        LGC estimates across R replications and M grid points.
    rho_gt : np.ndarray, shape (M,)
        Ground-truth LGC values.
    eff_n_reps : np.ndarray, shape (R, M)
        Effective sample sizes.

    Returns
    -------
    pd.DataFrame
        Columns: bias, sd, rmse, fisher_sd, mean_eff_n, per grid point.
    """
    bias = np.nanmean(rho_hat_reps, axis=0) - rho_gt
    sd = np.nanstd(rho_hat_reps, axis=0, ddof=0)
    rmse = np.sqrt(bias**2 + sd**2)
    fisher_sd = sd / np.maximum(1.0 - rho_gt**2, 1e-6)
    mean_eff_n = np.nanmean(eff_n_reps, axis=0)

    return pd.DataFrame({
        "bias": bias,
        "sd": sd,
        "rmse": rmse,
        "fisher_sd": fisher_sd,
        "mean_eff_n": mean_eff_n,
    })


def aggregate_ise(
    ise_values: np.ndarray,
) -> Dict[str, float]:
    """
    Aggregate ISE values across replications.

    Parameters
    ----------
    ise_values : np.ndarray, shape (R,)
        ISE per replication.

    Returns
    -------
    dict with keys 'mean_ise', 'sd_ise'.
    """
    valid = ise_values[np.isfinite(ise_values)]
    return {
        "mean_ise": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "sd_ise": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
    }


def scarcity_regression(
    sd_vals: np.ndarray,
    rho_gt_vals: np.ndarray,
    eff_n_vals: np.ndarray,
    min_eff_n: float = 5.0,
) -> Dict[str, float]:
    """
    Fit log-log regression: log[sd/(1-rho^2)] ~ a + b*log(eff_n).

    Parameters
    ----------
    sd_vals : np.ndarray
        Pointwise standard deviations.
    rho_gt_vals : np.ndarray
        Pointwise ground-truth LGC values.
    eff_n_vals : np.ndarray
        Pointwise effective sample sizes.
    min_eff_n : float
        Minimum eff_n to include in regression.

    Returns
    -------
    dict with keys 'slope', 'intercept', 'r_squared'.
    """
    import statsmodels.api as sm

    fisher_sd = sd_vals / np.maximum(1.0 - rho_gt_vals**2, 1e-6)
    mask = (
        (eff_n_vals >= min_eff_n)
        & np.isfinite(sd_vals)
        & np.isfinite(rho_gt_vals)
        & np.isfinite(eff_n_vals)
        & (sd_vals > 0)
        & (fisher_sd > 0)
    )
    if mask.sum() < 5:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "n_obs": 0}

    y = np.log(fisher_sd[mask])
    x = np.log(eff_n_vals[mask])
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    return {
        "intercept": float(model.params[0]),
        "slope": float(model.params[1]),
        "r_squared": float(model.rsquared),
        "n_obs": int(mask.sum()),
    }
