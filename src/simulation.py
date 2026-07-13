"""
Core Monte Carlo simulation runner for LGC experiments.
# Custom — Context7 found no library equivalent (LGC simulation pipeline)

Implements paired design: same raw sample processed by all three transforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from src.copulas import simulate_copula_with_margins, ALL_FAMILIES, CopulaFamily
from src.lgc import lgc_surface, global_bandwidth, make_eval_grid, density_weighted_ise
from src.transforms import gaussianize, TransformType
from src.nts import NIGDistribution, symmetric_nts, skewed_nts


# ---------------------------------------------------------------------------
# Ground-truth surface construction
# ---------------------------------------------------------------------------

def build_truth_surface(
    family: CopulaFamily,
    tau: float,
    marginal: NIGDistribution,
    grid: np.ndarray,
    n_gt: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    """
    Construct a DGP-specific ground-truth LGC surface.

    Uses an independent auxiliary sample of size n_gt Gaussianized via
    the empirical transform, then fits LGC at the evaluation grid.

    Parameters
    ----------
    family : str
        Copula family.
    tau : float
        Kendall's tau.
    marginal : NIGDistribution
        Marginal distribution.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    n_gt : int
        Ground-truth sample size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (M,)
        Ground-truth LGC surface.
    """
    rng = np.random.default_rng(seed)
    _, X_gt = simulate_copula_with_margins(family, tau, n_gt, marginal, rng)
    Z_gt = gaussianize(X_gt, transform_type="empirical")
    b_gt = global_bandwidth(n_gt)
    rho_gt, _ = lgc_surface(Z_gt, grid, b_gt)
    return rho_gt


# ---------------------------------------------------------------------------
# Single replication: all three transforms
# ---------------------------------------------------------------------------

def run_replication(
    family: CopulaFamily,
    tau: float,
    n: int,
    marginal: NIGDistribution,
    grid: np.ndarray,
    rho_gt: np.ndarray,
    transforms: List[TransformType],
    replication: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Run a single Monte Carlo replication with all specified transforms.

    Paired design: same raw sample used for all transforms.

    Parameters
    ----------
    family : str
        Copula family.
    tau : float
        Kendall's tau.
    n : int
        Sample size.
    marginal : NIGDistribution
        Marginal distribution.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    rho_gt : np.ndarray, shape (M,)
        Ground-truth LGC surface.
    transforms : list of str
        Transform types to apply.
    replication : int
        Replication index.
    base_seed : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        Per-transform ISE for this replication.
    """
    seed = base_seed + replication * 100
    rng = np.random.default_rng(seed)

    # Simulate raw sample (shared across transforms)
    _, X = simulate_copula_with_margins(family, tau, n, marginal, rng)
    b0 = global_bandwidth(n)

    records = []
    for tf in transforms:
        Z = gaussianize(X, transform_type=tf, marginal=marginal)
        rho_hat, eff_n = lgc_surface(Z, grid, b0)
        ise = density_weighted_ise(rho_hat, rho_gt, eff_n)

        for j in range(len(grid)):
            records.append({
                "family": family,
                "tau": tau,
                "n": n,
                "replication": replication,
                "transform": tf,
                "point_idx": j,
                "x1": grid[j, 0],
                "x2": grid[j, 1],
                "rho_hat": rho_hat[j],
                "rho_gt": rho_gt[j],
                "eff_n": eff_n[j],
                "ISE": ise,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full Monte Carlo cell (one copula × n)
# ---------------------------------------------------------------------------

def run_mc_cell(
    family: CopulaFamily,
    tau: float,
    n: int,
    n_reps: int,
    marginal: NIGDistribution,
    grid: np.ndarray,
    transforms: List[TransformType],
    n_gt: int = 20_000,
    base_seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run complete Monte Carlo experiment for one (family, tau, n) cell.

    Parameters
    ----------
    family : str
        Copula family.
    tau : float
        Kendall's tau.
    n : int
        Sample size.
    n_reps : int
        Number of replications.
    marginal : NIGDistribution
        Marginal distribution.
    grid : np.ndarray
        Evaluation grid.
    transforms : list of str
        Transform variants.
    n_gt : int
        Ground-truth auxiliary sample size.
    base_seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Per-replication, per-transform, per-point results.
    """
    # Build ground-truth surface (one per DGP)
    truth_seed = base_seed + hash(family) % 10000
    rho_gt = build_truth_surface(family, tau, marginal, grid, n_gt, seed=truth_seed)

    all_reps = []
    for r in range(n_reps):
        if verbose:
            print(f"  {family} tau={tau} n={n} rep={r+1}/{n_reps}")
        df_rep = run_replication(
            family, tau, n, marginal, grid, rho_gt, transforms, r, base_seed
        )
        all_reps.append(df_rep)

    return pd.concat(all_reps, ignore_index=True)
