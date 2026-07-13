"""
Experiment 4 – Bias-Expansion Validation for LGC Estimator.

Under the second-order bias expansion:

  E[ rho_hat(x; b) ] ≈ rho(x) + beta(x) * b^2

the per-point slope of E[rho_hat] against b^2 should recover beta(x).
This experiment checks the theory by:
  (a) sweeping a bandwidth ladder b_j = c_j * n^(-1/6),
  (b) estimating E[rho_hat](x; b_j) via R replications per bandwidth,
  (c) regressing the mean estimates on b_j^2 at each grid point to obtain
      empirical slopes c(x),
  (d) comparing c(x) to the pilot-based beta_tilde(x) via correlation and
      a through-origin proportionality constant gamma.

Outputs
-------
results/exp_4_bias_expansion.csv  : per-point slope, intercept, R^2, beta_tilde
results/exp_4_slope_vs_beta.png   : scatter of c(x) vs beta_tilde(x)
"""

# ---------------------------------------------------------------------------
# FAST_MODE flag — set to False for the full experiment
# ---------------------------------------------------------------------------
FAST_MODE = True

import pathlib
import sys

# Ensure project root is on sys.path so 'src.*' imports work from any cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.nts import symmetric_nts
from src.copulas import simulate_copula_with_margins
from src.lgc import lgc_surface, global_bandwidth, make_eval_grid
from src.transforms import gaussianize
from src.simulation import build_truth_surface
from src.bandwidth import estimate_beta_field, pilot_density
from exp.utils import ensure_results_dir, save_results, RESULTS_DIR


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

_BW_MULTIPLIERS_FULL = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
_FAMILIES_FULL = ["clayton", "gumbel"]

_FULL = dict(
    R=120,
    BW_MULTIPLIERS=_BW_MULTIPLIERS_FULL,
    FAMILIES=_FAMILIES_FULL,
)

_FAST = dict(
    R=10,
    BW_MULTIPLIERS=_BW_MULTIPLIERS_FULL[:2],
    FAMILIES=["clayton"],
)

N = 4_000
N_GT = 20_000
TAU = 0.5
N_SIDE = 11
TRANSFORM = "empirical"
BASE_SEED = 4000
VERBOSE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_beta_tilde(
    family: str,
    tau: float,
    marginal,
    grid: np.ndarray,
    n_gt: int,
    seed: int,
) -> np.ndarray:
    """
    Build a pilot sample and estimate the bias functional beta_tilde(x).

    Steps
    -----
    1. Simulate n_gt observations and Gaussianize with empirical transform.
    2. Estimate pilot LGC surface at global_bandwidth(n_gt).
    3. Estimate pilot density at each grid point.
    4. Call estimate_beta_field to obtain beta_tilde(x).

    Parameters
    ----------
    family, tau : copula specification
    marginal    : NIGDistribution
    grid        : np.ndarray, shape (M, 2)
    n_gt        : int, pilot sample size
    seed        : int, random seed

    Returns
    -------
    np.ndarray, shape (M,)  — beta_tilde at each grid point
    """
    rng = np.random.default_rng(seed)
    _, X_pilot = simulate_copula_with_margins(family, tau, n_gt, marginal, rng)
    Z_pilot = gaussianize(X_pilot, transform_type="empirical")
    b_pilot = global_bandwidth(n_gt)
    rho_pilot, _ = lgc_surface(Z_pilot, grid, b_pilot)
    f_pilot = pilot_density(Z_pilot, grid, b_pilot)
    return estimate_beta_field(rho_pilot, f_pilot, grid, n_side=N_SIDE)


def _mean_rho_over_reps(
    family: str,
    tau: float,
    n: int,
    marginal,
    grid: np.ndarray,
    b: float,
    R: int,
    base_seed: int,
) -> np.ndarray:
    """
    Estimate mean LGC surface at bandwidth b over R replications.

    Each replication draws a fresh sample; the same seeds are used for all
    bandwidths so replications are paired across the bandwidth ladder.

    Parameters
    ----------
    family, tau : copula specification
    n           : int, sample size
    marginal    : NIGDistribution
    grid        : np.ndarray, shape (M, 2)
    b           : float, bandwidth for this run
    R           : int, number of replications
    base_seed   : int, seed base (rep r gets seed base_seed + r * 100)

    Returns
    -------
    np.ndarray, shape (M,)  — mean rho_hat across R replications (NaN where unsupported)
    """
    M = len(grid)
    rho_hat_reps = np.full((R, M), np.nan)

    for r in range(R):
        seed_r = base_seed + r * 100
        rng = np.random.default_rng(seed_r)
        _, X = simulate_copula_with_margins(family, tau, n, marginal, rng)
        Z = gaussianize(X, transform_type=TRANSFORM)
        rho_hat, _ = lgc_surface(Z, grid, b)
        rho_hat_reps[r] = rho_hat

    return np.nanmean(rho_hat_reps, axis=0)


def _per_point_regression(
    mean_rho_arr: np.ndarray,
    b_sq: np.ndarray,
) -> tuple:
    """
    Regress mean_rho_hat(x; b_j) on b_j^2 independently at each grid point.

    Parameters
    ----------
    mean_rho_arr : np.ndarray, shape (n_bw, M)
        Mean LGC estimates; rows = bandwidths, columns = grid points.
    b_sq         : np.ndarray, shape (n_bw,)
        Squared bandwidths b_j^2.

    Returns
    -------
    slopes     : np.ndarray, shape (M,)  — c(x): slope vs b^2
    intercepts : np.ndarray, shape (M,)  — a(x): intercept (≈ rho_gt)
    r2_vals    : np.ndarray, shape (M,)  — R^2 of per-point fit
    """
    M = mean_rho_arr.shape[1]
    slopes = np.full(M, np.nan)
    intercepts = np.full(M, np.nan)
    r2_vals = np.full(M, np.nan)

    for j in range(M):
        y_j = mean_rho_arr[:, j]
        valid = np.isfinite(y_j)
        if valid.sum() < 2:
            continue
        result = scipy_stats.linregress(b_sq[valid], y_j[valid])
        slopes[j] = result.slope
        intercepts[j] = result.intercept
        r2_vals[j] = result.rvalue ** 2

    return slopes, intercepts, r2_vals


def _compute_summary_stats(
    slopes: np.ndarray,
    beta_tilde: np.ndarray,
    r2_vals: np.ndarray,
) -> dict:
    """
    Compute summary statistics: median R^2, Pearson correlation, gamma.

    gamma is the through-origin OLS proportionality constant:
      gamma = sum( c(x) * beta_tilde(x) ) / sum( beta_tilde(x)^2 )

    Parameters
    ----------
    slopes     : np.ndarray, shape (M,)  — empirical slopes c(x)
    beta_tilde : np.ndarray, shape (M,)  — pilot bias functional
    r2_vals    : np.ndarray, shape (M,)  — per-point R^2

    Returns
    -------
    dict with keys: median_r2, correlation, pvalue, gamma, n_supported
    """
    supported = np.isfinite(slopes) & np.isfinite(beta_tilde)
    n_sup = int(supported.sum())

    median_r2 = float(np.nanmedian(r2_vals))

    if n_sup >= 3:
        corr, pval = scipy_stats.pearsonr(slopes[supported], beta_tilde[supported])
        bt = beta_tilde[supported]
        cx = slopes[supported]
        gamma = float(np.dot(cx, bt) / np.dot(bt, bt))
    else:
        corr, pval, gamma = np.nan, np.nan, np.nan

    return {
        "median_r2": median_r2,
        "correlation": float(corr),
        "pvalue": float(pval),
        "gamma": gamma,
        "n_supported": n_sup,
    }


def _run_family(
    family: str,
    R: int,
    bw_multipliers: np.ndarray,
    marginal,
    grid: np.ndarray,
) -> pd.DataFrame:
    """
    Run the full bias-expansion experiment for one copula family.

    Parameters
    ----------
    family        : str, copula family
    R             : int, replications per bandwidth
    bw_multipliers: np.ndarray, shape (n_bw,) — c_j in b_j = c_j * n^(-1/6)
    marginal      : NIGDistribution
    grid          : np.ndarray, shape (M, 2)

    Returns
    -------
    pd.DataFrame with per-point results including slope_c, intercept_a, r2,
    beta_tilde, and summary stats (median_r2, corr_c_beta, gamma) broadcast
    to every row.
    """
    M = len(grid)
    bandwidths = bw_multipliers * (N ** (-1.0 / 6.0))
    n_bw = len(bandwidths)
    b_sq = bandwidths ** 2

    truth_seed = BASE_SEED + hash(family) % 10_000
    pilot_seed = truth_seed + 1

    # Step 1: ground truth surface
    print(f"  Building ground-truth surface (n_gt={N_GT}) ...")
    rho_gt = build_truth_surface(family, TAU, marginal, grid, n_gt=N_GT, seed=truth_seed)

    # Step 5: pilot beta_tilde
    print(f"  Computing beta_tilde from pilot (n_gt={N_GT}) ...")
    beta_tilde = _compute_beta_tilde(family, TAU, marginal, grid, N_GT, seed=pilot_seed)

    # Steps 2–3: mean LGC for each bandwidth across R reps
    mean_rho_arr = np.full((n_bw, M), np.nan)

    for bi, b_j in enumerate(bandwidths):
        c_j = bw_multipliers[bi]
        if VERBOSE:
            print(f"  bw_mult={c_j:.1f}  b={b_j:.4f}  ({R} reps) ...")
        mean_rho_arr[bi] = _mean_rho_over_reps(
            family, TAU, N, marginal, grid, b_j, R, base_seed=BASE_SEED
        )
        n_valid = int(np.isfinite(mean_rho_arr[bi]).sum())
        print(f"    → {n_valid}/{M} supported grid points")

    # Step 4: per-point regression on b^2
    slopes, intercepts, r2_vals = _per_point_regression(mean_rho_arr, b_sq)

    # Steps 6–7: summary statistics
    stats = _compute_summary_stats(slopes, beta_tilde, r2_vals)
    print(
        f"  median R² = {stats['median_r2']:.4f} | "
        f"corr(c, β̃) = {stats['correlation']:.4f} | "
        f"γ = {stats['gamma']:.4f} | "
        f"n_supported = {stats['n_supported']}"
    )

    # Step 8: assemble per-point DataFrame
    records = []
    for j in range(M):
        records.append({
            "family": family,
            "point_idx": j,
            "x1": grid[j, 0],
            "x2": grid[j, 1],
            "rho_gt": rho_gt[j],
            "slope_c": slopes[j],
            "intercept_a": intercepts[j],
            "r2": r2_vals[j],
            "beta_tilde": beta_tilde[j],
            "median_r2": stats["median_r2"],
            "corr_c_beta": stats["correlation"],
            "gamma": stats["gamma"],
        })

    return pd.DataFrame(records)


def _plot_slope_vs_beta(df_results: pd.DataFrame, families: list, out_path: pathlib.Path) -> None:
    """
    Scatter plot of empirical slope c(x) vs pilot bias functional beta_tilde(x).

    One colour per copula family; through-origin fit line using all families
    combined.

    Parameters
    ----------
    df_results : pd.DataFrame  Point-level results for all families.
    families   : list of str   Family names in the order to plot.
    out_path   : pathlib.Path  Output PNG path.
    """
    colours = plt.cm.tab10(np.linspace(0, 0.4, len(families)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for colour, family in zip(colours, families):
        df_f = df_results[df_results["family"] == family]
        mask = np.isfinite(df_f["slope_c"].to_numpy()) & np.isfinite(df_f["beta_tilde"].to_numpy())
        ax.scatter(
            df_f["beta_tilde"].to_numpy()[mask],
            df_f["slope_c"].to_numpy()[mask],
            alpha=0.55,
            s=20,
            color=colour,
            label=family,
        )

    # Through-origin fit line using all families pooled
    all_bt = df_results["beta_tilde"].to_numpy()
    all_cx = df_results["slope_c"].to_numpy()
    all_mask = np.isfinite(all_bt) & np.isfinite(all_cx)
    if all_mask.sum() >= 2:
        gamma_all = float(np.dot(all_cx[all_mask], all_bt[all_mask]) / np.dot(all_bt[all_mask], all_bt[all_mask]))
        x_range = np.array([all_bt[all_mask].min(), all_bt[all_mask].max()])
        ax.plot(
            x_range,
            gamma_all * x_range,
            "k--",
            lw=2,
            label=f"Through-origin: γ = {gamma_all:.3f}",
        )
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.axvline(0, color="grey", lw=0.8, ls=":")

    ax.set_xlabel("β̃(x)  [pilot bias functional]", fontsize=12)
    ax.set_ylabel("c(x)  [empirical slope vs b²]", fontsize=12)
    ax.set_title(
        "Exp 4 — Bias-Expansion Validation: Empirical Slope vs β̃(x)",
        fontsize=11,
        pad=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = _FAST if FAST_MODE else _FULL
    R = settings["R"]
    BW_MULTIPLIERS = settings["BW_MULTIPLIERS"]
    FAMILIES = settings["FAMILIES"]

    bandwidths = BW_MULTIPLIERS * (N ** (-1.0 / 6.0))

    mode_label = "FAST" if FAST_MODE else "FULL"
    print("=" * 60)
    print(f"Experiment 4 — Bias-Expansion Validation  [{mode_label} MODE]")
    print(f"  copulas    : {FAMILIES}")
    print(f"  n          : {N}  |  tau={TAU}  |  n_gt={N_GT}")
    print(f"  bw mults   : {BW_MULTIPLIERS.tolist()}")
    print(f"  bandwidths : {np.round(bandwidths, 4).tolist()}")
    print(f"  R          : {R}  |  transform={TRANSFORM}")
    print("=" * 60)

    marginal = symmetric_nts()
    grid = make_eval_grid()

    all_frames = []
    for family in FAMILIES:
        print(f"\n[Exp 4] family={family}")
        df_fam = _run_family(family, R, BW_MULTIPLIERS, marginal, grid)
        all_frames.append(df_fam)

    df_results = pd.concat(all_frames, ignore_index=True)

    # Step 8: save per-point results
    ensure_results_dir()
    save_results(df_results, "exp_4_bias_expansion.csv")

    # Step 7: print consolidated summary
    print("\n--- Summary by family ---")
    summary_cols = ["family", "median_r2", "corr_c_beta", "gamma"]
    summary = (
        df_results[summary_cols]
        .drop_duplicates(subset=["family"])
        .reset_index(drop=True)
    )
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))

    # Step 9: scatter plot
    _plot_slope_vs_beta(df_results, FAMILIES, RESULTS_DIR / "exp_4_slope_vs_beta.png")


if __name__ == "__main__":
    main()
