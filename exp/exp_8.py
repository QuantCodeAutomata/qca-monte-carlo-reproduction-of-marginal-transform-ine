"""
Experiment 8 - Real-Data Application: Bootstrap Stability of Global vs Adaptive LGC.

Purpose:
    Apply LGC estimation to filtered financial return pairs (SPY/TLT, SPY/EEM),
    and measure how bootstrap variability differs between the global plug-in and
    the corrected adaptive bandwidth estimators across the joint distribution surface.

Settings:
    - ETF pairs: SPY/TLT and SPY/EEM (full mode) / SPY/TLT only (FAST_MODE)
    - Daily log returns 2005-2024 after AR(1)-GARCH(1,1)-t filtering
    - B = 200 i.i.d. bootstrap resamples (full) / 10 (FAST_MODE)
    - Grid: make_eval_grid() (11x11 + 5 diagonal = 126 points)

Fallback (if data download fails):
    Synthetic bivariate normal residuals: cov=[[1,0.3],[0.3,1]], n=1000, seed=42.

Outputs:
    results/exp_8_garch_diagnostics.csv   -- per-ticker GARCH model diagnostics
    results/exp_8_bootstrap_stability.csv -- per-(pair, grid-point) bootstrap SDs + ratio
    results/exp_8_lgc_surfaces.png        -- global and adaptive rho surfaces per pair
    results/exp_8_sd_ratio.png            -- scatter of SD ratio vs eff_n per pair
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lgc import (
    lgc_surface,
    global_bandwidth,
    make_eval_grid,
)
from src.transforms import gaussianize
from src.bandwidth import corrected_adaptive_bandwidth, lgc_surface_adaptive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST_MODE = True

if FAST_MODE:
    PAIRS = [("SPY", "TLT")]
    B = 10
else:
    PAIRS = [("SPY", "TLT"), ("SPY", "EEM")]
    B = 200

EFF_N_THRESHOLD = 40      # "sparse" band threshold for median SD-ratio report
N_SIDE = 11               # make_eval_grid regular grid side
N_GRID_REG = N_SIDE ** 2  # 121 regular grid points

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# GARCH filtering
# ---------------------------------------------------------------------------

def _safe_param(params: pd.Series, *names: str) -> float:
    """Extract the first matching parameter name; return NaN if none found."""
    for name in names:
        if name in params.index:
            return float(params[name])
    return np.nan


def fit_garch(returns_series: pd.Series, ticker: str) -> tuple[pd.Series, dict]:
    """
    Fit AR(1)-GARCH(1,1)-t to a return series and return standardized residuals.

    Parameters
    ----------
    returns_series : pd.Series
        Log-return series (daily, as raw fractions).
    ticker : str
        Ticker label for diagnostics.

    Returns
    -------
    std_resid : pd.Series
        Standardized residuals indexed by date.
    diag : dict
        Model diagnostics dictionary.
    """
    from arch import arch_model
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Scale to percentage returns for numerical stability; std_resid is unaffected
    r_pct = returns_series * 100.0

    am = arch_model(r_pct, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = am.fit(disp="off")

    std_resid = result.std_resid.dropna()

    # Ljung-Box on squared standardized residuals (lag=10)
    lb = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    params = result.params
    diag = {
        "ticker": ticker,
        "n_obs": int(len(std_resid)),
        "log_likelihood": float(result.loglikelihood),
        "ar1_coef": _safe_param(params, "ar.L1", "ar.L1[r_pct]", "phi.L1", "Phi.L1"),
        "omega": _safe_param(params, "omega"),
        "alpha": _safe_param(params, "alpha[1]"),
        "beta": _safe_param(params, "beta[1]"),
        "nu_df": _safe_param(params, "nu"),
        "lb_sq_stat_lag10": lb_stat,
        "lb_sq_pvalue_lag10": lb_pvalue,
    }

    return std_resid, diag


def generate_synthetic_residuals(
    n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback: synthetic bivariate normal residuals as specified.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    resid1, resid2 : np.ndarray, each shape (n,)
    """
    np.random.seed(seed)
    cov = [[1.0, 0.3], [0.3, 1.0]]
    residuals = np.random.multivariate_normal([0.0, 0.0], cov, n)
    return residuals[:, 0], residuals[:, 1]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_filter(tickers: list[str]) -> tuple[dict[str, pd.Series], list[dict]]:
    """
    Load ETF prices, compute log returns, and apply AR(1)-GARCH(1,1)-t filtering.

    Falls back to synthetic residuals for all tickers if download or GARCH fails.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols needed.

    Returns
    -------
    std_resids : dict[str, pd.Series]
        Standardized residuals per ticker (date-indexed).
    garch_diags : list[dict]
        GARCH diagnostics per ticker.
    """
    std_resids: dict[str, pd.Series] = {}
    garch_diags: list[dict] = []

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
        from download import download_etf_data, compute_returns

        prices = download_etf_data()
        returns_df = compute_returns(prices, log_returns=True)

        for ticker in tickers:
            if ticker not in returns_df.columns:
                raise KeyError(f"{ticker} not in downloaded data")
            ret = returns_df[ticker].dropna()
            std_resid, diag = fit_garch(ret, ticker)
            std_resids[ticker] = std_resid
            garch_diags.append(diag)
            print(f"  GARCH fitted for {ticker}: n={diag['n_obs']}, "
                  f"alpha={diag['alpha']:.4f}, beta={diag['beta']:.4f}, "
                  f"nu={diag['nu_df']:.2f}, LB-p={diag['lb_sq_pvalue_lag10']:.3f}")

    except Exception as exc:
        print(f"  Data/GARCH pipeline failed ({exc}); using synthetic residuals.")
        n_synth = 1000
        r1, r2 = generate_synthetic_residuals(n=n_synth, seed=42)
        dates = pd.date_range("2005-01-01", periods=n_synth, freq="B")
        for i, ticker in enumerate(tickers):
            synth = r1 if i == 0 else r2
            std_resids[ticker] = pd.Series(synth, index=dates, name=ticker)
            garch_diags.append({
                "ticker": ticker,
                "n_obs": n_synth,
                "log_likelihood": np.nan,
                "ar1_coef": np.nan,
                "omega": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "nu_df": np.nan,
                "lb_sq_stat_lag10": np.nan,
                "lb_sq_pvalue_lag10": np.nan,
            })

    return std_resids, garch_diags


def align_pair(
    std_resids: dict[str, pd.Series],
    ticker_a: str,
    ticker_b: str,
) -> np.ndarray:
    """
    Align two standardized-residual series on common dates and return as (n, 2) array.

    Parameters
    ----------
    std_resids : dict[str, pd.Series]
        Standardized residuals per ticker.
    ticker_a, ticker_b : str
        The two tickers forming the pair.

    Returns
    -------
    np.ndarray, shape (n, 2)
    """
    sa = std_resids[ticker_a]
    sb = std_resids[ticker_b]
    common = sa.index.intersection(sb.index)
    arr = np.column_stack([sa.loc[common].values, sb.loc[common].values])
    return arr


# ---------------------------------------------------------------------------
# LGC surface fitting (both estimators)
# ---------------------------------------------------------------------------

def fit_lgc_both(
    Z: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit global and corrected-adaptive LGC surfaces.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.

    Returns
    -------
    rho_global, eff_n_global, rho_adaptive, eff_n_adaptive : each shape (M,)
    """
    n = Z.shape[0]
    b0 = global_bandwidth(n)

    rho_global, eff_n_global = lgc_surface(Z, grid, b0)

    b_adaptive = corrected_adaptive_bandwidth(Z, grid)
    rho_adaptive, eff_n_adaptive = lgc_surface_adaptive(Z, grid, b_adaptive)

    return rho_global, eff_n_global, rho_adaptive, eff_n_adaptive


# ---------------------------------------------------------------------------
# Bootstrap stability analysis
# ---------------------------------------------------------------------------

def bootstrap_stability(
    Z: np.ndarray,
    grid: np.ndarray,
    B: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run B i.i.d. bootstrap resamples and collect rho_hat for both estimators.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized standardized residuals for a single pair.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    B : int
        Number of bootstrap resamples.
    seed : int
        RNG seed.

    Returns
    -------
    rho_global_boots : np.ndarray, shape (B, M)
    rho_adaptive_boots : np.ndarray, shape (B, M)
    """
    n, M = Z.shape[0], grid.shape[0]
    rho_global_boots = np.full((B, M), np.nan)
    rho_adaptive_boots = np.full((B, M), np.nan)

    rng = np.random.default_rng(seed)

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        Z_boot = Z[idx]

        b0_boot = global_bandwidth(n)
        rho_g, _ = lgc_surface(Z_boot, grid, b0_boot)
        rho_global_boots[b] = rho_g

        b_adapt = corrected_adaptive_bandwidth(Z_boot, grid)
        rho_a, _ = lgc_surface_adaptive(Z_boot, grid, b_adapt)
        rho_adaptive_boots[b] = rho_a

        if (b + 1) % max(1, B // 5) == 0:
            print(f"    bootstrap {b+1}/{B}", flush=True)

    return rho_global_boots, rho_adaptive_boots


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_lgc_surfaces(
    grid: np.ndarray,
    pair_results: list[dict],
    out_path: str,
) -> None:
    """
    Plot global and adaptive rho surfaces for each pair on the regular 11x11 grid.

    Parameters
    ----------
    grid : np.ndarray, shape (M, 2)
        Evaluation grid from make_eval_grid().
    pair_results : list[dict]
        Each dict has keys: 'pair_label', 'rho_global', 'rho_adaptive'.
    out_path : str
        File path for saved figure.
    """
    n_pairs = len(pair_results)
    g1d = np.linspace(-3.0, 3.0, N_SIDE)

    fig, axes = plt.subplots(
        n_pairs, 2,
        figsize=(10, 4.5 * n_pairs),
        squeeze=False,
    )

    cmap = "RdBu_r"
    vmin, vmax = -1.0, 1.0

    for row, res in enumerate(pair_results):
        label = res["pair_label"]

        for col, (rho, estimator) in enumerate([
            (res["rho_global"], "Global LGC"),
            (res["rho_adaptive"], "Adaptive LGC"),
        ]):
            ax = axes[row, col]
            rho_2d = np.full((N_SIDE, N_SIDE), np.nan)
            if rho is not None:
                rho_2d = rho[:N_GRID_REG].reshape(N_SIDE, N_SIDE)

            pm = ax.pcolormesh(g1d, g1d, rho_2d, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading="auto")
            fig.colorbar(pm, ax=ax, label=r"$\hat{\rho}(z)$")
            ax.set_xlabel(r"$Z_1$", fontsize=11)
            ax.set_ylabel(r"$Z_2$", fontsize=11)
            ax.set_title(f"{label} — {estimator}", fontsize=12)
            ax.set_aspect("equal")

    fig.suptitle("Local Gaussian Correlation Surfaces", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sd_ratio_scatter(
    pair_results: list[dict],
    out_path: str,
) -> None:
    """
    Scatter plot of SD ratio (adaptive / global) vs eff_n for each pair.

    Parameters
    ----------
    pair_results : list[dict]
        Each dict has keys: 'pair_label', 'eff_n_orig', 'sd_ratio'.
    out_path : str
        File path for saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]
    colors = ["steelblue", "coral", "forestgreen"]

    for i, res in enumerate(pair_results):
        eff_n = res["eff_n_orig"]
        sd_ratio = res["sd_ratio"]
        mask = np.isfinite(eff_n) & np.isfinite(sd_ratio)
        ax.scatter(
            eff_n[mask], sd_ratio[mask],
            label=res["pair_label"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            alpha=0.6, s=30,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="ratio = 1 (equal stability)")
    ax.axvline(EFF_N_THRESHOLD, color="gray", linestyle=":", linewidth=1.2,
               label=f"eff_n = {EFF_N_THRESHOLD}")
    ax.set_xlabel("Effective sample size (eff_n)", fontsize=12)
    ax.set_ylabel(r"SD ratio  (SD$_\mathrm{adaptive}$ / SD$_\mathrm{global}$)",
                  fontsize=12)
    ax.set_title("Bootstrap SD Ratio vs Effective Sample Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Experiment 8 — FAST_MODE={FAST_MODE}")
    print(f"  Pairs: {PAIRS}, B={B}")

    grid = make_eval_grid()
    M = grid.shape[0]

    # -----------------------------------------------------------------------
    # Step 1-4: Load data, compute returns, filter with GARCH, align pairs
    # -----------------------------------------------------------------------
    print("\n--- Step 1-4: Data loading and GARCH filtering ---")
    all_tickers = sorted({t for pair in PAIRS for t in pair})
    std_resids, garch_diags = load_and_filter(all_tickers)

    # Save GARCH diagnostics
    df_garch = pd.DataFrame(garch_diags)
    garch_path = os.path.join(RESULTS_DIR, "exp_8_garch_diagnostics.csv")
    df_garch.to_csv(garch_path, index=False)
    print(f"Saved {garch_path}")

    # -----------------------------------------------------------------------
    # Step 5-10: Per-pair analysis
    # -----------------------------------------------------------------------
    boot_records: list[dict] = []
    pair_plot_data: list[dict] = []
    pair_sd_data: list[dict] = []

    for ticker_a, ticker_b in PAIRS:
        pair_label = f"{ticker_a}/{ticker_b}"
        print(f"\n--- Pair: {pair_label} ---")

        # Align residuals on common dates
        raw_pair = align_pair(std_resids, ticker_a, ticker_b)
        n_pair = raw_pair.shape[0]
        print(f"  Aligned residuals: n={n_pair}")

        # Step 5: Gaussianize residuals
        Z_pair = gaussianize(raw_pair, transform_type="empirical")

        # Step 6: Fit global and adaptive LGC on original data
        print("  Fitting original LGC surfaces...")
        rho_global_orig, eff_n_global_orig, rho_adaptive_orig, _ = fit_lgc_both(
            Z_pair, grid
        )

        # Step 7: Bootstrap stability
        print(f"  Running {B} bootstrap resamples...")
        rho_global_boots, rho_adaptive_boots = bootstrap_stability(
            Z_pair, grid, B, seed=42
        )

        # Step 8: Bootstrap SD at each grid point
        sd_global = np.nanstd(rho_global_boots, axis=0, ddof=1)
        sd_adaptive = np.nanstd(rho_adaptive_boots, axis=0, ddof=1)

        # Step 9: SD ratio
        sd_ratio = sd_adaptive / (sd_global + 1e-10)

        # Step 10: Summary statistics
        valid = np.isfinite(sd_ratio)
        sparse = valid & (eff_n_global_orig < EFF_N_THRESHOLD)

        med_ratio_all = float(np.nanmedian(sd_ratio[valid])) if valid.any() else np.nan
        med_ratio_sparse = (float(np.nanmedian(sd_ratio[sparse]))
                            if sparse.any() else np.nan)
        share_adaptive_stable = (float(np.mean(sd_adaptive[valid] < sd_global[valid]))
                                 if valid.any() else np.nan)

        print(f"  Median SD ratio (all cells):         {med_ratio_all:.4f}")
        print(f"  Median SD ratio (eff_n < {EFF_N_THRESHOLD} band): {med_ratio_sparse:.4f}")
        print(f"  Share of cells where adaptive more stable: {share_adaptive_stable:.3f}")

        # Accumulate per-grid-point bootstrap records
        for j in range(M):
            boot_records.append({
                "pair": pair_label,
                "point_idx": j,
                "x1": grid[j, 0],
                "x2": grid[j, 1],
                "eff_n_orig": eff_n_global_orig[j],
                "global_rho_orig": rho_global_orig[j],
                "adaptive_rho_orig": rho_adaptive_orig[j],
                "global_sd": sd_global[j],
                "adaptive_sd": sd_adaptive[j],
                "sd_ratio": sd_ratio[j],
            })

        pair_plot_data.append({
            "pair_label": pair_label,
            "rho_global": rho_global_orig,
            "rho_adaptive": rho_adaptive_orig,
        })

        pair_sd_data.append({
            "pair_label": pair_label,
            "eff_n_orig": eff_n_global_orig,
            "sd_ratio": sd_ratio,
        })

    # -----------------------------------------------------------------------
    # Save bootstrap stability CSV
    # -----------------------------------------------------------------------
    df_boot = pd.DataFrame(boot_records)
    boot_path = os.path.join(RESULTS_DIR, "exp_8_bootstrap_stability.csv")
    df_boot.to_csv(boot_path, index=False)
    print(f"\nSaved {boot_path}")

    # -----------------------------------------------------------------------
    # Plot 1: LGC surfaces (global and adaptive) per pair
    # -----------------------------------------------------------------------
    surf_path = os.path.join(RESULTS_DIR, "exp_8_lgc_surfaces.png")
    plot_lgc_surfaces(grid, pair_plot_data, surf_path)
    print(f"Saved {surf_path}")

    # -----------------------------------------------------------------------
    # Plot 2: SD ratio vs eff_n scatter per pair
    # -----------------------------------------------------------------------
    scatter_path = os.path.join(RESULTS_DIR, "exp_8_sd_ratio.png")
    plot_sd_ratio_scatter(pair_sd_data, scatter_path)
    print(f"Saved {scatter_path}")

    print("\nExperiment 8 complete.")


if __name__ == "__main__":
    main()
