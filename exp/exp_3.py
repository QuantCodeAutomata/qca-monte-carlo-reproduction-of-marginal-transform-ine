"""
Experiment 3 – Scarcity Diagnostic for Tail LGC Estimation Error.

For each (copula, n) cell the empirical-transform LGC is estimated over R
replications.  Pointwise bias, SD, RMSE, and Fisher-normalized SD are
computed and then pooled across all cells to fit the log-log scarcity
regression:

  log[ SD / (1 - rho^2) ]  ~  a + b * log(eff_n)

Outputs
-------
results/exp_3_diagnostics.csv    : per-point diagnostics for every (family, n)
results/exp_3_regression.csv     : slope, intercept, R^2, n_obs from OLS fit
results/exp_3_scarcity_plot.png  : scatter of log(eff_n) vs log(Fisher SD)
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

from src.nts import symmetric_nts
from src.lgc import make_eval_grid
from src.simulation import run_mc_cell
from src.metrics import pointwise_diagnostics, scarcity_regression
from exp.utils import ensure_results_dir, save_results, RESULTS_DIR


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

_FULL = dict(
    R=50,
    NS=[500, 1000, 2500, 5000],
    COPULAS=["gaussian", "clayton", "t4", "gumbel"],
)

_FAST = dict(
    R=3,
    NS=[500, 1000],
    COPULAS=["gaussian", "clayton"],
)

TAU = 0.5
N_GT = 20_000
TRANSFORM = "empirical"
BASE_SEED = 42
VERBOSE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot_cell(df_emp: pd.DataFrame, R: int, M: int):
    """
    Extract (R, M) rho_hat / eff_n arrays and (M,) rho_gt from per-rep DataFrame.

    Parameters
    ----------
    df_emp : pd.DataFrame
        Rows from run_mc_cell filtered to a single transform.
    R : int
        Number of replications.
    M : int
        Number of grid points.

    Returns
    -------
    rho_hat_reps : np.ndarray, shape (R, M)
    eff_n_reps   : np.ndarray, shape (R, M)
    rho_gt_arr   : np.ndarray, shape (M,)
    """
    rho_hat_reps = (
        df_emp.pivot_table(
            index="replication", columns="point_idx", values="rho_hat", aggfunc="first"
        )
        .reindex(index=range(R), columns=range(M))
        .to_numpy()
    )
    eff_n_reps = (
        df_emp.pivot_table(
            index="replication", columns="point_idx", values="eff_n", aggfunc="first"
        )
        .reindex(index=range(R), columns=range(M))
        .to_numpy()
    )
    rho_gt_arr = (
        df_emp.groupby("point_idx")["rho_gt"]
        .first()
        .reindex(range(M))
        .to_numpy()
    )
    return rho_hat_reps, eff_n_reps, rho_gt_arr


def _run_all_cells(copulas, ns, R, marginal, grid) -> pd.DataFrame:
    """
    Run MC for every (family, n) cell and return pooled pointwise diagnostics.

    Parameters
    ----------
    copulas : list of str
    ns      : list of int
    R       : int
    marginal : NIGDistribution
    grid    : np.ndarray, shape (M, 2)

    Returns
    -------
    pd.DataFrame
        Concatenated pointwise diagnostics across all cells.
    """
    M = len(grid)
    frames = []
    total = len(copulas) * len(ns)
    idx = 0

    for family in copulas:
        for n in ns:
            idx += 1
            print(f"\n[{idx}/{total}] family={family}  n={n}  R={R}")
            df_cell = run_mc_cell(
                family=family,
                tau=TAU,
                n=n,
                n_reps=R,
                marginal=marginal,
                grid=grid,
                transforms=[TRANSFORM],
                n_gt=N_GT,
                base_seed=BASE_SEED,
                verbose=VERBOSE,
            )

            df_emp = df_cell[df_cell["transform"] == TRANSFORM]
            rho_hat_reps, eff_n_reps, rho_gt_arr = _pivot_cell(df_emp, R, M)

            diag = pointwise_diagnostics(rho_hat_reps, rho_gt_arr, eff_n_reps)
            diag["family"] = family
            diag["n"] = n
            diag["point_idx"] = np.arange(M)
            diag["x1"] = grid[:, 0]
            diag["x2"] = grid[:, 1]
            diag["rho_gt"] = rho_gt_arr
            frames.append(diag)

    return pd.concat(frames, ignore_index=True)


def _fit_scarcity_regression(df_all: pd.DataFrame) -> dict:
    """
    Fit log-log scarcity regression pooled across all cells.

    Parameters
    ----------
    df_all : pd.DataFrame
        Pooled per-point diagnostics containing 'sd', 'rho_gt', 'mean_eff_n'.

    Returns
    -------
    dict with keys: slope, intercept, r_squared, n_obs
    """
    return scarcity_regression(
        sd_vals=df_all["sd"].to_numpy(),
        rho_gt_vals=df_all["rho_gt"].to_numpy(),
        eff_n_vals=df_all["mean_eff_n"].to_numpy(),
    )


def _plot_scarcity(df_all: pd.DataFrame, reg: dict, out_path: pathlib.Path) -> None:
    """
    Scatter plot of log(eff_n) vs log(Fisher-normalized SD) with OLS fit line.

    Parameters
    ----------
    df_all   : pd.DataFrame  Pooled pointwise diagnostics.
    reg      : dict          Regression results (slope, intercept, r_squared).
    out_path : pathlib.Path  Output PNG path.
    """
    fisher_sd = df_all["sd"] / np.maximum(1.0 - df_all["rho_gt"] ** 2, 1e-6)
    plot_mask = (
        (df_all["mean_eff_n"] >= 5)
        & np.isfinite(df_all["sd"])
        & np.isfinite(df_all["rho_gt"])
        & (df_all["sd"] > 0)
        & (fisher_sd > 0)
    )

    log_eff_n = np.log(df_all.loc[plot_mask, "mean_eff_n"].to_numpy())
    log_fisher_sd = np.log(fisher_sd[plot_mask].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        log_eff_n,
        log_fisher_sd,
        alpha=0.25,
        s=8,
        c="steelblue",
        label="Grid points",
    )

    if np.isfinite(reg.get("slope", np.nan)) and np.isfinite(reg.get("intercept", np.nan)):
        x_line = np.linspace(log_eff_n.min(), log_eff_n.max(), 300)
        y_line = reg["intercept"] + reg["slope"] * x_line
        ax.plot(
            x_line,
            y_line,
            "r-",
            lw=2,
            label=(
                f"OLS fit: slope={reg['slope']:.3f}, "
                f"R²={reg['r_squared']:.3f} "
                f"(n={reg['n_obs']})"
            ),
        )

    ax.set_xlabel("log(eff_n)", fontsize=12)
    ax.set_ylabel("log(Fisher-normalized SD)", fontsize=12)
    ax.set_title(
        "Exp 3 — Scarcity Regression: Fisher SD vs Effective Sample Size",
        fontsize=11,
        pad=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

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
    NS = settings["NS"]
    COPULAS = settings["COPULAS"]

    mode_label = "FAST" if FAST_MODE else "FULL"
    print("=" * 60)
    print(f"Experiment 3 — Scarcity Diagnostic  [{mode_label} MODE]")
    print(f"  copulas  : {COPULAS}")
    print(f"  n values : {NS}")
    print(f"  transform: {TRANSFORM}")
    print(f"  R        : {R}  |  tau={TAU}  |  n_gt={N_GT}")
    print("=" * 60)

    marginal = symmetric_nts()
    grid = make_eval_grid()

    # Step 1–2: run MC cells and compute pointwise diagnostics
    df_all = _run_all_cells(COPULAS, NS, R, marginal, grid)

    # Step 7: save per-point diagnostics
    ensure_results_dir()
    save_results(df_all, "exp_3_diagnostics.csv")

    print(f"\nDiagnostics: {len(df_all)} rows "
          f"({len(df_all['family'].unique())} families × "
          f"{len(df_all['n'].unique())} n values × "
          f"{len(grid)} grid points)")

    # Steps 4–5: fit pooled scarcity regression
    reg = _fit_scarcity_regression(df_all)
    print(
        f"\n--- Scarcity regression (pooled) ---\n"
        f"  slope     : {reg['slope']:.4f}\n"
        f"  intercept : {reg['intercept']:.4f}\n"
        f"  R²        : {reg['r_squared']:.4f}\n"
        f"  n_obs     : {reg['n_obs']}"
    )

    save_results(pd.DataFrame([reg]), "exp_3_regression.csv")

    # Step 6: scatter plot
    _plot_scarcity(df_all, reg, RESULTS_DIR / "exp_3_scarcity_plot.png")


if __name__ == "__main__":
    main()
