"""
Experiment 7 - Strong-Dependence Failure Diagnostics for Adaptive LGC Bandwidths.

Purpose:
    Investigate where and why the corrected adaptive bandwidth fails under strong
    dependence (tau=0.8) by measuring cap-hit frequency, comparing global vs
    adaptive ISE across sample sizes, and running an oracle-scale sweep at n=2500
    to find the ISE-optimal global scale factor c* for each copula family.

Settings:
    - Copulas: Clayton, t4, Gumbel (full mode), Clayton + t4 (FAST_MODE)
    - tau = 0.8
    - n in {500, 1000, 2500, 5000} (full) / {500, 2500} (FAST_MODE)
    - R = 25 replications (full) / 2 (FAST_MODE)
    - Symmetric NTS (NIG) marginals
    - Oracle c-scale sweep at n = 2500

Outputs:
    results/exp_7_diagnostics.csv    -- per-rep global/adaptive ISE + cap-hit freq
    results/exp_7_oracle_sweep.csv   -- mean ISE per (copula, c) at n=2500
    results/exp_7_oracle_sweep.png   -- ISE vs c line plot per copula
    results/exp_7_ise_comparison.png -- grouped bar chart: global vs adaptive ISE
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running as a script from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.copulas import make_bicop
from src.lgc import (
    lgc_surface,
    global_bandwidth,
    make_eval_grid,
    density_weighted_ise,
)
from src.transforms import gaussianize
from src.bandwidth import (
    corrected_adaptive_bandwidth,
    lgc_surface_adaptive,
    pilot_density,
    estimate_beta_field,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST_MODE = True

TAU = 0.8
N_SWEEP = 2500          # sample size for oracle sweep
N_GT = 20_000           # ground-truth auxiliary sample size
N_SIDE = 11             # regular grid side (make_eval_grid default)
BASE_SEED = 42

if FAST_MODE:
    COPULAS = ["clayton", "t4"]
    N_LIST = [500, 2500]
    R = 2
    C_STEP = 0.5
else:
    COPULAS = ["clayton", "t4", "gumbel"]
    N_LIST = [500, 1000, 2500, 5000]
    R = 25
    C_STEP = 0.1

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fast simulation helpers
# ---------------------------------------------------------------------------

def _simulate_gaussianize(
    copula: str,
    tau: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n bivariate observations from a copula and empirically Gaussianize.

    Empirical Gaussianization is rank-invariant: Phi^{-1}((rank - 0.5)/n) is
    identical whether applied to copula uniforms U or to F^{-1}(U) for any
    strictly-increasing marginal CDF F.  This avoids the slow NIG ppf call while
    preserving the intended copula dependence structure used by the experiment.

    Parameters
    ----------
    copula : str
        Copula family name.
    tau : float
        Kendall's tau.
    n : int
        Sample size.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Empirically Gaussianized bivariate sample.
    """
    cop = make_bicop(copula, tau)
    seed_int = int(rng.integers(0, 2**31))
    U = cop.simulate(n=n, seeds=[seed_int, seed_int + 1])
    return gaussianize(U, transform_type="empirical")


def _build_truth_surface(
    copula: str,
    tau: float,
    grid: np.ndarray,
    n_gt: int = N_GT,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """
    Build a ground-truth LGC surface using a large copula sample.

    Uses the same rank-invariant approach as _simulate_gaussianize.

    Parameters
    ----------
    copula : str
        Copula family name.
    tau : float
        Kendall's tau.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    n_gt : int
        Auxiliary sample size.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (M,)
        Ground-truth LGC surface.
    """
    rng = np.random.default_rng(seed)
    Z_gt = _simulate_gaussianize(copula, tau, n_gt, rng)
    b_gt = global_bandwidth(n_gt)
    rho_gt, _ = lgc_surface(Z_gt, grid, b_gt)
    return rho_gt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_adaptive_shape(
    Z: np.ndarray,
    grid: np.ndarray,
    n_side: int = N_SIDE,
) -> tuple[np.ndarray, float]:
    """
    Compute the un-normalized adaptive shape s(x) and global bandwidth b0.

    Mirrors the internal shape_raw computation inside
    corrected_adaptive_bandwidth() before the budget-neutral normalization step.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 2)
        Gaussianized data.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    n_side : int
        Side length of the regular grid sub-block.

    Returns
    -------
    shape_raw : np.ndarray, shape (M,)
        Un-normalized AMISE shape s(x).
    b0 : float
        Global plug-in bandwidth.
    """
    n = Z.shape[0]
    b0 = global_bandwidth(n)

    rho_pilot, _ = lgc_surface(Z, grid, b0)
    f_pilot = pilot_density(Z, grid, b0)
    beta_tilde = estimate_beta_field(rho_pilot, f_pilot, grid, n_side=n_side)

    rho2 = np.where(np.isfinite(rho_pilot), rho_pilot ** 2, 0.0)
    numerator = (1.0 - rho2) ** 2
    denom = np.abs(f_pilot + 1e-12) * (beta_tilde ** 2 + 1e-12)
    shape_raw = (numerator / (denom + 1e-12)) ** (1.0 / 6.0)
    return shape_raw, b0


def run_diag_rep(
    copula: str,
    n: int,
    rng: np.random.Generator,
    grid: np.ndarray,
    rho_gt: np.ndarray,
) -> dict:
    """
    One diagnostic replication: global ISE, adaptive ISE, cap-hit frequency.

    Parameters
    ----------
    copula : str
        Copula family name.
    n : int
        Sample size.
    rng : np.random.Generator
        RNG for this replication.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    rho_gt : np.ndarray, shape (M,)
        Ground-truth LGC surface.

    Returns
    -------
    dict with keys: global_ise, adaptive_ise, cap_hit_freq
    """
    Z = _simulate_gaussianize(copula, TAU, n, rng)

    b0 = global_bandwidth(n)

    # Global LGC
    rho_global, eff_n_global = lgc_surface(Z, grid, b0)
    ise_global = density_weighted_ise(rho_global, rho_gt, eff_n_global)

    # Corrected adaptive LGC
    b_adaptive = corrected_adaptive_bandwidth(Z, grid)
    rho_adaptive, eff_n_adaptive = lgc_surface_adaptive(Z, grid, b_adaptive)
    ise_adaptive = density_weighted_ise(rho_adaptive, rho_gt, eff_n_adaptive)

    # Upper-cap hit frequency: proportion of grid points clipped to 3*b0
    b_hi = 3.0 * b0
    cap_hit_freq = float(np.mean(b_adaptive >= b_hi - 1e-10))

    return {
        "global_ise": ise_global,
        "adaptive_ise": ise_adaptive,
        "cap_hit_freq": cap_hit_freq,
    }


def oracle_sweep_rep(
    copula: str,
    n: int,
    rng: np.random.Generator,
    grid: np.ndarray,
    rho_gt: np.ndarray,
    c_values: np.ndarray,
) -> np.ndarray:
    """
    One oracle-sweep replication: ISE for each scale c.

    For each c, sets b_c(x) = clip(c * s(x), 0.5*b0, 3*b0) where s(x) is the
    un-normalized adaptive shape, then evaluates ISE.

    Parameters
    ----------
    copula : str
        Copula family name.
    n : int
        Sample size (should be N_SWEEP = 2500).
    rng : np.random.Generator
        RNG for this replication.
    grid : np.ndarray, shape (M, 2)
        Evaluation grid.
    rho_gt : np.ndarray, shape (M,)
        Ground-truth LGC surface.
    c_values : np.ndarray
        Scale factors to evaluate.

    Returns
    -------
    np.ndarray, shape (len(c_values),)
        ISE for each scale factor c.
    """
    Z = _simulate_gaussianize(copula, TAU, n, rng)

    shape_raw, b0 = compute_adaptive_shape(Z, grid)
    b_lo = 0.5 * b0
    b_hi = 3.0 * b0

    ise_per_c = np.full(len(c_values), np.nan)
    for i, c in enumerate(c_values):
        b_c = np.clip(c * shape_raw, b_lo, b_hi)
        b_c[~np.isfinite(b_c)] = b0
        rho_c, eff_n_c = lgc_surface_adaptive(Z, grid, b_c)
        ise_per_c[i] = density_weighted_ise(rho_c, rho_gt, eff_n_c)

    return ise_per_c


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Experiment 7 — FAST_MODE={FAST_MODE}")
    print(f"  Copulas: {COPULAS}, n_list: {N_LIST}, R={R}, c_step={C_STEP}")

    grid = make_eval_grid()

    # Ground-truth surfaces cached per copula (independent of sample size n)
    rho_gt_cache: dict[str, np.ndarray] = {}

    # -----------------------------------------------------------------------
    # Part 1: Diagnostics — global vs adaptive ISE + cap-hit frequency
    # -----------------------------------------------------------------------
    print("\n--- Part 1: Diagnostics ---")
    diag_records = []

    for copula in COPULAS:
        # Build ground truth once per copula
        if copula not in rho_gt_cache:
            print(f"  Building ground truth: {copula} tau={TAU}")
            rho_gt_cache[copula] = _build_truth_surface(
                copula, TAU, grid, n_gt=N_GT, seed=BASE_SEED
            )
        rho_gt = rho_gt_cache[copula]

        for n in N_LIST:
            print(f"  {copula} n={n:5d}", end="", flush=True)
            for r in range(R):
                rng = np.random.default_rng(BASE_SEED + r * 1000 + hash(copula) % 997)
                result = run_diag_rep(copula, n, rng, grid, rho_gt)
                diag_records.append({
                    "copula": copula,
                    "n": n,
                    "rep": r,
                    "global_ise": result["global_ise"],
                    "adaptive_ise": result["adaptive_ise"],
                    "cap_hit_freq": result["cap_hit_freq"],
                })
                print(".", end="", flush=True)
            print()

    df_diag = pd.DataFrame(diag_records)
    diag_path = os.path.join(RESULTS_DIR, "exp_7_diagnostics.csv")
    df_diag.to_csv(diag_path, index=False)
    print(f"\nSaved {diag_path}")

    # Summary table
    diag_summary = (
        df_diag.groupby(["copula", "n"])
        .agg(
            mean_global_ise=("global_ise", "mean"),
            mean_adaptive_ise=("adaptive_ise", "mean"),
            mean_cap_hit=("cap_hit_freq", "mean"),
        )
        .reset_index()
    )
    print("\nDiagnostics summary:")
    print(diag_summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # Part 2: Oracle-scale sweep at n=2500
    # -----------------------------------------------------------------------
    print("\n--- Part 2: Oracle-scale sweep (n=2500) ---")
    c_values = np.arange(0.3, 3.0, C_STEP)

    sweep_records = []
    optimal_c: dict[str, float] = {}

    for copula in COPULAS:
        rho_gt = rho_gt_cache[copula]
        print(f"  Oracle sweep: {copula}", end="", flush=True)

        # R replications at N_SWEEP
        ise_matrix = np.full((R, len(c_values)), np.nan)
        for r in range(R):
            rng = np.random.default_rng(BASE_SEED + r * 1000 + hash(copula) % 997)
            ise_matrix[r] = oracle_sweep_rep(
                copula, N_SWEEP, rng, grid, rho_gt, c_values
            )
            print(".", end="", flush=True)
        print()

        mean_ise = np.nanmean(ise_matrix, axis=0)
        opt_idx = int(np.nanargmin(mean_ise))
        optimal_c[copula] = float(c_values[opt_idx])

        for i, c in enumerate(c_values):
            sweep_records.append({
                "copula": copula,
                "c": round(float(c), 6),
                "mean_ise": mean_ise[i],
            })

        print(f"    Optimal c* = {optimal_c[copula]:.2f}, "
              f"mean ISE = {mean_ise[opt_idx]:.6f}")

    df_sweep = pd.DataFrame(sweep_records)
    sweep_path = os.path.join(RESULTS_DIR, "exp_7_oracle_sweep.csv")
    df_sweep.to_csv(sweep_path, index=False)
    print(f"\nSaved {sweep_path}")

    # -----------------------------------------------------------------------
    # Plot 1: Oracle sweep ISE vs c
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    for i, copula in enumerate(COPULAS):
        sub = df_sweep[df_sweep["copula"] == copula].sort_values("c")
        ax.plot(sub["c"], sub["mean_ise"],
                marker=markers[i % len(markers)], label=copula, linewidth=1.8)
        c_opt = optimal_c[copula]
        ise_opt = sub.loc[sub["c"].sub(c_opt).abs().idxmin(), "mean_ise"]
        ax.axvline(c_opt, color=f"C{i}", linestyle="--", alpha=0.5)
        ax.annotate(f"c*={c_opt:.2f}", xy=(c_opt, ise_opt),
                    xytext=(c_opt + 0.05, ise_opt),
                    fontsize=8, color=f"C{i}")

    ax.set_xlabel("Scale factor c", fontsize=12)
    ax.set_ylabel("Mean ISE", fontsize=12)
    ax.set_title(f"Oracle Sweep: Mean ISE vs c  (n={N_SWEEP}, tau={TAU})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sweep_plot_path = os.path.join(RESULTS_DIR, "exp_7_oracle_sweep.png")
    fig.savefig(sweep_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {sweep_plot_path}")

    # -----------------------------------------------------------------------
    # Plot 2: Bar chart — global vs adaptive ISE at tau=0.8
    # -----------------------------------------------------------------------
    agg = (
        df_diag.groupby(["copula", "n"])
        .agg(
            mean_global=("global_ise", "mean"),
            sd_global=("global_ise", "std"),
            mean_adaptive=("adaptive_ise", "mean"),
            sd_adaptive=("adaptive_ise", "std"),
        )
        .reset_index()
    )

    n_copulas = len(COPULAS)
    fig, axes = plt.subplots(
        1, n_copulas, figsize=(4.5 * n_copulas, 5), sharey=False
    )
    if n_copulas == 1:
        axes = [axes]

    x = np.arange(len(N_LIST))
    width = 0.35

    for ax, copula in zip(axes, COPULAS):
        sub = agg[agg["copula"] == copula].sort_values("n")
        n_labels = [str(ni) for ni in sub["n"].values]

        bars_g = ax.bar(
            x - width / 2, sub["mean_global"].values, width,
            label="Global", color="steelblue", alpha=0.85,
            yerr=sub["sd_global"].fillna(0).values, capsize=4,
        )
        bars_a = ax.bar(
            x + width / 2, sub["mean_adaptive"].values, width,
            label="Adaptive", color="coral", alpha=0.85,
            yerr=sub["sd_adaptive"].fillna(0).values, capsize=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(n_labels)
        ax.set_xlabel("Sample size n", fontsize=11)
        ax.set_ylabel("Mean ISE", fontsize=11)
        ax.set_title(f"{copula}  (tau={TAU})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Global vs Adaptive LGC: Mean ISE at tau=0.8", fontsize=13, y=1.01)
    fig.tight_layout()
    ise_plot_path = os.path.join(RESULTS_DIR, "exp_7_ise_comparison.png")
    fig.savefig(ise_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ise_plot_path}")

    print("\nExperiment 7 complete.")


if __name__ == "__main__":
    main()
