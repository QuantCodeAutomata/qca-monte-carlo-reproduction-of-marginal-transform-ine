"""
Experiment 5 – Adaptive AMISE vs Global vs Heuristic Bandwidth at Moderate Dependence.

Settings:
- 4 copulas, tau=0.5, symmetric NTS marginals
- n in {500, 1000, 2500, 5000}, R=50 replications
- 3 estimators: global plug-in, corrected adaptive, heuristic adaptive
- Paired design: same sample used for all estimators per replication
- n_gt=20000 for ground-truth LGC surface
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bandwidth import (
    corrected_adaptive_bandwidth,
    heuristic_adaptive_bandwidth,
    lgc_surface_adaptive,
)
from src.copulas import ALL_FAMILIES, make_bicop
from src.lgc import density_weighted_ise, global_bandwidth, lgc_surface, make_eval_grid
from src.metrics import aggregate_ise
from src.transforms import empirical_transform

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST_MODE = True

if FAST_MODE:
    FAMILIES = ALL_FAMILIES[:2]       # gaussian, clayton
    N_VALUES = [500, 1000]
    R = 2
    ESTIMATORS = ["global", "corrected"]
else:
    FAMILIES = ALL_FAMILIES           # gaussian, clayton, t4, gumbel
    N_VALUES = [500, 1000, 2500, 5000]
    R = 50
    ESTIMATORS = ["global", "corrected", "heuristic"]

TAU = 0.5
N_GT = 20_000
BASE_SEED = 5000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fast simulation helpers
# ---------------------------------------------------------------------------
# The empirical transform uses only ranks, so applying it to the raw copula
# uniform samples U gives exactly the same Gaussianized matrix Z as first
# transforming U via the marginal ppf (which is a monotone, rank-preserving
# map) and then applying the empirical transform. This bypasses the slow
# scipy norminvgauss ppf, which uses element-wise numerical root-finding.

def _make_truth_surface(
    family: str, tau: float, grid: np.ndarray, n_gt: int, seed: int
) -> np.ndarray:
    """Build ground-truth LGC surface from a large copula sample."""
    rng = np.random.default_rng(seed)
    cop = make_bicop(family, tau)
    seed_int = int(rng.integers(0, 2**31))
    U = cop.simulate(n=n_gt, seeds=[seed_int, seed_int + 1])
    Z = empirical_transform(U)
    b_gt = global_bandwidth(n_gt)
    rho_gt, _ = lgc_surface(Z, grid, b_gt)
    return rho_gt


def _simulate_z(family: str, tau: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a copula sample of size n and Gaussianize via empirical transform."""
    cop = make_bicop(family, tau)
    seed_int = int(rng.integers(0, 2**31))
    U = cop.simulate(n=n, seeds=[seed_int, seed_int + 1])
    return empirical_transform(U)


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    """Run the paired Monte Carlo experiment and return raw per-replication ISEs."""
    grid = make_eval_grid()
    raw_records: list[dict] = []

    for fam_idx, family in enumerate(FAMILIES):
        print(f"\n[Exp 5] family={family}")

        truth_seed = BASE_SEED + hash(family) % 10_000
        rho_gt = _make_truth_surface(family, TAU, grid, N_GT, seed=truth_seed)
        print(f"  Ground-truth surface built (n_gt={N_GT})")

        for n_idx, n in enumerate(N_VALUES):
            b0 = global_bandwidth(n)
            print(f"  n={n}  b0={b0:.4f}  {R} reps ...")

            for r in range(R):
                rep_seed = BASE_SEED + fam_idx * 100_000 + n_idx * 10_000 + r * 100
                rng = np.random.default_rng(rep_seed)
                Z = _simulate_z(family, TAU, n, rng)

                # Global plug-in
                rho_g, eff_g = lgc_surface(Z, grid, b0)
                raw_records.append({
                    "family": family, "n": n, "replication": r,
                    "estimator": "global",
                    "ISE": density_weighted_ise(rho_g, rho_gt, eff_g),
                })

                # Corrected AMISE adaptive
                b_corr = corrected_adaptive_bandwidth(Z, grid)
                rho_c, eff_c = lgc_surface_adaptive(Z, grid, b_corr)
                raw_records.append({
                    "family": family, "n": n, "replication": r,
                    "estimator": "corrected",
                    "ISE": density_weighted_ise(rho_c, rho_gt, eff_c),
                })

                # Heuristic adaptive (full mode only)
                if "heuristic" in ESTIMATORS:
                    b_heur = heuristic_adaptive_bandwidth(Z, grid)
                    rho_h, eff_h = lgc_surface_adaptive(Z, grid, b_heur)
                    raw_records.append({
                        "family": family, "n": n, "replication": r,
                        "estimator": "heuristic",
                        "ISE": density_weighted_ise(rho_h, rho_gt, eff_h),
                    })

    return pd.DataFrame(raw_records)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/sd ISE per (copula, n, estimator) and compute % change vs global."""
    rows: list[dict] = []
    for (family, n, estimator), grp in raw_df.groupby(["family", "n", "estimator"]):
        stats = aggregate_ise(grp["ISE"].values)
        rows.append({
            "family": family, "n": n, "estimator": estimator,
            "mean_ise": stats["mean_ise"], "sd_ise": stats["sd_ise"],
        })
    summary = pd.DataFrame(rows)

    # % change: (adaptive_mean - global_mean) / global_mean * 100
    global_mean = (
        summary[summary["estimator"] == "global"]
        .set_index(["family", "n"])["mean_ise"]
    )
    pct_col = []
    for _, row in summary.iterrows():
        key = (row["family"], row["n"])
        if row["estimator"] != "global" and key in global_mean.index:
            g = global_mean.loc[key]
            pct_col.append((row["mean_ise"] - g) / g * 100.0)
        else:
            pct_col.append(np.nan)
    summary["pct_change_vs_global"] = pct_col

    return summary


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def make_bar_chart(summary: pd.DataFrame) -> None:
    """Bar chart of mean ISE by estimator and copula at n=2500 (or nearest)."""
    available_n = sorted(summary["n"].unique())
    plot_n = 2500 if 2500 in available_n else available_n[-1]

    df_plot = summary[summary["n"] == plot_n].copy()
    if df_plot.empty:
        print("  [Warning] No data for bar chart; skipping.")
        return

    families = sorted(df_plot["family"].unique())
    estimators = sorted(df_plot["estimator"].unique())
    x = np.arange(len(families))
    width = 0.8 / len(estimators)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, est in enumerate(estimators):
        vals = []
        for fam in families:
            mask = (df_plot["family"] == fam) & (df_plot["estimator"] == est)
            vals.append(df_plot.loc[mask, "mean_ise"].values[0] if mask.any() else np.nan)
        offset = (i - (len(estimators) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=est)

    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_xlabel("Copula Family")
    ax.set_ylabel("Mean ISE")
    ax.set_title(f"Exp 5 — Mean ISE by Estimator / Copula  (tau={TAU}, n={plot_n})")
    ax.legend(title="Estimator")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "exp_5_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 5 — Adaptive vs Global vs Heuristic Bandwidth")
    print(f"  FAST_MODE={FAST_MODE}")
    print(f"  families={FAMILIES}  n={N_VALUES}  R={R}")
    print(f"  estimators={ESTIMATORS}  tau={TAU}")
    print("=" * 60)

    raw_df = run_experiment()

    raw_path = os.path.join(RESULTS_DIR, "exp_5_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw results  → {raw_path}")

    summary = build_summary(raw_df)
    summary_path = os.path.join(RESULTS_DIR, "exp_5_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary      → {summary_path}")

    print("\n--- Summary ---")
    print(summary.to_string(index=False))

    make_bar_chart(summary)
    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
