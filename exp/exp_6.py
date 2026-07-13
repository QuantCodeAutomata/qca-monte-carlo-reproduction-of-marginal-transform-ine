"""
Experiment 6 – Dependence-Regime Map for Adaptive LGC Bandwidths.

Settings:
- 4 copulas, tau in {0.2, 0.5, 0.8}, symmetric NTS marginals
- n in {500, 1000, 2500, 5000}
- R=50 for tau=0.5, R=25 for tau in {0.2, 0.8}
- 2 estimators: global plug-in, corrected adaptive
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

from src.bandwidth import corrected_adaptive_bandwidth, lgc_surface_adaptive
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
    TAU_VALUES = [0.2, 0.5]
    N_VALUES = [500, 1000]
    R_BY_TAU: dict[float, int] = {0.2: 2, 0.5: 2, 0.8: 2}
else:
    FAMILIES = ALL_FAMILIES           # gaussian, clayton, t4, gumbel
    TAU_VALUES = [0.2, 0.5, 0.8]
    N_VALUES = [500, 1000, 2500, 5000]
    R_BY_TAU = {0.2: 25, 0.5: 50, 0.8: 25}

N_GT = 20_000
BASE_SEED = 6000
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
        for tau_idx, tau in enumerate(TAU_VALUES):
            R = R_BY_TAU[tau]
            print(f"\n[Exp 6] family={family}  tau={tau}  R={R}")

            truth_seed = BASE_SEED + hash(family) % 10_000 + int(tau * 1000)
            rho_gt = _make_truth_surface(family, tau, grid, N_GT, seed=truth_seed)
            print(f"  Ground-truth surface built (n_gt={N_GT})")

            for n_idx, n in enumerate(N_VALUES):
                b0 = global_bandwidth(n)
                print(f"  n={n}  b0={b0:.4f}  {R} reps ...")

                for r in range(R):
                    rep_seed = (
                        BASE_SEED
                        + fam_idx * 1_000_000
                        + tau_idx * 100_000
                        + n_idx * 10_000
                        + r * 100
                    )
                    rng = np.random.default_rng(rep_seed)
                    Z = _simulate_z(family, tau, n, rng)

                    # Global plug-in
                    rho_g, eff_g = lgc_surface(Z, grid, b0)
                    raw_records.append({
                        "family": family, "tau": tau, "n": n, "replication": r,
                        "estimator": "global",
                        "ISE": density_weighted_ise(rho_g, rho_gt, eff_g),
                    })

                    # Corrected AMISE adaptive
                    b_corr = corrected_adaptive_bandwidth(Z, grid)
                    rho_c, eff_c = lgc_surface_adaptive(Z, grid, b_corr)
                    raw_records.append({
                        "family": family, "tau": tau, "n": n, "replication": r,
                        "estimator": "corrected",
                        "ISE": density_weighted_ise(rho_c, rho_gt, eff_c),
                    })

    return pd.DataFrame(raw_records)


# ---------------------------------------------------------------------------
# Regime map construction
# ---------------------------------------------------------------------------

def build_regime_map(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the dependence-regime map table.

    Returns
    -------
    regime_map : pd.DataFrame
        Pivoted table – rows=(family, tau), columns=n, cells=% ISE change.
    regime_df : pd.DataFrame
        Long-form version with one row per (family, tau, n).
    """
    # Aggregate mean ISE per (family, tau, n, estimator)
    summary_rows: list[dict] = []
    for (family, tau, n, estimator), grp in raw_df.groupby(
        ["family", "tau", "n", "estimator"]
    ):
        stats = aggregate_ise(grp["ISE"].values)
        summary_rows.append({
            "family": family, "tau": tau, "n": n, "estimator": estimator,
            "mean_ise": stats["mean_ise"], "sd_ise": stats["sd_ise"],
        })
    summary = pd.DataFrame(summary_rows)

    # % change: (corrected_mean - global_mean) / global_mean * 100
    global_idx = summary[summary["estimator"] == "global"].set_index(
        ["family", "tau", "n"]
    )["mean_ise"]
    corr_idx = summary[summary["estimator"] == "corrected"].set_index(
        ["family", "tau", "n"]
    )["mean_ise"]

    regime_rows: list[dict] = []
    for key in global_idx.index:
        g = global_idx.loc[key]
        c = corr_idx.loc[key] if key in corr_idx.index else np.nan
        pct = (c - g) / g * 100.0 if np.isfinite(g) and g != 0 else np.nan
        family, tau, n = key
        regime_rows.append({"family": family, "tau": tau, "n": n, "pct_change": pct})

    regime_df = pd.DataFrame(regime_rows)

    # Pivot: rows=(family, tau), columns=n
    regime_map = regime_df.pivot_table(
        index=["family", "tau"], columns="n", values="pct_change"
    )
    regime_map.columns.name = "n"

    return regime_map, regime_df


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def make_heatmap(regime_map: pd.DataFrame) -> None:
    """
    Heatmap of % ISE change (corrected adaptive vs global).

    Green = adaptive reduces ISE (negative % change).
    Red   = adaptive increases ISE (positive % change).
    """
    data = regime_map.values.astype(float)
    row_labels = [f"{fam}\nτ={tau}" for fam, tau in regime_map.index]
    col_labels = [str(int(c)) for c in regime_map.columns]

    vmax = float(np.nanmax(np.abs(data)))
    vmax = max(vmax, 1.0)

    fig_h = max(4, len(row_labels) * 0.65)
    fig_w = max(6, len(col_labels) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # RdYlGn_r: negative values → green (adaptive better), positive → red (worse)
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Sample size n")
    ax.set_title("Exp 6 — % ISE Change: Corrected Adaptive vs Global\n"
                 "(green = adaptive better, red = adaptive worse)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="% ISE Change (corrected vs global)")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "exp_6_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 6 — Dependence-Regime Map for Adaptive LGC")
    print(f"  FAST_MODE={FAST_MODE}")
    print(f"  families={FAMILIES}  tau={TAU_VALUES}  n={N_VALUES}")
    print(f"  R_BY_TAU={R_BY_TAU}")
    print("=" * 60)

    raw_df = run_experiment()

    raw_path = os.path.join(RESULTS_DIR, "exp_6_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw results  → {raw_path}")

    regime_map, regime_df = build_regime_map(raw_df)

    regime_map_path = os.path.join(RESULTS_DIR, "exp_6_regime_map.csv")
    regime_map.to_csv(regime_map_path)
    print(f"Regime map   → {regime_map_path}")

    print("\n--- Regime Map (% ISE Change: Corrected vs Global) ---")
    print(regime_map.to_string())

    make_heatmap(regime_map)
    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
