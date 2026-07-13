"""
Experiment 1 – Monte Carlo Reproduction of Marginal-Transform Inertness for LGC.

Runs a full factorial MC study over copula families × sample sizes × transform
types using a symmetric NTS (NIG) marginal.  For each (family, n) cell the
same raw sample is gaussianized by all three transforms (paired design).

Outputs
-------
results/exp_1_raw.csv     : one row per (family, tau, n, replication, transform)
results/exp_1_summary.csv : mean / sd ISE aggregated by (family, tau, n, transform)
results/exp_1_ise_table.png : table-style heatmap of mean ISE
"""

# ---------------------------------------------------------------------------
# FAST_MODE flag — set to False for the full experiment
# ---------------------------------------------------------------------------
FAST_MODE = True

import pathlib
import sys

# Ensure project root is on sys.path so 'src.*' imports work from any cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.nts import symmetric_nts
from src.lgc import make_eval_grid
from src.simulation import run_mc_cell
from src.metrics import aggregate_ise
from exp.utils import ensure_results_dir, save_results, RESULTS_DIR


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

_FULL = dict(
    R=50,
    NS=[500, 1000, 2500, 5000],
    COPULAS=["gaussian", "clayton", "t4", "gumbel"],
    TRANSFORMS=["empirical", "oracle", "fitted"],
)

_FAST = dict(
    R=3,
    NS=[500, 1000],
    COPULAS=["gaussian", "clayton"],
    TRANSFORMS=["empirical", "oracle"],
)

TAU = 0.5
N_GT = 20_000
BASE_SEED = 42
VERBOSE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_all_cells(copulas, ns, transforms, marginal, grid):
    """Iterate over (family, n) cells and concatenate full MC output."""
    frames = []
    total = len(copulas) * len(ns)
    idx = 0
    settings = _FAST if FAST_MODE else _FULL
    R = settings["R"]

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
                transforms=transforms,
                n_gt=N_GT,
                base_seed=BASE_SEED,
                verbose=VERBOSE,
            )
            frames.append(df_cell)

    return pd.concat(frames, ignore_index=True)


def _extract_raw_ise(df_all: pd.DataFrame) -> pd.DataFrame:
    """Collapse full per-point output to one ISE row per replication × transform."""
    group_cols = ["family", "tau", "n", "replication", "transform"]
    return (
        df_all
        .groupby(group_cols, sort=False)["ISE"]
        .first()
        .reset_index()
    )


def _build_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ISE statistics grouped by (family, tau, n, transform)."""
    group_cols = ["family", "tau", "n", "transform"]
    records = []
    for keys, grp in df_raw.groupby(group_cols, sort=False):
        agg = aggregate_ise(grp["ISE"].values)
        records.append(dict(zip(group_cols, keys), **agg))
    return pd.DataFrame(records)


def _plot_heatmap(df_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Save table-style heatmap of mean ISE by (copula × n) and transform."""
    col_order = [t for t in ["empirical", "oracle", "fitted"]
                 if t in df_summary["transform"].unique()]

    pivot = df_summary.pivot_table(
        index=["family", "n"], columns="transform", values="mean_ise"
    )[col_order]

    # Flat row labels: "gaussian / n=500"
    pivot.index = [f"{fam} / n={n}" for fam, n in pivot.index]
    pivot.columns.name = "Transform"

    n_rows, n_cols = pivot.shape
    fig_w = max(6, n_cols * 2.8 + 1.5)
    fig_h = max(3, n_rows * 0.65 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".5f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="lightgrey",
        ax=ax,
        cbar_kws={"label": "Mean ISE", "shrink": 0.75},
    )
    ax.set_title(
        "Experiment 1 — Mean ISE by Copula, Sample Size, and Transform",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Transform", fontsize=10)
    ax.set_ylabel("Copula / Sample Size", fontsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

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
    TRANSFORMS = settings["TRANSFORMS"]

    mode_label = "FAST" if FAST_MODE else "FULL"
    print("=" * 60)
    print(f"Experiment 1 — Marginal-Transform Inertness  [{mode_label} MODE]")
    print(f"  copulas    : {COPULAS}")
    print(f"  n values   : {NS}")
    print(f"  transforms : {TRANSFORMS}")
    print(f"  R          : {R}  |  tau={TAU}  |  n_gt={N_GT}")
    print("=" * 60)

    marginal = symmetric_nts()
    grid = make_eval_grid()

    df_all = _run_all_cells(COPULAS, NS, TRANSFORMS, marginal, grid)

    df_raw = _extract_raw_ise(df_all)
    df_summary = _build_summary(df_raw)

    ensure_results_dir()
    save_results(df_raw, "exp_1_raw.csv")
    save_results(df_summary, "exp_1_summary.csv")

    print("\n--- Summary table ---")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_rows", 200):
        print(df_summary.to_string(index=False))

    _plot_heatmap(df_summary, RESULTS_DIR / "exp_1_ise_table.png")


if __name__ == "__main__":
    main()
