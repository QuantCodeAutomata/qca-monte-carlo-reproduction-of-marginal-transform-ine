"""
Experiment 2 – Monte Carlo Reproduction of Asymmetric Marginal Robustness.

Evaluates LGC estimation under a skewed NTS (NIG) marginal (target skewness
≈ -0.84) across copula families and gaussianization transforms at a fixed
sample size n=2500.

Outputs
-------
results/exp_2_raw.csv          : one row per (family, tau, n, replication, transform)
results/exp_2_summary.csv      : mean / sd ISE aggregated by (family, tau, n, transform)
results/exp_2_ise_comparison.png : grouped bar chart of mean ISE by (copula, transform)
"""

# ---------------------------------------------------------------------------
# FAST_MODE flag — set to False for the full experiment
# ---------------------------------------------------------------------------
FAST_MODE = True

import pathlib
import sys

# Ensure project root is on sys.path so 'src.*' imports work from any cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nts import skewed_nts
from src.lgc import make_eval_grid
from src.simulation import run_mc_cell
from src.metrics import aggregate_ise
from exp.utils import ensure_results_dir, save_results, RESULTS_DIR


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

_FULL = dict(
    R=25,
    COPULAS=["gaussian", "clayton", "t4", "gumbel"],
)

_FAST = dict(
    R=3,
    COPULAS=["gaussian", "clayton"],
)

N = 2_500
TAU = 0.5
N_GT = 20_000
TRANSFORMS = ["empirical", "oracle", "fitted"]
TARGET_SKEW = -0.84
BASE_SEED = 42
VERBOSE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_all_cells(copulas, transforms, marginal, grid):
    """Iterate over copula families and concatenate full MC output."""
    frames = []
    settings = _FAST if FAST_MODE else _FULL
    R = settings["R"]
    total = len(copulas)

    for idx, family in enumerate(copulas, start=1):
        print(f"\n[{idx}/{total}] family={family}  n={N}  R={R}")
        df_cell = run_mc_cell(
            family=family,
            tau=TAU,
            n=N,
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


def _plot_barchart(df_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Save grouped bar chart of mean ISE by copula and transform."""
    transform_order = [t for t in ["empirical", "oracle", "fitted"]
                       if t in df_summary["transform"].unique()]
    family_order = [f for f in ["gaussian", "clayton", "t4", "gumbel"]
                    if f in df_summary["family"].unique()]

    n_families = len(family_order)
    n_transforms = len(transform_order)
    x = np.arange(n_families)
    bar_width = 0.75 / n_transforms

    # Colour palette: one colour per transform
    colours = plt.cm.Set2(np.linspace(0, 0.6, n_transforms))

    fig, ax = plt.subplots(figsize=(max(6, n_families * 2.0), 5))

    for i, tf in enumerate(transform_order):
        subset = df_summary[df_summary["transform"] == tf].set_index("family")
        means = np.array([
            subset.loc[f, "mean_ise"] if f in subset.index else np.nan
            for f in family_order
        ])
        sds = np.array([
            subset.loc[f, "sd_ise"] if f in subset.index else 0.0
            for f in family_order
        ])
        # Replace NaN sd with 0 for error bars
        sds = np.where(np.isfinite(sds), sds, 0.0)

        offsets = (i - n_transforms / 2.0 + 0.5) * bar_width
        ax.bar(
            x + offsets,
            means,
            bar_width,
            label=tf,
            yerr=sds,
            capsize=4,
            color=colours[i],
            alpha=0.85,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(family_order, fontsize=10)
    ax.set_xlabel("Copula Family", fontsize=11)
    ax.set_ylabel("Mean ISE", fontsize=11)
    ax.set_title(
        f"Experiment 2 — Mean ISE by Copula and Transform\n"
        f"(Skewed NTS marginal, target skew={TARGET_SKEW}, n={N})",
        fontsize=11,
        pad=10,
    )
    ax.legend(title="Transform", fontsize=9, title_fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

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
    COPULAS = settings["COPULAS"]

    mode_label = "FAST" if FAST_MODE else "FULL"
    print("=" * 60)
    print(f"Experiment 2 — Asymmetric Marginal Robustness  [{mode_label} MODE]")
    print(f"  copulas    : {COPULAS}")
    print(f"  n          : {N}  |  tau={TAU}")
    print(f"  transforms : {TRANSFORMS}")
    print(f"  R          : {R}  |  n_gt={N_GT}")
    print(f"  target skew: {TARGET_SKEW}")
    print("=" * 60)

    marginal = skewed_nts(target_skew=TARGET_SKEW)
    grid = make_eval_grid()

    df_all = _run_all_cells(COPULAS, TRANSFORMS, marginal, grid)

    df_raw = _extract_raw_ise(df_all)
    df_summary = _build_summary(df_raw)

    ensure_results_dir()
    save_results(df_raw, "exp_2_raw.csv")
    save_results(df_summary, "exp_2_summary.csv")

    print("\n--- Summary table ---")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_rows", 200):
        print(df_summary.to_string(index=False))

    _plot_barchart(df_summary, RESULTS_DIR / "exp_2_ise_comparison.png")


if __name__ == "__main__":
    main()
