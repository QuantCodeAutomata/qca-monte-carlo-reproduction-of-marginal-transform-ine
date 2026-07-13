"""Shared utilities for LGC experiment scripts."""

import pathlib

import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def ensure_results_dir() -> pathlib.Path:
    """Create the results/ directory if it does not already exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_results(df: pd.DataFrame, filename: str) -> pathlib.Path:
    """Save *df* as a CSV file inside the results/ directory.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    filename : str
        Target filename (e.g. ``'exp_1_raw.csv'``).

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    ensure_results_dir()
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path
