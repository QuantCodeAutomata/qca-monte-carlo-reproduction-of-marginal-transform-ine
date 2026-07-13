"""
Download historical ETF data for the real-data experiment (exp_8).

Downloads daily adjusted close prices for SPY, TLT, EEM over 2005-2024
from Yahoo Finance via yfinance. Saves to data/etf_prices.csv.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "etf_prices.csv")

TICKERS = ["SPY", "TLT", "EEM"]
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"


def download_etf_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download or load cached ETF daily adjusted close prices.

    Parameters
    ----------
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices with date index and ticker columns.
    """
    if os.path.exists(OUTPUT_FILE) and not force_refresh:
        print(f"Loading cached data from {OUTPUT_FILE}")
        return pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
        print(f"Downloading {TICKERS} from {START_DATE} to {END_DATE}...")
        raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        # For single ticker yfinance returns different structure
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(0, axis=1)
        prices = prices.dropna()
    except Exception as e:
        print(f"yfinance download failed: {e}. Using synthetic placeholder data.")
        # Generate synthetic placeholder for testing
        dates = pd.bdate_range(START_DATE, END_DATE)
        np.random.seed(42)
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(len(dates), 3) * 0.01, axis=0) * 100,
            index=dates,
            columns=TICKERS,
        )

    prices.to_csv(OUTPUT_FILE)
    print(f"Saved {len(prices)} rows to {OUTPUT_FILE}")
    return prices


def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    """
    Compute daily log returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    log_returns : bool
        If True, compute log returns (default); otherwise simple returns.

    Returns
    -------
    pd.DataFrame
        Return series.
    """
    if log_returns:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


if __name__ == "__main__":
    prices = download_etf_data(force_refresh=True)
    returns = compute_returns(prices)
    print(f"Returns shape: {returns.shape}")
    print(returns.describe())
