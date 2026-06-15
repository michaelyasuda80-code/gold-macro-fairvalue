"""Yahoo Finance data fetching for gold macro analysis.

All series are daily close, USD-denominated where applicable. We avoid FRED so
that the app deploys cleanly on Streamlit Cloud with zero secrets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Series:
    ticker: str
    label: str
    group: str
    transform: str  # "level" | "log" | "yield"
    note: str = ""


# Curated universe. group lets the UI cluster checkboxes.
UNIVERSE: tuple[Series, ...] = (
    # --- target ---
    Series("GC=F", "Gold (front-month futures)", "target", "log",
           "COMEX gold futures, USD/oz"),
    # --- USD / FX ---
    Series("DX-Y.NYB", "US Dollar Index (DXY)", "usd", "log"),
    Series("JPY=X", "USD/JPY", "usd", "log"),
    Series("EURUSD=X", "EUR/USD", "usd", "log"),
    Series("CNY=X", "USD/CNY", "usd", "log",
           "China gold demand proxy"),
    # --- Rates (real-yield proxy is built from these) ---
    Series("^TNX", "US 10Y yield", "rates", "yield"),
    Series("^FVX", "US 5Y yield", "rates", "yield"),
    Series("^TYX", "US 30Y yield", "rates", "yield"),
    Series("TIP", "TIPS ETF (real-yield proxy, inverse)", "rates", "log"),
    Series("IEF", "7-10Y Treasury ETF", "rates", "log"),
    Series("TLT", "20+Y Treasury ETF", "rates", "log"),
    # --- Energy / commodities ---
    Series("CL=F", "WTI crude", "commodity", "log"),
    Series("BZ=F", "Brent crude", "commodity", "log"),
    Series("NG=F", "Natural gas", "commodity", "log"),
    Series("HG=F", "Copper (China/industrial)", "commodity", "log"),
    Series("SI=F", "Silver (gold/silver ratio cross-check)", "commodity", "log"),
    # --- Risk ---
    Series("^VIX", "VIX (equity vol)", "risk", "log"),
    Series("^GSPC", "S&P 500", "risk", "log"),
    Series("EEM", "EM equities ETF", "risk", "log"),
    # --- Crypto (alt store of value) ---
    Series("BTC-USD", "Bitcoin", "alt", "log"),
)

GOLD_TICKER = "GC=F"


def fetch_raw(tickers: Iterable[str], start: str = "2015-01-01",
              end: str | None = None) -> pd.DataFrame:
    """Download Close prices for given tickers from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker.
    Missing tickers are dropped silently.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        out = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        # Single ticker case — yfinance returns flat columns
        out = raw[["Close"]].rename(columns={"Close": next(iter(tickers))})
    out = out.sort_index().ffill().dropna(how="all")
    return out


def transform(series_def: Series, raw: pd.Series) -> pd.Series:
    """Apply per-series transform to make it model-ready."""
    s = raw.copy()
    if series_def.transform == "log":
        s = np.log(s.replace(0, np.nan))
    elif series_def.transform == "yield":
        # Yahoo returns 10Y yield as e.g. 45.0 meaning 4.50%.
        # Convert to percent (4.50).
        s = s / 10.0
    return s.rename(series_def.ticker)


def build_panel(universe: Iterable[Series] = UNIVERSE,
                start: str = "2015-01-01",
                end: str | None = None) -> pd.DataFrame:
    """Fetch all tickers, apply transforms, return a clean joined panel."""
    tickers = [s.ticker for s in universe]
    raw = fetch_raw(tickers, start=start, end=end)
    cols = []
    for s in universe:
        if s.ticker in raw.columns:
            t = transform(s, raw[s.ticker])
            # Skip series that came back entirely empty (delisted / bad ticker)
            # so one missing feed can't wipe out the whole panel via dropna.
            if t.notna().sum() == 0:
                continue
            cols.append(t)
    panel = pd.concat(cols, axis=1)
    # Drop any column that is mostly empty, then drop rows with gaps.
    keep = [c for c in panel.columns if panel[c].notna().mean() > 0.5]
    panel = panel[keep].dropna(how="any")
    return panel


def add_engineered(panel: pd.DataFrame) -> pd.DataFrame:
    """Add derived series the model cares about (real-yield proxy, BEI proxy)."""
    out = panel.copy()
    # Inflation expectation proxy: log(TIP) - log(IEF).
    # When BEI rises, TIPs outperform nominals -> ratio rises.
    if "TIP" in out and "IEF" in out:
        out["BEI_PROXY"] = out["TIP"] - out["IEF"]
    # Real-yield proxy = 10Y nominal - BEI proxy (scaled).
    # We scale the proxy so it sits in a sensible range; the regression
    # absorbs the scale via its beta, so absolute calibration doesn't matter.
    if "^TNX" in out and "BEI_PROXY" in out:
        bei_centered = out["BEI_PROXY"] - out["BEI_PROXY"].rolling(252, min_periods=60).mean()
        out["REAL_YIELD_PROXY"] = out["^TNX"] - 100 * bei_centered
    # Gold/Silver ratio (sentiment, not a driver — keep for cross-check)
    if "GC=F" in out and "SI=F" in out:
        out["GOLD_SILVER"] = out["GC=F"] - out["SI=F"]
    return out.dropna(how="any")


# Default factor set used by the model. Curated for low multicollinearity.
DEFAULT_FACTORS: tuple[str, ...] = (
    "REAL_YIELD_PROXY",
    "DX-Y.NYB",
    "CL=F",
    "^VIX",
    "^GSPC",
    "BTC-USD",
    "CNY=X",
)
