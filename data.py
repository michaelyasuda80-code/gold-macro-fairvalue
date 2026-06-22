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
    Series("^IRX", "US 13w T-bill yield (front-end/policy)", "rates", "yield"),
    Series("^FVX", "US 5Y yield", "rates", "yield"),
    Series("^TNX", "US 10Y yield", "rates", "yield"),
    Series("^TYX", "US 30Y yield", "rates", "yield"),
    Series("TIP", "TIPS ETF (real-yield proxy, inverse)", "rates", "log"),
    Series("IEF", "7-10Y Treasury ETF", "rates", "log"),
    Series("TLT", "20+Y Treasury ETF", "rates", "log"),
    Series("1482.T", "Japan JGB ETF (iShares Core; inverse of JP yield)",
           "rates", "log", "Japan 10Y proxy; price up = JP yield down"),
    # --- Credit (risk appetite / financial conditions) ---
    Series("HYG", "High-yield corp bond ETF", "credit", "log"),
    Series("LQD", "Investment-grade corp bond ETF", "credit", "log"),
    # --- Energy / commodities ---
    Series("CL=F", "WTI crude", "commodity", "log"),
    Series("BZ=F", "Brent crude", "commodity", "log"),
    Series("NG=F", "Natural gas", "commodity", "log"),
    Series("HG=F", "Copper (China/industrial)", "commodity", "log"),
    Series("SI=F", "Silver (gold/silver ratio cross-check)", "commodity", "log"),
    Series("PL=F", "Platinum", "commodity", "log"),
    Series("PA=F", "Palladium", "commodity", "log"),
    # --- Risk ---
    Series("^VIX", "VIX (equity vol)", "risk", "log"),
    Series("^GSPC", "S&P 500", "risk", "log"),
    Series("EEM", "EM equities ETF", "risk", "log"),
    Series("FXI", "China large-cap ETF", "risk", "log"),
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


def fetch_jgb10y() -> pd.Series:
    """Japan 10Y JGB yield (%) from the Ministry of Finance (official, free,
    no key). Combines the all-history file with the current-year file so the
    latest weeks aren't missing. Returns an empty Series on any failure so the
    rest of the app still works without it.
    """
    import io
    import urllib.request

    base = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    frames = []
    for u in (base + "historical/jgbcme_all.csv", base + "jgbcme.csv"):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            df = pd.read_csv(io.StringIO(txt), skiprows=1)
            dt = pd.to_datetime(df["Date"], errors="coerce")
            s = pd.to_numeric(df["10Y"], errors="coerce")
            s.index = dt
            frames.append(s.dropna())
        except Exception:
            continue
    if not frames:
        return pd.Series(dtype="float64", name="JP10Y")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.rename("JP10Y")


def _eia_dnav(fname: str) -> pd.Series:
    """Read one EIA dnav weekly .xls (no API key) into a dated Series."""
    import io
    import urllib.request

    url = "https://www.eia.gov/dnav/pet/hist_xls/" + fname
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna().set_index("date")["v"]


def fetch_oil_supply() -> pd.DataFrame:
    """US crude supply fundamentals from the EIA (official, free, no key):
    commercial crude inventories and US field production. Returns log-levels so
    they slot straight into the model. Empty frame on failure (graceful).
    """
    out = {}
    for col, fname in (("INV_CRUDE", "WCESTUS1w.xls"), ("PROD_US", "WCRFPUS2w.xls")):
        try:
            out[col] = np.log(_eia_dnav(fname))
        except Exception:
            continue
    return pd.DataFrame(out)


def transform(series_def: Series, raw: pd.Series) -> pd.Series:
    """Apply per-series transform to make it model-ready."""
    s = raw.copy()
    if series_def.transform == "log":
        s = np.log(s.replace(0, np.nan))
    elif series_def.transform == "yield":
        # Yahoo's ^TNX/^FVX/^TYX/^IRX already come as percent (e.g. 4.49 = 4.49%).
        # No scaling needed — leave as-is so the real-yield proxy is in real %.
        s = s
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
    # Keep columns that have a reasonable amount of history, but DON'T drop rows
    # globally: a late-starting series (e.g. a 2018+ JGB ETF) must not truncate
    # history for other assets. Each model applies its own dropna on just the
    # columns it uses (see model.fit_ols etc.), so leading NaNs are harmless.
    keep = [c for c in panel.columns if panel[c].notna().mean() > 0.3]
    panel = panel[keep].dropna(how="all")
    # Join the official Japan 10Y JGB yield (non-Yahoo source). Aligned to the
    # panel's calendar and forward-filled across JP/US holiday mismatches.
    jgb = fetch_jgb10y()
    if not jgb.empty:
        panel = panel.join(jgb)
        panel["JP10Y"] = panel["JP10Y"].ffill()
    # Oil supply fundamentals from EIA (crude inventory, US production).
    supply = fetch_oil_supply()
    if not supply.empty:
        panel = panel.join(supply)
        for c in supply.columns:
            panel[c] = panel[c].ffill()
    return panel


def add_engineered(panel: pd.DataFrame) -> pd.DataFrame:
    """Add derived series the model cares about (real-yield proxy, BEI, credit…)."""
    out = panel.copy()
    # Inflation expectation proxy: log(TIP) - log(IEF).
    # When BEI rises, TIPs outperform nominals -> ratio rises.
    if "TIP" in out and "IEF" in out:
        out["BEI_PROXY"] = out["TIP"] - out["IEF"]
    # Real-yield proxy = 10Y nominal (%) - breakeven-momentum adjustment.
    # ^TNX is now in true percent, so this sits in a real-yield-like range and
    # is dominated by the nominal level (correct), nudged by recent breakeven.
    if "^TNX" in out and "BEI_PROXY" in out:
        bei_centered = out["BEI_PROXY"] - out["BEI_PROXY"].rolling(252, min_periods=60).mean()
        out["REAL_YIELD_PROXY"] = out["^TNX"] - 100 * bei_centered
    # Credit risk appetite: log(HYG) - log(LQD). Falls when high-yield
    # underperforms investment-grade, i.e. credit stress / risk-off.
    if "HYG" in out and "LQD" in out:
        out["CREDIT_PROXY"] = out["HYG"] - out["LQD"]
    # Yield-curve slope (10Y - 5Y), a growth/term-premium signal.
    if "^TNX" in out and "^FVX" in out:
        out["CURVE_10Y_5Y"] = out["^TNX"] - out["^FVX"]
    # US–Japan 10Y rate differential (the textbook USD/JPY driver), in % points.
    if "^TNX" in out and "JP10Y" in out:
        out["US_JP_10Y_DIFF"] = out["^TNX"] - out["JP10Y"]
    # Gold/Silver ratio (sentiment, not a driver — keep for cross-check)
    if "GC=F" in out and "SI=F" in out:
        out["GOLD_SILVER"] = out["GC=F"] - out["SI=F"]
    # Keep NaNs (don't global-dropna): each model drops on its own columns, so
    # one late-starting / engineered series can't shorten the others' history.
    return out.dropna(how="all")


# Default factor set used by the model. Curated for low multicollinearity.
DEFAULT_FACTORS: tuple[str, ...] = (
    "REAL_YIELD_PROXY",
    "DX-Y.NYB",
    "CL=F",
    "^VIX",
    "^GSPC",
    "BTC-USD",
    "CNY=X",
    "CREDIT_PROXY",
)

# Every factor the sidebar can offer (curated + extended), grouped for display.
ALL_FACTORS: tuple[str, ...] = (
    # rates / inflation
    "REAL_YIELD_PROXY", "BEI_PROXY", "^IRX", "^FVX", "^TNX", "^TYX",
    "CURVE_10Y_5Y", "JP10Y", "US_JP_10Y_DIFF", "TIP", "IEF", "TLT", "1482.T",
    # usd / fx
    "DX-Y.NYB", "JPY=X", "EURUSD=X", "CNY=X",
    # credit
    "CREDIT_PROXY", "HYG", "LQD",
    # supply (oil fundamentals)
    "INV_CRUDE", "PROD_US",
    # commodity
    "CL=F", "BZ=F", "NG=F", "HG=F", "SI=F", "PL=F", "PA=F",
    # risk
    "^VIX", "^GSPC", "EEM", "FXI",
    # alt
    "BTC-USD",
)

# Japanese display labels for every factor (UI + charts). Fallback = code.
FACTOR_LABELS_JA: dict[str, str] = {
    "GC=F": "金（COMEX先物）",
    "DX-Y.NYB": "ドル指数(DXY)",
    "JPY=X": "ドル円",
    "EURUSD=X": "ユーロドル",
    "CNY=X": "ドル人民元",
    "^IRX": "米13週Tビル(政策金利)",
    "^FVX": "米5年金利",
    "^TNX": "米10年金利",
    "^TYX": "米30年金利",
    "CURVE_10Y_5Y": "利回り曲線(10年-5年)",
    "JP10Y": "日本10年金利(財務省)",
    "US_JP_10Y_DIFF": "日米10年金利差",
    "TIP": "TIPS ETF",
    "IEF": "米7-10年債ETF",
    "TLT": "米20年超債ETF",
    "1482.T": "日本国債ETF(日本金利の逆)",
    "HYG": "ハイイールド債ETF",
    "LQD": "投資適格債ETF",
    "CREDIT_PROXY": "クレジット選好(HY/IG)",
    "INV_CRUDE": "米原油在庫(EIA)",
    "PROD_US": "米原油生産(EIA)",
    "CL=F": "WTI原油",
    "BZ=F": "ブレント原油",
    "NG=F": "天然ガス",
    "HG=F": "銅",
    "SI=F": "銀",
    "PL=F": "プラチナ",
    "PA=F": "パラジウム",
    "^VIX": "VIX(株式恐怖指数)",
    "^GSPC": "S&P500",
    "EEM": "新興国株ETF",
    "FXI": "中国大型株ETF",
    "BTC-USD": "ビットコイン",
    "BEI_PROXY": "期待インフレ(BEI近似)",
    "REAL_YIELD_PROXY": "実質金利(近似)",
    "GOLD_SILVER": "金銀レシオ",
}


def label_ja(code: str) -> str:
    """Japanese label for a factor code, falling back to the code itself."""
    return FACTOR_LABELS_JA.get(code, code)


# ---------------- Assets (multi-asset dashboard) ----------------

@dataclass(frozen=True)
class Asset:
    key: str                          # internal id, also widget-key namespace
    target: str                       # ticker to explain
    name: str                         # Japanese display name
    unit: str                         # price unit for axes
    icon: str                         # emoji for the tab
    default_factors: tuple[str, ...]  # model default
    exclude: tuple[str, ...]          # factors not offered (self / circular)
    price_decimals: int = 0
    price_prefix: str = "$"           # currency symbol before the number
    price_suffix: str = ""            # unit after the number (e.g. 円)
    crosscheck: tuple = ()            # (ticker, label) for a non-circular
    #                                   benchmark spread panel, e.g. WTI vs Brent


# Crude oil drivers. Demand/financial side kept lean (one broad equity proxy)
# after diagnosing that copper + EM/China were collinear and — being in their
# own electrification-driven bull market — decoupled from oil and inflated the
# fair value badly (fitted ~$126 vs ~$77 spot). Supply side added: EIA crude
# inventory (-) and US field production (-). Brent/other crudes stay out
# (circular). This trades some R² for a far more realistic fair value.
OIL_DEFAULT_FACTORS: tuple[str, ...] = (
    "DX-Y.NYB",
    "^GSPC",
    "^VIX",
    "^TNX",
    "CREDIT_PROXY",
    "INV_CRUDE",
    "PROD_US",
)

OIL_TICKER = "CL=F"

# USD/JPY drivers (the rate-differential story, made explicit):
#   ^TNX + JP10Y = US & Japan 10Y yields, the two legs of the differential
#                  (JP10Y is the official MOF yield, not the ETF proxy)
#   ^IRX       = US front-end / policy
#   EURUSD=X   = broad-dollar strength via a clean single pair
#   ^VIX,^GSPC = risk-off / safe-haven & carry
#   CL=F       = oil; Japan is an energy importer → terms of trade
#   CREDIT_PROXY = financial conditions
# Two separate legs fit better than a single forced differential; the explicit
# US_JP_10Y_DIFF factor is also offered for users who prefer one clean driver.
# DXY/CNY stay excluded (broad-USD / another USD pair = more circular than EUR).
JPY_DEFAULT_FACTORS: tuple[str, ...] = (
    "^TNX",
    "JP10Y",
    "^IRX",
    "EURUSD=X",
    "^VIX",
    "^GSPC",
    "CL=F",
    "CREDIT_PROXY",
)

ASSETS: dict[str, Asset] = {
    "gold": Asset(
        key="gold", target="GC=F", name="金", unit="USD/oz", icon="🪙",
        default_factors=DEFAULT_FACTORS,
        exclude=("GC=F", "GOLD_SILVER"),
        price_decimals=0,
    ),
    "oil": Asset(
        key="oil", target="CL=F", name="原油(WTI)", unit="USD/bbl", icon="🛢️",
        default_factors=OIL_DEFAULT_FACTORS,
        exclude=("CL=F", "BZ=F", "GOLD_SILVER"),
        price_decimals=1,
        crosscheck=("BZ=F", "Brent"),
    ),
    "jpy": Asset(
        key="jpy", target="JPY=X", name="ドル円", unit="円/ドル", icon="💴",
        default_factors=JPY_DEFAULT_FACTORS,
        exclude=("JPY=X", "DX-Y.NYB", "CNY=X", "GOLD_SILVER"),
        price_decimals=1, price_prefix="", price_suffix="円",
    ),
}


def factor_options(asset: Asset) -> list[str]:
    """Selectable factors for an asset: everything except its excludes/target."""
    return [f for f in ALL_FACTORS if f not in asset.exclude and f != asset.target]
