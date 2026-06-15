"""Fair-value model for gold.

Approach:
1. Regress log(gold) on chosen macro factors (OLS).
2. Residual = actual - fitted. A persistent positive residual = gold trades rich
   versus what macros justify; negative = cheap.
3. Z-score the residual on a rolling window for signal sizing.
4. Decompose the latest level into per-factor contributions for the dashboard.

We use levels (not differences) because gold and most drivers are cointegrated
over multi-year horizons. For robustness the app exposes both full-sample and
rolling-window fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


@dataclass
class FitResult:
    coef: pd.Series          # incl. const
    fitted: pd.Series        # in target units (log price)
    resid: pd.Series         # actual - fitted (log space)
    r2: float
    factors: list[str]


def fit_ols(panel: pd.DataFrame, target: str, factors: Sequence[str]) -> FitResult:
    """Plain OLS on levels (target is already log-transformed upstream)."""
    df = panel[[target, *factors]].dropna()
    y = df[target]
    X = sm.add_constant(df[list(factors)])
    res = sm.OLS(y, X).fit()
    return FitResult(
        coef=res.params,
        fitted=res.fittedvalues,
        resid=res.resid,
        r2=float(res.rsquared),
        factors=list(factors),
    )


def rolling_beta(panel: pd.DataFrame, target: str, factors: Sequence[str],
                 window: int = 252) -> pd.DataFrame:
    """Rolling OLS betas. Useful for spotting regime shifts in the relationship.

    Uses statsmodels RollingOLS (vectorized) — orders of magnitude faster than
    a Python loop of individual OLS fits, which matters for interactive reruns.
    """
    df = panel[[target, *factors]].dropna()
    y = df[target]
    X = sm.add_constant(df[list(factors)])
    res = RollingOLS(y, X, window=window).fit()
    betas = res.params.dropna(how="all")
    return betas


def residual_zscore(resid: pd.Series, window: int = 126) -> pd.Series:
    mu = resid.rolling(window, min_periods=window // 2).mean()
    sd = resid.rolling(window, min_periods=window // 2).std()
    return (resid - mu) / sd


def contribution_breakdown(fit: FitResult, panel: pd.DataFrame,
                           target: str, baseline: str = "mean") -> pd.DataFrame:
    """Per-factor contribution to the latest fitted value.

    contribution_i = beta_i * (x_i_now - x_i_baseline)

    baseline:
      "mean"   - long-run mean across the fit window (default)
      "1y_ago" - value 252 trading days ago

    Returns a DataFrame keyed by factor with columns:
      beta, x_now, x_base, contrib (log units), contrib_pct (% of fitted move)
    """
    df = panel[[target, *fit.factors]].dropna()
    last = df.iloc[-1]
    if baseline == "1y_ago" and len(df) > 252:
        base = df.iloc[-252]
    else:
        base = df.mean()

    rows = []
    for f in fit.factors:
        beta = float(fit.coef.get(f, 0.0))
        x_now = float(last[f])
        x_base = float(base[f])
        contrib = beta * (x_now - x_base)
        rows.append({
            "factor": f,
            "beta": beta,
            "x_now": x_now,
            "x_base": x_base,
            "contrib_log": contrib,
        })
    out = pd.DataFrame(rows).set_index("factor")
    # Convert log-space contribution to approximate % of price.
    out["contrib_pct"] = (np.exp(out["contrib_log"]) - 1) * 100
    return out


def change_attribution(fit: FitResult, panel: pd.DataFrame, target: str,
                       lookback: int = 21) -> pd.DataFrame:
    """Attribute the CHANGE in fair value over `lookback` days to each factor.

    This is the intuitive "why did gold move?" view: it decomposes the recent
    move in the model's fair value into per-factor pushes, using

        Δcontribution_i = beta_i * (x_i[t] - x_i[t-lookback])

    Because the target is in log space, contributions are in log-points; we
    report them ×100 so they read as ~percentage points and sum (with the
    residual change) to the total % change in fair value.

    Returns rows per factor plus an "ACTUAL (move)" and "FAIR (model)" summary.
    """
    df = panel[[target, *fit.factors]].dropna()
    if len(df) <= lookback:
        lookback = max(1, len(df) // 4)
    now = df.iloc[-1]
    past = df.iloc[-1 - lookback]

    rows = []
    fair_change_log = 0.0
    for f in fit.factors:
        beta = float(fit.coef.get(f, 0.0))
        dx = float(now[f] - past[f])
        contrib_log = beta * dx
        fair_change_log += contrib_log
        rows.append({"factor": f, "delta_x": dx, "beta": beta,
                     "contrib_pts": contrib_log * 100})
    out = pd.DataFrame(rows).set_index("factor").sort_values("contrib_pts")

    actual_change_log = float(now[target] - past[target])
    summary = {
        "fair_change_pct": (np.exp(fair_change_log) - 1) * 100,
        "actual_change_pct": (np.exp(actual_change_log) - 1) * 100,
        "lookback": lookback,
    }
    return out, summary


def fair_price(fit: FitResult, panel: pd.DataFrame) -> pd.Series:
    """Return fitted price in original (non-log) units."""
    return np.exp(fit.fitted)


def mispricing(panel: pd.DataFrame, fit: FitResult, target: str) -> pd.DataFrame:
    """Time series of actual, fair, residual, and z-score."""
    actual = np.exp(panel[target].loc[fit.fitted.index])
    fair = np.exp(fit.fitted)
    z = residual_zscore(fit.resid)
    return pd.DataFrame({
        "actual": actual,
        "fair": fair,
        "resid_log": fit.resid,
        "resid_pct": (actual / fair - 1) * 100,
        "z": z,
    })
