"""Streamlit dashboard: Gold fair-value vs market, with macro contribution.

Run locally:   streamlit run app.py
Deploy:        https://share.streamlit.io  (point at this repo, branch, app.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as D
import model as M

st.set_page_config(
    page_title="Gold Macro Fair Value",
    page_icon="🪙",
    layout="wide",
)


# ---------------- caching ----------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching market data…")
def load_panel(start: str) -> pd.DataFrame:
    raw = D.build_panel(start=start)
    return D.add_engineered(raw)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Computing rolling betas…")
def cached_rolling_beta(start: str, factors: tuple[str, ...], window: int) -> pd.DataFrame:
    # start is part of the cache key so a wider window invalidates correctly.
    panel = load_panel(start)
    return M.rolling_beta(panel, D.GOLD_TICKER, list(factors), window=window)


# ---------------- sidebar ----------------

st.sidebar.title("Settings")

start_choice = st.sidebar.selectbox(
    "History window",
    options=["2018-01-01", "2015-01-01", "2010-01-01"],
    index=0,
)

available_factors = list(D.DEFAULT_FACTORS)
factors = st.sidebar.multiselect(
    "Factors in fair-value model",
    options=[
        "REAL_YIELD_PROXY", "^TNX", "^FVX", "^TYX",
        "DX-Y.NYB", "JPY=X", "EURUSD=X", "CNY=X",
        "CL=F", "BZ=F", "HG=F", "SI=F", "NG=F",
        "^VIX", "^GSPC", "EEM",
        "BTC-USD",
        "TIP", "IEF", "TLT",
        "BEI_PROXY",
    ],
    default=available_factors,
    help="Curated default avoids multicollinearity. Add/remove freely.",
)

zwin = st.sidebar.slider("Residual z-score window (days)", 30, 252, 126, step=10)
roll_win = st.sidebar.slider("Rolling-beta window (days)", 60, 504, 252, step=20)
baseline_choice = st.sidebar.radio(
    "Contribution baseline",
    options=["mean", "1y_ago"],
    horizontal=True,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Yahoo Finance (delayed). Real yield is a proxy built from "
    "10Y nominal minus a TIP/IEF-derived breakeven. For research/education "
    "only — not investment advice."
)


# ---------------- fetch + fit ----------------

panel = load_panel(start_choice)

# Guard: drop factors not in panel (e.g. data-fetch hiccup)
factors = [f for f in factors if f in panel.columns]
if not factors:
    st.error("No factors available — try a wider history window.")
    st.stop()

fit = M.fit_ols(panel, D.GOLD_TICKER, factors)
mp = M.mispricing(panel, fit, D.GOLD_TICKER)
contrib = M.contribution_breakdown(fit, panel, D.GOLD_TICKER, baseline=baseline_choice)


# ---------------- header KPIs ----------------

st.title("🪙 Gold Macro Fair Value")
st.caption(
    f"Last data: **{panel.index[-1].date()}**  ·  "
    f"R² of fit: **{fit.r2:.3f}**  ·  factors: **{len(factors)}**"
)

latest = mp.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot (GC=F)", f"${latest.actual:,.0f}")
c2.metric("Fair value", f"${latest.fair:,.0f}",
          delta=f"{latest.resid_pct:+.2f}% rich" if latest.resid_pct > 0 else f"{latest.resid_pct:+.2f}% cheap")
c3.metric("Z-score", f"{latest.z:+.2f} σ",
          help="Residual z-score. |z|>2 = potential mispricing")
signal = "🔴 RICH" if latest.z > 2 else "🟢 CHEAP" if latest.z < -2 else "⚪ NEUTRAL"
c4.metric("Signal (|z|>2)", signal)


# ---------------- chart 1: actual vs fair + residual ----------------

st.subheader("Spot vs Fair value, and residual z-score")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.65, 0.35],
)
fig.add_trace(go.Scatter(x=mp.index, y=mp.actual, name="Spot", line=dict(width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=mp.index, y=mp.fair, name="Fair value", line=dict(width=2, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=mp.index, y=mp.z, name="Z-score", line=dict(width=1.5)), row=2, col=1)
fig.add_hline(y=2, line=dict(dash="dot", width=1, color="red"), row=2, col=1)
fig.add_hline(y=-2, line=dict(dash="dot", width=1, color="green"), row=2, col=1)
fig.add_hline(y=0, line=dict(width=1, color="gray"), row=2, col=1)
fig.update_yaxes(title_text="USD/oz", row=1, col=1)
fig.update_yaxes(title_text="σ", row=2, col=1)
fig.update_layout(
    height=620, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)


# ---------------- chart 2: change attribution (why did gold move?) ----------------

st.subheader("Why did gold move? — change attribution")

lb_label = st.radio(
    "Lookback",
    options=["1W", "1M", "3M", "6M", "1Y"],
    index=1, horizontal=True,
)
lb_days = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}[lb_label]

attr, summ = M.change_attribution(fit, panel, D.GOLD_TICKER, lookback=lb_days)

s1, s2, s3 = st.columns(3)
s1.metric(f"Actual move ({lb_label})", f"{summ['actual_change_pct']:+.2f}%")
s2.metric("Model-implied move", f"{summ['fair_change_pct']:+.2f}%")
s3.metric("Unexplained (residual)",
          f"{summ['actual_change_pct'] - summ['fair_change_pct']:+.2f}%",
          help="Actual minus model. Large unexplained move = potential mispricing.")

wfall = go.Figure(go.Bar(
    x=attr["contrib_pts"],
    y=attr.index,
    orientation="h",
    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in attr["contrib_pts"]],
    text=[f"{v:+.2f}" for v in attr["contrib_pts"]],
    textposition="auto",
))
wfall.update_layout(
    height=max(280, 44 * len(attr)),
    xaxis_title=f"Push on fair value over {lb_label} (percentage points; sums to model-implied move)",
    margin=dict(l=10, r=10, t=10, b=30),
)
st.plotly_chart(wfall, use_container_width=True)

with st.expander("Level contribution vs long-run baseline (advanced)"):
    st.caption("Decomposes today's fair-value *level* vs the chosen baseline. "
               "Trending assets (S&P, BTC) dominate here — use the change view above "
               "for the day-to-day story.")
    st.dataframe(contrib.style.format({
        "beta": "{:+.4f}", "x_now": "{:.4f}", "x_base": "{:.4f}",
        "contrib_log": "{:+.4f}", "contrib_pct": "{:+.2f}%",
    }), use_container_width=True)


# ---------------- chart 3: rolling beta ----------------

st.subheader(f"Rolling betas ({roll_win}-day window)")

rb = cached_rolling_beta(start_choice, tuple(factors), roll_win)

rb_fig = go.Figure()
for col in [c for c in rb.columns if c != "const"]:
    rb_fig.add_trace(go.Scatter(x=rb.index, y=rb[col], name=col, mode="lines"))
rb_fig.update_layout(
    height=420, hovermode="x unified",
    yaxis_title="beta",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(rb_fig, use_container_width=True)


# ---------------- chart 4: correlation heatmap ----------------

st.subheader("Daily-change correlation (last 252 days)")
ret = panel[[D.GOLD_TICKER, *factors]].diff().tail(252)
corr = ret.corr()
heat = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale="RdBu", zmin=-1, zmax=1,
    text=corr.round(2).values, texttemplate="%{text}",
))
heat.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(heat, use_container_width=True)


# ---------------- footer ----------------

with st.expander("Methodology"):
    st.markdown("""
**Model.** OLS on price levels: `log(GC=F) = α + Σ βᵢ · xᵢ + ε`.
Most factors are log-prices; rates are in percent. The real-yield proxy is
`10Y nominal − 100 × centered(log(TIP) − log(IEF))`, which tracks FRED's
`DFII10` series closely enough for direction-finding.

**Mispricing.** Residual ε in log space ≈ % deviation. Z-score on a rolling
window normalizes for regime-dependent residual volatility. |z| > 2 is the
conventional statistical-arbitrage threshold.

**Caveats.** OLS levels can be spurious if series aren't cointegrated; use the
rolling-beta panel to sanity-check stability. Yahoo data has occasional gaps
on futures contracts around roll dates.
""")
