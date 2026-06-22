"""Streamlit ダッシュボード: 資産（金・原油…）のマクロ理論値 vs 実勢。

ローカル実行:  streamlit run app.py
デプロイ:      https://share.streamlit.io （このリポジトリ / branch / app.py を指定）
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
    page_title="マクロ理論値ダッシュボード",
    page_icon="📊",
    layout="wide",
)


# ---------------- キャッシュ ----------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner="市場データを取得中…")
def load_panel(start: str) -> pd.DataFrame:
    raw = D.build_panel(start=start)
    return D.add_engineered(raw)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="ローリングβを計算中…")
def cached_rolling_beta(start: str, target: str, factors: tuple[str, ...],
                        window: int) -> pd.DataFrame:
    panel = load_panel(start)
    return M.rolling_beta(panel, target, list(factors), window=window)


def jp(codes):
    """factorコード（単体orリスト）を日本語ラベルへ。"""
    if isinstance(codes, str):
        return D.label_ja(codes)
    return [D.label_ja(c) for c in codes]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_news(tickers: tuple[str, ...], query: str) -> list[dict]:
    """English (yfinance) + Japanese (Google News) news, merged, newest first."""
    items = D.fetch_yf_news(tickers) + (D.fetch_jp_news(query) if query else [])
    items.sort(key=lambda x: x["time"], reverse=True)
    seen, out = set(), []
    for x in items:
        key = x["title"][:48]
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def rel_time(ts: pd.Timestamp) -> str:
    """UTC timestamp -> '◯分前 / ◯時間前 / ◯日前'."""
    mins = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    if mins < 60:
        return f"{int(max(mins, 0))}分前"
    if mins < 60 * 24:
        return f"{int(mins // 60)}時間前"
    return f"{int(mins // (60 * 24))}日前"


# ---------------- 解釈文の自動生成 ----------------

def build_narrative(asset_name: str, summ: dict, attr: pd.DataFrame,
                    z: float, lb_label: str) -> str:
    actual = summ["actual_change_pct"]
    fair = summ["fair_change_pct"]
    unexp = actual - fair

    pos = attr[attr["contrib_pts"] > 0].sort_values("contrib_pts", ascending=False).head(2)
    neg = attr[attr["contrib_pts"] < 0].sort_values("contrib_pts").head(2)
    pos_txt = "、".join(f"{jp(i)}({r.contrib_pts:+.1f}pt)" for i, r in pos.iterrows()) or "なし"
    neg_txt = "、".join(f"{jp(i)}({r.contrib_pts:+.1f}pt)" for i, r in neg.iterrows()) or "なし"

    if unexp < 0:
        flow = (f"差の {unexp:+.1f}%（未説明分）は、マクロ要因では説明できない"
                f"**{asset_name}固有の売り圧力**と読めます。")
    elif unexp > 0:
        flow = (f"差の {unexp:+.1f}%（未説明分）は、マクロでは説明できない"
                f"**{asset_name}固有の買い圧力**と読めます。")
    else:
        flow = "実勢と理論値はほぼ一致しており、マクロでほぼ説明できています。"

    if z > 2:
        zt = f"残差zスコアは **{z:+.2f}σ** で、統計的に **割高**（理論値より高い）圏です。"
    elif z < -2:
        zt = f"残差zスコアは **{z:+.2f}σ** で、統計的に **割安**（理論値より安い）圏です。"
    elif z >= 1:
        zt = f"残差zスコアは **{z:+.2f}σ** で、やや割高寄りですが中立圏です。"
    elif z <= -1:
        zt = f"残差zスコアは **{z:+.2f}σ** で、やや割安寄りですが中立圏です。"
    else:
        zt = f"残差zスコアは **{z:+.2f}σ** で、ほぼ理論値どおり（中立）です。"

    return (
        f"**直近{lb_label}**、{asset_name}は実勢で **{actual:+.1f}%** 動きました。"
        f"マクロ理論値は **{fair:+.1f}%** を示唆しています。{flow}\n\n"
        f"- 理論値を**押し上げた**主因：{pos_txt}\n"
        f"- 理論値を**押し下げた**主因：{neg_txt}\n\n"
        f"{zt}"
    )


# ---------------- 1資産ぶんのダッシュボード描画 ----------------

def render_dashboard(cfg: D.Asset):
    k = cfg.key  # ウィジェットキーの名前空間
    # Read optional fields defensively: on Streamlit Cloud a freshly pushed
    # app.py can briefly run against a not-yet-reloaded data.py module, so an
    # Asset instance may lack newer fields. getattr keeps the app from crashing.
    crosscheck = getattr(cfg, "crosscheck", ())
    context = getattr(cfg, "context", ())
    news_tickers = getattr(cfg, "news_tickers", ())
    news_query = getattr(cfg, "news_query", "")

    with st.expander("⚙️ モデル設定（期間・ファクター・窓）", expanded=False):
        cset1, cset2 = st.columns([1, 2])
        with cset1:
            start_choice = st.selectbox(
                "期間", options=["2018-01-01", "2015-01-01", "2010-01-01"],
                index=0, key=f"{k}_period",
            )
            zwin = st.slider("残差zスコアの窓（日）", 30, 252, 126, step=10, key=f"{k}_zwin")
            roll_win = st.slider("ローリングβの窓（日）", 60, 504, 252, step=20, key=f"{k}_roll")
            baseline_choice = st.radio(
                "寄与の基準", options=["mean", "1y_ago"],
                format_func=lambda x: "長期平均" if x == "mean" else "1年前",
                horizontal=True, key=f"{k}_base",
            )
        with cset2:
            factors = st.multiselect(
                "モデルに使うマクロファクター",
                options=D.factor_options(cfg),
                default=list(cfg.default_factors),
                format_func=D.label_ja,
                help="既定はマルチコリニアリティを抑えた推奨セット。自由に増減できます。",
                key=f"{k}_factors",
            )

    panel = load_panel(start_choice)
    factors = [f for f in factors if f in panel.columns and f != cfg.target]
    if not factors:
        st.error("有効なファクターがありません。期間を広げるか、ファクターを選び直してください。")
        return

    fit = M.fit_ols(panel, cfg.target, factors)
    mp = M.mispricing(panel, fit, cfg.target)
    contrib = M.contribution_breakdown(fit, panel, cfg.target, baseline=baseline_choice)

    def price(v):
        return f"{cfg.price_prefix}{v:,.{cfg.price_decimals}f}{cfg.price_suffix}"

    # --- KPI ---
    st.caption(
        f"最新データ：**{panel.index[-1].date()}**　·　"
        f"決定係数 R²：**{fit.r2:.3f}**　·　ファクター数：**{len(factors)}**"
    )
    latest = mp.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"現値（{cfg.name}）", price(latest.actual))
    c2.metric("理論値（フェアバリュー）", price(latest.fair),
              delta=f"{latest.resid_pct:+.2f}% 割高" if latest.resid_pct > 0
              else f"{latest.resid_pct:+.2f}% 割安")
    c3.metric("残差zスコア", f"{latest.z:+.2f} σ",
              help="|z|>2 で統計的な割高・割安の目安")
    signal = "🔴 割高" if latest.z > 2 else "🟢 割安" if latest.z < -2 else "⚪ 中立"
    c4.metric("シグナル（|z|>2）", signal)

    # --- なぜ動いたか + 解釈パネル ---
    st.subheader(f"なぜ{cfg.name}は動いたのか — 変化の寄与度分解")
    lb_label = st.radio(
        "対象期間", options=["1W", "1M", "3M", "6M", "1Y"], index=1, horizontal=True,
        format_func=lambda x: {"1W": "1週間", "1M": "1ヶ月", "3M": "3ヶ月",
                               "6M": "6ヶ月", "1Y": "1年"}[x],
        key=f"{k}_lookback",
    )
    lb_days = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}[lb_label]
    lb_jp = {"1W": "1週間", "1M": "1ヶ月", "3M": "3ヶ月", "6M": "6ヶ月", "1Y": "1年"}[lb_label]

    attr, summ = M.change_attribution(fit, panel, cfg.target, lookback=lb_days)

    st.info(build_narrative(cfg.name, summ, attr, float(latest.z), lb_jp))

    s1, s2, s3 = st.columns(3)
    s1.metric(f"実勢の変化（{lb_jp}）", f"{summ['actual_change_pct']:+.2f}%")
    s2.metric("理論値の変化", f"{summ['fair_change_pct']:+.2f}%")
    s3.metric("未説明分（残差）",
              f"{summ['actual_change_pct'] - summ['fair_change_pct']:+.2f}%",
              help="実勢 − 理論。大きいほどマクロ外の固有要因。")

    wfall = go.Figure(go.Bar(
        x=attr["contrib_pts"], y=jp(list(attr.index)), orientation="h",
        marker_color=["#d62728" if v < 0 else "#2ca02c" for v in attr["contrib_pts"]],
        text=[f"{v:+.2f}" for v in attr["contrib_pts"]], textposition="auto",
    ))
    wfall.update_layout(
        height=max(280, 44 * len(attr)),
        xaxis_title=f"理論値への寄与（{lb_jp}・%ポイント／合計＝理論値の変化）",
        margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(wfall, use_container_width=True, key=f"{k}_wfall")

    with st.expander("水準ベースの寄与（長期基準との比較・上級者向け）"):
        st.caption("今日の理論値『水準』を基準（長期平均 or 1年前）との差に分解します。"
                   "トレンドのある資産が大きく出るため、日々の物語は上の"
                   "『変化の分解』を参照してください。")
        contrib_disp = contrib.copy()
        contrib_disp.index = jp(list(contrib_disp.index))
        st.dataframe(contrib_disp.style.format({
            "beta": "{:+.4f}", "x_now": "{:.4f}", "x_base": "{:.4f}",
            "contrib_log": "{:+.4f}", "contrib_pct": "{:+.2f}%",
        }), use_container_width=True)

    # --- チャート1: 実勢 vs 理論値 + 残差 ---
    st.subheader("現値 vs 理論値、そして残差zスコア")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=mp.index, y=mp.actual, name="現値", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=mp.index, y=mp.fair, name="理論値", line=dict(width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=mp.index, y=mp.z, name="zスコア", line=dict(width=1.5)), row=2, col=1)
    fig.add_hline(y=2, line=dict(dash="dot", width=1, color="red"), row=2, col=1)
    fig.add_hline(y=-2, line=dict(dash="dot", width=1, color="green"), row=2, col=1)
    fig.add_hline(y=0, line=dict(width=1, color="gray"), row=2, col=1)
    fig.update_yaxes(title_text=cfg.unit, row=1, col=1)
    fig.update_yaxes(title_text="σ", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True, key=f"{k}_main")

    # --- チャート3: ローリングβ ---
    st.subheader(f"ローリングβ（{roll_win}日窓）— マクロ感応度の推移")
    rb = cached_rolling_beta(start_choice, cfg.target, tuple(factors), roll_win)
    rb_fig = go.Figure()
    for col in [c for c in rb.columns if c != "const"]:
        rb_fig.add_trace(go.Scatter(x=rb.index, y=rb[col], name=jp(col), mode="lines"))
    rb_fig.update_layout(height=420, hovermode="x unified", yaxis_title="β（感応度）",
                         legend=dict(orientation="h", y=1.1))
    st.plotly_chart(rb_fig, use_container_width=True, key=f"{k}_rb")

    # --- チャート4: 相関ヒートマップ ---
    st.subheader("日次変化の相関（直近252日）")
    ret = panel[[cfg.target, *factors]].diff().tail(252)
    corr = ret.corr()
    labels = jp(list(corr.columns))
    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}"))
    heat.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, use_container_width=True, key=f"{k}_heat")

    # --- クロスチェック: ベンチマークとのスプレッド（循環しない参考） ---
    if crosscheck and crosscheck[0] in panel.columns:
        bench_t, bench_l = crosscheck
        st.subheader(f"クロスチェック：{cfg.name} − {bench_l} スプレッド")
        cc = panel[[cfg.target, bench_t]].dropna()
        own = np.exp(cc[cfg.target]); bench = np.exp(cc[bench_t])
        spread = own - bench
        sp_now = float(spread.iloc[-1])
        st.caption(
            f"最新：{cfg.name} {price(float(own.iloc[-1]))} − {bench_l} "
            f"{cfg.price_prefix}{float(bench.iloc[-1]):,.{cfg.price_decimals}f}"
            f"{cfg.price_suffix} ＝ スプレッド {sp_now:+.1f}。"
            "スプレッドが平常域なら割安は『世界的』、大きく負なら『米国固有（供給過剰・在庫膨張）』のサイン。"
            "※ベンチマークはモデルの説明変数には入れない（原油で原油を説明＝循環のため）。"
        )
        ccfig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              row_heights=[0.6, 0.4])
        ccfig.add_trace(go.Scatter(x=cc.index, y=own, name=cfg.name, line=dict(width=2)), row=1, col=1)
        ccfig.add_trace(go.Scatter(x=cc.index, y=bench, name=bench_l, line=dict(width=2)), row=1, col=1)
        ccfig.add_trace(go.Scatter(x=cc.index, y=spread, name="スプレッド", line=dict(width=1.5)), row=2, col=1)
        ccfig.add_hline(y=0, line=dict(width=1, color="gray"), row=2, col=1)
        ccfig.update_yaxes(title_text=cfg.unit, row=1, col=1)
        ccfig.update_yaxes(title_text="差", row=2, col=1)
        ccfig.update_layout(height=460, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                            margin=dict(t=30))
        st.plotly_chart(ccfig, use_container_width=True, key=f"{k}_cc")

    # --- コンテキスト: 構造要因（貿易収支など）を価格と並べて表示 ---
    if context and context[0] in panel.columns:
        ctx_col, ctx_label = context
        st.subheader(f"構造の背景：{cfg.name} と {ctx_label}")
        cx = panel[[cfg.target, ctx_col]].dropna()
        st.caption(
            "為替の長期の地合いを作る構造要因の参考表示です（日次の値動きは金利差・"
            "リスク等が主役）。モデルにも小さな比重で組み込まれています。"
        )
        cxfig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              row_heights=[0.6, 0.4])
        cxfig.add_trace(go.Scatter(x=cx.index, y=np.exp(cx[cfg.target]),
                                   name=cfg.name, line=dict(width=2)), row=1, col=1)
        cxfig.add_trace(go.Scatter(x=cx.index, y=cx[ctx_col], name=ctx_label,
                                   line=dict(width=1.8, color="#FFB300")), row=2, col=1)
        cxfig.add_hline(y=0, line=dict(width=1, color="gray"), row=2, col=1)
        cxfig.update_yaxes(title_text=cfg.unit, row=1, col=1)
        cxfig.update_yaxes(title_text="兆円", row=2, col=1)
        cxfig.update_layout(height=460, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                            margin=dict(t=30))
        st.plotly_chart(cxfig, use_container_width=True, key=f"{k}_ctx")

    # --- 関連ニュース（英語=yfinance ＋ 日本語=Google News、新しい順） ---
    if news_tickers or news_query:
        st.subheader(f"📰 {cfg.name}の関連ニュース")
        news = load_news(tuple(news_tickers), news_query)
        if not news:
            st.caption("ニュースを取得できませんでした（時間をおいて再表示されます）。")
        else:
            for it in news[:12]:
                tag = "🇯🇵" if it["lang"] == "JP" else "🇺🇸"
                st.markdown(
                    f"{tag} `{rel_time(it['time'])}` · {it['source']}  \n"
                    f"[{it['title']}]({it['url']})"
                )
            st.caption("出典：Yahoo! Finance（英語）／ Google ニュース日本語版"
                       "（Yahoo!ニュース等を含む）。30分キャッシュ。")


# ---------------- ページ全体 ----------------

st.title("📊 マクロ理論値ダッシュボード")
st.caption("マクロ要因から各資産の理論価格を推定し、実勢との乖離（割安・割高）を可視化します。")

st.sidebar.title("このアプリについて")
st.sidebar.caption(
    "各タブの資産について、マクロ要因で理論価格を推定し、実勢との乖離を分析します。"
    "各タブ内の「⚙️ モデル設定」から期間・ファクター・窓を変更できます。\n\n"
    "データ：Yahoo Finance（遅延あり）。リサーチ・教育目的であり、投資助言ではありません。"
)

tab_gold, tab_oil, tab_jpy = st.tabs(["🪙 金", "🛢️ 原油(WTI)", "💴 ドル円"])
with tab_gold:
    render_dashboard(D.ASSETS["gold"])
with tab_oil:
    render_dashboard(D.ASSETS["oil"])
with tab_jpy:
    render_dashboard(D.ASSETS["jpy"])

with st.expander("計算方法（メソドロジー）"):
    st.markdown("""
**モデル.** 価格水準のOLS回帰： `log(資産) = α + Σ βᵢ · xᵢ + ε`。
多くのファクターは対数価格、金利は%。各資産ごとにドライバーを変えています
（金＝実質金利・ドル等、原油＝ドル・銅・株・中国需要等）。

**ミスプライス.** 対数空間の残差 ε ≒ %乖離。ローリング窓でzスコア化し、
レジーム依存の残差ボラを正規化します。|z| > 2 が統計的裁定の慣習的しきい値。

**変化の寄与.** 各ファクターの変化 × β で理論値の変化を要因分解。
合計＋未説明分（残差の変化）が実勢の変化に一致します。

**注意.** 系列が共和分でない場合、水準OLSは見せかけの回帰になりえます。
ローリングβで関係の安定性を確認してください。投資助言ではありません。
""")
