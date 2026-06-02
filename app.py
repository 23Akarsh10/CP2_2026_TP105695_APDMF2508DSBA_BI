"""
app.py  —  Stock Predictor  (Upgraded)
========================================
Streamlit dashboard wired to the upgraded engine.py.

New panels vs original
-----------------------
+ 5-Day Forecast panel with Plotly chart
+ Sharpe Ratio in leaderboard
+ Stop-Loss Advisory card
+ Per-headline sentiment table
+ Ensemble sentiment score
+ Portfolio Ranking tab (all 10 stocks)
+ Risk disclaimer footer
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from engine import (
    fetch_data, add_features, StockPreprocessor,
    walk_forward_validation, benchmark_sentiment,
    ChampionForecaster, stop_loss_advisory, build_ranking_table,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

BURSA_10 = {
    "Maybank":            "1155.KL",
    "Public Bank":        "1295.KL",
    "CIMB Group":         "1023.KL",
    "TNB":                "5347.KL",
    "Petronas Chemicals": "5183.KL",
    "IHH Healthcare":     "5225.KL",
    "Press Metal":        "8869.KL",
    "CelcomDigi":         "6947.KL",
    "MISC":               "3816.KL",
    "Hong Leong Bank":    "5819.KL",
}

MODEL_COLOURS = {
    "XGBoost": "#3b82f6",
    "LSTM":    "#f59e0b",
    "GRU":     "#8b5cf6",
}

SENTIMENT_COLOURS = {
    "Positive": "#10b981",
    "Neutral":  "#9ca3af",
    "Negative": "#ef4444",
}

st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

# ── Dark theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d2f39;
    }
    div[data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #2d2f39;
    }
    .forecast-card {
        background: #1a1c24;
        border: 1px solid #2d2f39;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .advisory-box {
        background: #1a1c24;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 0.95em;
    }
    .disclaimer {
        color: #6b7280;
        font-size: 0.78em;
        border-top: 1px solid #2d2f39;
        padding-top: 12px;
        margin-top: 30px;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.82em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load(symbol: str):
    df = fetch_data(symbol)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=False)
def cached_tournament(symbol: str, _df):
    """
    Runs walk-forward tournament for all 3 models, selects champion,
    trains ChampionForecaster on full data, returns everything.
    """
    proc             = StockPreprocessor()
    X, y, scaled     = proc.prepare_data(_df)
    results          = {}

    for mtype in ["XGBoost", "LSTM", "GRU"]:
        res = walk_forward_validation(X, y, model_type=mtype, n_splits=5)
        res["actuals_rm"] = proc.inverse_transform(res["actuals"])
        res["preds_rm"]   = proc.inverse_transform(res["predictions"])
        results[mtype]    = res

    # Champion = lowest MAE
    champion = min(results, key=lambda k: results[k]["avg_mae"])

    # Retrain champion on full data & forecast 5 days
    fc = ChampionForecaster(proc, epochs=15)
    fc.fit(X, y, scaled, champion)
    forecast_5 = fc.forecast(n_days=5)

    return results, champion, forecast_5, proc


@st.cache_data(ttl=1800, show_spinner=False)
def cached_sentiment(symbol: str):
    return benchmark_sentiment(symbol)


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO RANKING  (all 10 stocks, cached separately)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_ranking():
    summaries = {}
    for name, sym in BURSA_10.items():
        try:
            df               = cached_load(sym)
            results, champ, fc5, proc = cached_tournament(sym, df)
            sent             = cached_sentiment(sym)
            last_close       = float(df["Close"].iloc[-1])
            best             = results[champ]
            summaries[name]  = {
                "last_close":      last_close,
                "forecast":        fc5,
                "champion":        champ,
                "mae":             best["avg_mae"],
                "da":              best["avg_acc"],
                "sharpe":          best["avg_sharpe"],
                "sentiment_label": sent["E_Label"],
            }
        except Exception:
            pass    # skip failed tickers silently
    return build_ranking_table(summaries)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

st.sidebar.title("📈 Stock Predictor")
st.sidebar.markdown("---")

selected_name  = st.sidebar.selectbox("Select Bursa Corporate Ticker",
                                       list(BURSA_10.keys()))
ticker_symbol  = BURSA_10[selected_name]
stop_loss_pct  = st.sidebar.slider("Stop-Loss Threshold (%)", 1, 15, 5) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("CRISP-DM Cycle")
st.sidebar.success("✅ Business Understanding")
st.sidebar.success("✅ Data Understanding")
st.sidebar.success("✅ Data Preparation")
st.sidebar.warning("⏳ Modelling (Active)")
st.sidebar.info("⏳ Evaluation")
st.sidebar.info("⏳ Deployment")

st.sidebar.markdown("---")
st.sidebar.caption("Data sourced from Yahoo Finance via yfinance.\n"
                   "For educational / research use only.")


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_stock, tab_rank = st.tabs(
    [f"📊 {selected_name} Analysis", "🏆 Portfolio Ranking"]
)


# ───────────────────────────────────────────────────────────────────
# TAB 1 — INDIVIDUAL STOCK ANALYSIS
# ───────────────────────────────────────────────────────────────────
with tab_stock:

    st.header(f"Stock Predictor — {selected_name} ({ticker_symbol})")

    # ── Load data ────────────────────────────────────────────────────
    with st.spinner(f"Downloading & processing {selected_name} …"):
        df = cached_load(ticker_symbol)

    with st.spinner("Running Model Tournament (LSTM · GRU · XGBoost) …"):
        results, champion, forecast_5, proc = cached_tournament(
            ticker_symbol, df)

    with st.spinner("Fetching live news & analysing sentiment …"):
        sentiment = cached_sentiment(ticker_symbol)

    last_price = float(df["Close"].iloc[-1])
    best_res   = results[champion]
    sl_info    = stop_loss_advisory(last_price, forecast_5, stop_loss_pct)

    # ── Row 1: Key metrics ───────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Price", f"RM {last_price:.4f}")
    with c2:
        st.metric("Tournament Winner", champion,
                  f"MAE: {best_res['avg_mae']:.4f}")
    with c3:
        dir_acc = best_res["avg_acc"] * 100
        st.metric("Directional Accuracy", f"{dir_acc:.1f}%")
    with c4:
        st.metric("Sharpe Ratio", f"{best_res['avg_sharpe']:.3f}")

    st.markdown("---")

    # ── Row 2: Leaderboard ───────────────────────────────────────────
    st.subheader("📊 Model Performance Benchmark (Walk-Forward Validation)")
    lb_rows = []
    for mtype, res in results.items():
        flag = " 🏆" if mtype == champion else ""
        lb_rows.append({
            "Model":                  mtype + flag,
            "Mean Absolute Error":    f"{res['avg_mae']:.4f}",
            "Directional Acc (%)":    f"{res['avg_acc']*100:.2f}%",
            "Sharpe Ratio":           f"{res['avg_sharpe']:.3f}",
        })
    st.table(pd.DataFrame(lb_rows))

    st.markdown("---")

    # ── Row 3: Backtest chart ────────────────────────────────────────
    st.subheader("📈 Backtest Tracker: Actual vs Predictions")
    fig_bt = go.Figure()

    actuals_rm = results["LSTM"]["actuals_rm"]    # same for all models
    fig_bt.add_trace(go.Scatter(
        y=actuals_rm, name="Actual Market Price",
        line=dict(color="#10b981", width=3)))

    for mtype, res in results.items():
        fig_bt.add_trace(go.Scatter(
            y=res["preds_rm"], name=f"{mtype} Prediction",
            line=dict(color=MODEL_COLOURS[mtype], dash="dot")))

    fig_bt.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Validation Steps", yaxis_title="Price (RM)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("---")

    # ── Row 4: 5-Day Forecast ────────────────────────────────────────
    st.subheader(f"🔭 5-Day Price Forecast  —  Champion: {champion}")

    day_cols = st.columns(5)
    for i, (col, price) in enumerate(zip(day_cols, forecast_5)):
        delta     = price - last_price
        delta_pct = delta / last_price * 100
        arrow     = "▲" if delta >= 0 else "▼"
        colour    = "#10b981" if delta >= 0 else "#ef4444"
        col.markdown(
            f"<div class='forecast-card'>"
            f"<div style='color:#9ca3af;font-size:0.8em'>Day {i+1}</div>"
            f"<div style='font-size:1.25em;font-weight:700'>RM {price:.4f}</div>"
            f"<div style='color:{colour};font-size:0.85em'>"
            f"{arrow} {delta_pct:+.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Forecast line chart
    st.markdown("")
    hist_window = df["Close"].iloc[-30:].values
    x_hist      = list(range(len(hist_window)))
    x_fc        = list(range(len(hist_window) - 1,
                             len(hist_window) - 1 + len(forecast_5) + 1))
    fc_y        = np.concatenate([[hist_window[-1]], forecast_5])

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=x_hist, y=hist_window,
        name="Last 30 Days", line=dict(color="#10b981", width=2)))
    fig_fc.add_trace(go.Scatter(
        x=x_fc, y=fc_y,
        name="5-Day Forecast", line=dict(color="#f59e0b", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=6)))
    fig_fc.add_vline(x=len(hist_window) - 1,
                     line_dash="dash", line_color="#6b7280",
                     annotation_text="Today", annotation_position="top")
    fig_fc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Trading Days", yaxis_title="Price (RM)",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ── Row 5: Stop-Loss Advisory ────────────────────────────────────
    st.subheader(f"🛡️ Stop-Loss Advisory  ({stop_loss_pct*100:.0f}% threshold)")

    sl_col1, sl_col2, sl_col3 = st.columns(3)
    sl_col1.metric("Entry Price",     f"RM {last_price:.4f}")
    sl_col2.metric("Stop-Loss Level", f"RM {sl_info['threshold']:.4f}",
                   f"-{stop_loss_pct*100:.0f}%")
    peak_fc  = float(np.max(forecast_5))
    exp_ret  = (peak_fc - last_price) / last_price * 100
    sl_col3.metric("Peak 5-Day Forecast", f"RM {peak_fc:.4f}",
                   f"{exp_ret:+.2f}%")

    advisory_colour = "#ef4444" if sl_info["breached"] else "#10b981"
    st.markdown(
        f"<div class='advisory-box' style='border-left-color:{advisory_colour}'>"
        f"{sl_info['advisory']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Row 6: Sentiment ─────────────────────────────────────────────
    st.subheader("🧠 Sentiment Analysis: VADER vs TextBlob")

    sc1, sc2, sc3 = st.columns(3)
    for col, label, score, engine_lbl in [
        (sc1, sentiment["V_Label"], sentiment["VADER"],    "VADER"),
        (sc2, sentiment["T_Label"], sentiment["TextBlob"], "TextBlob"),
        (sc3, sentiment["E_Label"], sentiment["Ensemble"], "Ensemble"),
    ]:
        badge_col = SENTIMENT_COLOURS.get(label, "#9ca3af")
        col.markdown(
            f"**{engine_lbl}** &nbsp;"
            f"<span class='sentiment-badge' "
            f"style='background:{badge_col};color:white'>{label}</span>",
            unsafe_allow_html=True,
        )
        col.progress(max(0.0, min(1.0, (score + 1) / 2)))
        col.caption(f"Score: {score:+.4f}")

    # Per-headline breakdown
    if sentiment["headlines"]:
        with st.expander(f"📰 Headline Breakdown  ({sentiment['n']} articles)"):
            hl_rows = []
            for h in sentiment["headlines"]:
                badge_col = SENTIMENT_COLOURS.get(h["label"], "#9ca3af")
                hl_rows.append({
                    "Headline":  h["title"][:90] + ("…" if len(h["title"]) > 90 else ""),
                    "VADER":     h["vader"],
                    "TextBlob":  h["textblob"],
                    "Avg":       h["avg"],
                    "Label":     h["label"],
                })
            st.dataframe(
                pd.DataFrame(hl_rows),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("No recent headlines found for this ticker.")

    # Risk cross-check callout
    if (sentiment["E_Label"] == "Negative"
            and exp_ret > 0
            and not sl_info["breached"]):
        st.warning(
            "⚡ **Signal conflict detected:** The model forecasts a price rise, "
            "but current news sentiment is Negative. Treat this trade with extra caution.")

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <strong>Disclaimer:</strong> Stock Predictor is a research and "
        "educational Decision Support System. It does <strong>not</strong> "
        "constitute financial advice. All forecasts carry uncertainty. "
        "Past model performance is not indicative of future results. "
        "Always conduct independent due diligence before investing."
        "</div>",
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────
# TAB 2 — PORTFOLIO RANKING
# ───────────────────────────────────────────────────────────────────
with tab_rank:

    st.header("🏆 Portfolio Ranking — All 10 Bursa Stocks")
    st.caption("Ranked by expected 5-day return. "
               "Results cached for 1 hour; first load may take a few minutes.")

    with st.spinner("Running full market tournament across all 10 stocks …"):
        rank_df = cached_ranking()

    if rank_df.empty:
        st.error("Could not retrieve data for any stock. "
                 "Check your internet connection.")
    else:
        # Colour-code Expected Return column
        def colour_return(val):
            colour = "#10b981" if val >= 0 else "#ef4444"
            return f"color: {colour}; font-weight: bold"

        def colour_sentiment(val):
            colour = SENTIMENT_COLOURS.get(val, "#9ca3af")
            return f"color: {colour}; font-weight: 600"

        styled = (
            rank_df.style
            .applymap(colour_return,   subset=["Expected Return"])
            .applymap(colour_sentiment, subset=["Sentiment"])
        )
        st.dataframe(styled, use_container_width=True)

        # Bar chart: expected return
        fig_rank = go.Figure(go.Bar(
            x=rank_df["Stock"],
            y=rank_df["Expected Return"],
            marker_color=[
                "#10b981" if r >= 0 else "#ef4444"
                for r in rank_df["Expected Return"]
            ],
            text=[f"{r:+.2f}%" for r in rank_df["Expected Return"]],
            textposition="outside",
        ))
        fig_rank.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Stock", yaxis_title="Expected 5-Day Return (%)",
            title="Expected 5-Day Return by Stock",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        # Sentiment distribution pie
        sent_counts = rank_df["Sentiment"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=sent_counts.index,
            values=sent_counts.values,
            marker_colors=[SENTIMENT_COLOURS.get(l, "#9ca3af")
                           for l in sent_counts.index],
            hole=0.4,
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=0, r=0, t=30, b=0),
            title="Market Sentiment Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ Rankings are model outputs for research purposes only. "
        "Not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )
