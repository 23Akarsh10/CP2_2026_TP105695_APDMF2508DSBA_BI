import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

from engine_final import (
    fetch_data,
    add_features,
    StockPreprocessor,
    walk_forward_validation,
    ChampionForecaster,
    run_sentiment_agent,
    stop_loss_advisory,
    get_current_session_label,
    BURSA_10,
    CONFIG,
)


CHAMPION_MODEL = "XGBoost"
CHAMPION_PARAMS = {
    "subsample": 0.7,
    "reg_lambda": 2.0,
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.01,
    "colsample_bytree": 1.0
}

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

LOGO_PATH = Path(__file__).parent / "logo.png"

st.set_page_config(
    page_title="Decision Support System for Stock Markets",
    layout="wide",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #2d2f39;
    }
    .metric-card {
        background: #1a1c24;
        border: 1px solid #2d2f39;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .disclaimer {
        color: #6b7280;
        font-size: 0.78em;
        border-top: 1px solid #2d2f39;
        padding-top: 12px;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TITLE BAR
# ═══════════════════════════════════════════════════════════════════

header_logo_col, header_title_col = st.columns([1, 6])
with header_logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with header_title_col:
    st.title("Decision Support System for Stock Markets")
    st.caption("Bursa Malaysia Top 10 - Quantitative Forecasting and Market Intelligence")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

SENTIMENT_COLOUR = {
    "Positive": "#10b981",
    "Neutral":  "#9ca3af",
    "Negative": "#ef4444",
}

ACTION_COLOUR = {
    "Buy":  "#10b981",
    "Sell": "#ef4444",
    "Hold": "#f59e0b",
}


def _suggested_action(exp_return: float, sentiment: str,
                       stop_breached: bool) -> str:
    """
    Derives a suggested action from three signals:
      - Expected 5-day return (quantitative)
      - Sentiment label (qualitative, from the local Llama 3 agent)
      - Stop-loss breach flag (risk management)

    Rules:
      If stop-loss is breached                        : Sell
      If return > 1% and sentiment is Positive        : Buy
      If return < -1% or sentiment is Negative        : Sell
      Otherwise                                        : Hold
    """
    if stop_breached:
        return "Sell"
    if exp_return > 1.0 and sentiment == "Positive":
        return "Buy"
    if exp_return < -1.0 or sentiment == "Negative":
        return "Sell"
    return "Hold"


def _trend_label(exp_return: float) -> str:
    """Bullish if expected return is positive, Bearish if negative."""
    return "Bullish" if exp_return >= 0 else "Bearish"

ENGINE_CACHE_DIR = Path("engine_cache")
ENGINE_CACHE_DIR.mkdir(exist_ok=True)


def _forecast_cache_path(symbol: str, session_label: str) -> Path:
    safe_symbol = symbol.replace(".", "_")
    return ENGINE_CACHE_DIR / f"{safe_symbol}__{session_label}.joblib"


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(symbol: str, session_label: str):
    """Downloads and feature-engineers data for one stock."""
    df = fetch_data(symbol)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=False)
def run_champion_forecast(symbol: str, session_label: str, _df):
    cache_path = _forecast_cache_path(symbol, session_label)

    if cache_path.exists():
        try:
            cached = joblib.load(cache_path)
            # Guards against a stale cache file left over from an older,
            # incompatible version of this pipeline (e.g. the previous
            # recursive-forecast design) - a mismatched shape here means
            # the cache predates the current output format, not that
            # anything is actually wrong with the data.
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached
        except Exception:
            pass  # corrupt/incompatible cache file - fall through and retrain

    proc = StockPreprocessor()
    X, y, scaled = proc.prepare_data(_df)
    champ = ChampionForecaster(proc)
    champ.fit(X, y, scaled, CHAMPION_MODEL, raw_df=_df, model_params=CHAMPION_PARAMS)
    forecast_5 = champ.forecast(n_days=5)

    wf_result = walk_forward_validation(
        _df, StockPreprocessor(), model_type=CHAMPION_MODEL,
        n_splits=5, model_params=CHAMPION_PARAMS)

    output = (forecast_5, proc, wf_result)
    try:
        joblib.dump(output, cache_path)
    except Exception:
        pass  # disk cache is a bonus, not required for the app to work

    return output


@st.cache_data(ttl=1800, show_spinner=False)
def load_sentiment(symbol: str, company_name: str, session_label: str):
    """
    Runs the local AI sentiment agent (Ollama + Llama 3) for one stock.
    Falls back to a documented Neutral response if no headlines are
    found or Ollama is unreachable — see engine.run_sentiment_agent().
    """
    return run_sentiment_agent(symbol, company_name)


# ═══════════════════════════════════════════════════════════════════
# BUILD MARKET OVERVIEW ROW FOR ONE STOCK
# ═══════════════════════════════════════════════════════════════════

def build_stock_row(idx: int, name: str, symbol: str,
                    session_label: str) -> dict | None:
    """
    Runs the full pipeline for one stock and returns a dict
    with all columns needed for the Market Overview table.
    Returns None if data cannot be fetched.
    """
    try:
        df = load_stock_data(symbol, session_label)
        fc5, proc, wf_result = run_champion_forecast(symbol, session_label, df)
        sentiment = load_sentiment(symbol, name, session_label)

        last_close = float(df["Close"].iloc[-1])
        exp_return = (float(fc5[-1]) - last_close) / last_close * 100
        sl_info    = stop_loss_advisory(last_close, fc5, 0.05)
        trend      = _trend_label(exp_return)
        sent_label = sentiment["sentiment_label"]
        action     = _suggested_action(
                         exp_return, sent_label, sl_info["breached"])

        return {
            "No.":              idx,
            "Stock":            name,
            "Last Close (RM)":  round(last_close, 4),
            "Expected Return":  round(exp_return, 2),
            "Trend":            trend,
            "Sentiment":        sent_label,
            "Sentiment Reason": sentiment["qualitative_rationale"],
            "Suggested Action": action,
            # Hidden — used in deep dive tab
            "_symbol":          symbol,
            "_forecast":        fc5,
            "_mae":             round(wf_result["avg_mae"],    4),
            "_da":              round(wf_result["avg_acc"]*100, 1),
            "_actuals_rm":      wf_result["actuals_rm"],
            "_preds_rm":        wf_result["predictions_rm"],
            "_sl_threshold":    sl_info["threshold"],
            "_sl_breached":     sl_info["breached"],
            "_sl_advisory":     sl_info["advisory"],
            "_df":              df,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# SESSION CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

session_label = get_current_session_label()


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=110)
st.sidebar.title("Decision Support System")
st.sidebar.markdown("---")
selected_name = st.sidebar.selectbox(
    "Select stock for deep dive",
    list(BURSA_10.keys()),
)
stop_loss_pct = st.sidebar.slider(
    "Stop-Loss Threshold (%)", 1, 15, 5) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Project Pipeline")
pipeline_stages = [
    "Data Collection",
    "Data Preprocessing and Feature Engineering",
    "Dataset Preparation",
    "Model Development and Hyperparameter Optimisation",
    "Model Evaluation and Selection",
    "Model Selection and Forecasting",
    "AI Based Market Intelligence",
    "Investment Decision Support",
]
st.sidebar.markdown(
    "\n".join(f"{i}. {stage}" for i, stage in enumerate(pipeline_stages, start=1))
)
st.sidebar.markdown("---")

session_caption = {
    "morning":    "Session: Morning (before 1:00 PM)",
    "half_day":   "Session: Half-Day (1:00 PM to 5:00 PM)",
    "end_of_day": "Session: End-of-Day (after 5:00 PM)",
}
st.sidebar.caption(session_caption.get(session_label, session_label))
st.sidebar.caption(
    "Data sourced from Yahoo Finance. Sentiment generated locally via "
    "Ollama and Llama 3. For research and educational use only.")

if st.sidebar.button("Refresh forecast (new session data)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for f in ENGINE_CACHE_DIR.glob("*.joblib"):
        f.unlink()
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_overview, tab_dive = st.tabs([
    "Market Overview",
    f"{selected_name} - Deep Dive",
])


# ───────────────────────────────────────────────────────────────────
# TAB 1 - MARKET OVERVIEW TABLE
# ───────────────────────────────────────────────────────────────────
with tab_overview:

    st.header("Market Overview - Bursa Malaysia Top 10")
    st.caption(
        f"All 10 stocks processed through the quantitative engine using "
        f"the champion model ({CHAMPION_MODEL}), selected offline via a "
        f"manual tournament that hyperparameter-tuned and compared "
        f"XGBoost, LSTM and GRU on leakage-free walk-forward MAE, "
        f"Directional Accuracy, and Sharpe Ratio (see run_tournament.py). "
        f"Sentiment and its reason are generated "
        f"locally by a Llama 3 agent reading live news and macro "
        f"indicators. Suggested action combines forecast direction, "
        f"sentiment, and stop-loss signal. Results refresh once per "
        f"session checkpoint (morning / half-day / end-of-day).")

    rows        = []
    stock_cache = {}

    progress = st.progress(0, text="Loading market data...")
    for i, (name, symbol) in enumerate(BURSA_10.items()):
        progress.progress(
            (i + 1) / len(BURSA_10),
            text=f"Processing {name}...")
        row = build_stock_row(i + 1, name, symbol, session_label)
        if row:
            stock_cache[name] = row
            rows.append({
                "Sl. No.":           row["No."],
                "Stock Name":        row["Stock"],
                "Expected Return %": row["Expected Return"],
                "Trend":             row["Trend"],
                "Sentiment":         row["Sentiment"],
                "Sentiment Reason":  row["Sentiment Reason"],
                "Suggested Action":  row["Suggested Action"],
            })
    progress.empty()

    if rows:
        overview_df = pd.DataFrame(rows).set_index("Sl. No.")

        # Style the table
        def colour_action(val):
            c = ACTION_COLOUR.get(val, "#9ca3af")
            return f"color: {c}; font-weight: bold"

        def colour_sentiment(val):
            c = SENTIMENT_COLOUR.get(val, "#9ca3af")
            return f"color: {c}; font-weight: 600"

        def colour_return(val):
            c = "#10b981" if val >= 0 else "#ef4444"
            return f"color: {c}; font-weight: bold"

        styled = (
            overview_df.style
            .map(colour_return,    subset=["Expected Return %"])
            .map(colour_sentiment, subset=["Sentiment"])
            .map(colour_action,    subset=["Suggested Action"])
        )
        st.dataframe(
            styled,
            use_container_width=True,
            column_config={
                "Sentiment Reason": st.column_config.TextColumn(
                    "Sentiment Reason", width="large"),
            },
        )
    else:
        st.error("Could not load any stock data. "
                 "Check your internet connection.")

    st.markdown(
        "<div class='disclaimer'>"
        "Suggested actions are model outputs for research purposes "
        "only. This is not financial advice."
        "<br><br>"
        "Developed by Akarsh Sriram (Student ID: TP105695), under the "
        "guidance of Dr. Preethi Subramanian, as part of the MSc Data "
        "Science and Business Analytics programme, Intake "
        "APDMF2508DSBA(BI), Asia Pacific University of Technology and "
        "Innovation (APU), in collaboration with De Montfort University."
        "</div>",
        unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# TAB 2 - STOCK DEEP DIVE
# ───────────────────────────────────────────────────────────────────
with tab_dive:

    st.header(f"{selected_name} - Deep Dive")

    symbol = BURSA_10[selected_name]

    with st.spinner(f"Running engine for {selected_name}..."):
        df = load_stock_data(symbol, session_label)
        fc5, proc, wf_result = run_champion_forecast(symbol, session_label, df)
        sentiment = load_sentiment(symbol, selected_name, session_label)

    last_close = float(df["Close"].iloc[-1])
    exp_return = (float(fc5[-1]) - last_close) / last_close * 100
    sl_info    = stop_loss_advisory(last_close, fc5, stop_loss_pct)
    sent_label = sentiment["sentiment_label"]
    action     = _suggested_action(
                     exp_return, sent_label, sl_info["breached"])
    da         = round(wf_result["avg_acc"] * 100, 1)
    mae        = round(wf_result["avg_mae"], 4)
    sharpe     = round(wf_result["avg_sharpe"], 3)
    actuals_rm = wf_result["actuals_rm"]
    preds_rm   = wf_result["predictions_rm"]

    # -- Section 1: Projected vs Actual line chart -----------------
    st.subheader("Projected vs Actual Prices")
    st.caption(
        "The chart below shows how closely the champion model's "
        "back-test predictions tracked actual market prices during "
        "walk-forward validation, followed by the 5-day forecast.")

    # Historical window: last 60 actual prices for context
    hist_close = df["Close"].iloc[-60:].values
    n_hist     = len(hist_close)
    x_hist     = list(range(n_hist))

    # Back-test predictions aligned to the end of history
    n_bt = min(len(actuals_rm), 60)
    x_bt = list(range(n_hist - n_bt, n_hist))

    # Forecast is now a single DIRECT forecast_horizon-day-ahead price
    # (ChampionForecaster no longer produces a day-by-day recursive
    # trajectory - see engine_final.py), so the chart draws one line
    # segment from today straight to that horizon point, rather than
    # plotting intermediate days that were never actually predicted.
    x_fc = [n_hist - 1, n_hist - 1 + CONFIG.forecast_horizon]
    fc_y = [hist_close[-1], float(fc5[-1])]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_hist, y=hist_close,
        name="Actual Price",
        line=dict(color="#10b981", width=2)))

    fig.add_trace(go.Scatter(
        x=x_bt, y=preds_rm[-n_bt:],
        name=f"{CHAMPION_MODEL} Back-test",
        line=dict(color="#f59e0b", width=1.5, dash="dot")))

    fig.add_trace(go.Scatter(
        x=x_fc, y=fc_y,
        name="5-Day Forecast",
        line=dict(color="#3b82f6", width=2),
        mode="lines+markers",
        marker=dict(size=6)))

    fig.add_vline(
        x=n_hist - 1,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text="Today",
        annotation_position="top")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Trading Days",
        yaxis_title="Price (RM)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Section 2: Model info widget -------------------------------
    st.subheader("Model Information")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champion Model", CHAMPION_MODEL)
    m2.metric("Directional Accuracy", f"{da}%",
              help="% of steps where model predicted correct direction "
                   "(leakage-free, RM-scale evaluation). Also used, "
                   "alongside MAE and Sharpe Ratio, when selecting the "
                   "champion architecture offline in run_tournament.py.")
    m3.metric("MAE", f"RM {mae}",
              help="Mean Absolute Error in Ringgit - lower is better. "
                   "Also used, alongside Directional Accuracy and Sharpe "
                   "Ratio, when selecting the champion architecture "
                   "offline in run_tournament.py.")
    m4.metric("Sharpe Ratio", f"{sharpe}",
              help="Annualised risk-adjusted return of a long/short "
                   "strategy driven by the model's directional signal "
                   "(approximation - ignores transaction costs). Also "
                   "used, alongside MAE and Directional Accuracy, when "
                   "selecting the champion architecture offline in "
                   "run_tournament.py.")

    with st.expander("Champion Hyperparameters"):
        st.caption(
            f"{CHAMPION_MODEL} is the hardcoded champion architecture. "
            "Its hyperparameters below were found via a manual, offline "
            "tournament (Optuna, falling back to RandomizedSearchCV if "
            "Optuna is unavailable) that compared XGBoost, LSTM and GRU "
            "on leakage-safe walk-forward validation, selecting the "
            "champion on a weighted composite of MAE, Directional "
            "Accuracy, and Sharpe Ratio - "
            "see run_tournament.py. That comparison is not repeated here; "
            "the app only ever trains and runs this one architecture.")
        st.markdown(f"**{CHAMPION_MODEL}**")
        st.json(CHAMPION_PARAMS)

    # Loss prevention rules
    st.markdown("**Loss Prevention Rules**")

    action_colour = ACTION_COLOUR.get(action, "#9ca3af")
    sl_colour     = "#ef4444" if sl_info["breached"] else "#10b981"

    st.markdown(
        f"<div class='metric-card'>"
        f"<b>Suggested Action: </b>"
        f"<span style='color:{action_colour};font-weight:700;"
        f"font-size:1.1em'>{action}</span><br><br>"
        f"<b>Stop-Loss Level ({stop_loss_pct*100:.0f}%):</b> "
        f"RM {sl_info['threshold']:.4f}<br>"
        f"<span style='color:{sl_colour}'>{sl_info['advisory']}</span>"
        f"</div>",
        unsafe_allow_html=True)

    st.caption(
        f"{CHAMPION_MODEL} is hardcoded as the champion, chosen offline "
        f"via a manual tournament (run_tournament.py) that compared it "
        f"against LSTM and GRU on 5-fold walk-forward MAE, Directional "
        f"Accuracy, and Sharpe Ratio, with a fresh scaler fit per fold "
        f"to avoid data leakage. "
        f"The metrics above are from validating just this champion on "
        f"the current stock. Directional accuracy measures whether the "
        f"model correctly predicted price direction (up/down) at each "
        f"step, anchored to the previous actual close. Sharpe Ratio is "
        f"the annualised risk-adjusted return of a signal driven by "
        f"that direction. Stop-loss threshold is adjustable in the "
        f"sidebar.")

    # -- Section 3: Sentiment (local Llama 3 agent) -----------------
    st.markdown("---")
    st.subheader("News and Macro Sentiment (Local AI Agent)")

    badge_col = SENTIMENT_COLOUR.get(sent_label, "#9ca3af")
    st.markdown(
        f"**Overall Sentiment** &nbsp;"
        f"<span style='background:{badge_col};color:white;"
        f"padding:3px 10px;border-radius:12px;"
        f"font-size:0.85em'>{sent_label}</span>",
        unsafe_allow_html=True)

    st.markdown(f"**Macro Impact:** {sentiment['macro_impact']}")
    st.markdown(f"**Rationale:** {sentiment['qualitative_rationale']}")

    st.caption(
        "Generated locally by a Llama 3 agent (via Ollama) reading "
        "live RSS headlines and Malaysia's OPR, CPI and GDP growth "
        "indicators. No paid API used. If no matching headlines are "
        "found, a documented Neutral default is shown instead of a "
        "guess.")

    # Signal conflict warning
    if sent_label == "Negative" and exp_return > 0:
        st.warning(
            "Signal conflict: the model forecasts a price rise "
            "but current news sentiment is Negative. "
            "Treat this signal with extra caution.")

    # -- Disclaimer --------------------------------------------------
    st.markdown(
        "<div class='disclaimer'>"
        "<strong>Disclaimer:</strong> This Decision Support System is "
        "a research and educational project. It does "
        "<strong>not</strong> constitute financial advice. All "
        "forecasts carry uncertainty. Past model performance is not "
        "indicative of future results."
        "<br><br>"
        "Developed by Akarsh Sriram (Student ID: TP105695), under the "
        "guidance of Dr. Preethi Subramanian, as part of the MSc Data "
        "Science and Business Analytics programme, Intake "
        "APDMF2508DSBA(BI), Asia Pacific University of Technology and "
        "Innovation (APU), in collaboration with De Montfort University."
        "</div>",
        unsafe_allow_html=True)