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
)

# ═══════════════════════════════════════════════════════════════════
# STOCK UNIVERSE
# ═══════════════════════════════════════════════════════════════════
# 10 highly liquid Bursa Malaysia stocks. Keys are display names used
# throughout the dashboard and passed to run_sentiment_agent() as the
# company_name for headline matching; values are Yahoo Finance tickers.

BURSA_10 = {
    "Maybank":            "1155.KL",
    "CIMB":                "1023.KL",
    "Public Bank":         "1295.KL",
    "TNB":                 "5347.KL",
    "Petronas Chemicals":  "5183.KL",
    "Maxis":               "6012.KL",
    "Axiata":              "6888.KL",
    "Genting":             "3182.KL",
    "KL Kepong":           "2445.KL",
    "IHH Healthcare":      "5225.KL",
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


# ═══════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════
# The engine is split into two independently cached stages so that
# model SELECTION (comparing XGBoost, LSTM and GRU) only ever runs
# once per stock, while model FORECASTING (retraining and running
# just the champion) refreshes on each session checkpoint.
#
#   Stage A - select_champion_model()
#       Runs walk-forward validation for all 3 candidate models and
#       picks the champion (lowest MAE). Cached to disk keyed ONLY by
#       symbol, so it is not repeated on every session or app restart.
#       It only reruns if its cache file is removed (see the
#       "Rebuild model selection" button in the sidebar).
#
#   Stage B - run_champion_forecast()
#       Retrains ONLY the already-selected champion model on the
#       latest available data and produces the 5-day forecast. Cached
#       to disk keyed by (symbol, session_label), so it refreshes once
#       per checkpoint (morning / half_day / end_of_day) as new data
#       is collected, without re-running the other two algorithms.
#
# TWO LAYERS for each stage:
#   1. st.cache_resource - in-memory, avoids recompute on Streamlit
#      reruns WITHIN the same running process (e.g. widget interaction).
#   2. Disk cache (joblib) - survives a full process restart (dev
#      server restart, redeploy, Streamlit Cloud sleep/wake).
#
# Neither stage returns a trained model object (XGBoost/Keras models
# are created and used internally, then discarded) - only plain
# numpy/dict/string results - so a lightweight joblib dump/load is
# enough; there is no need to serialize model binaries.

MODEL_SELECTION_CACHE_DIR = Path("model_selection_cache")
MODEL_SELECTION_CACHE_DIR.mkdir(exist_ok=True)

ENGINE_CACHE_DIR = Path("engine_cache")
ENGINE_CACHE_DIR.mkdir(exist_ok=True)


def _selection_cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace(".", "_")
    return MODEL_SELECTION_CACHE_DIR / f"{safe_symbol}.joblib"


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
def select_champion_model(symbol: str, _df):
    """
    Stage A - runs ONCE per stock, not once per session.

    Runs leakage-free walk-forward validation for all 3 base models
    (XGBoost, LSTM, GRU) and selects the champion (lowest MAE). This
    comparison is expensive and does not need to be repeated every
    time new data is collected; the champion architecture, once
    chosen, is reused going forward.

    Disk-cached per symbol only (no session_label in the key): if a
    selection already exists on disk, it is loaded instead of
    re-running all three algorithms. Use "Rebuild model selection" in
    the sidebar to force this to run again.
    """
    cache_path = _selection_cache_path(symbol)

    if cache_path.exists():
        try:
            return joblib.load(cache_path)
        except Exception:
            pass  # corrupt/incompatible cache file - fall through and rerun

    proc = StockPreprocessor()

    results = {}
    for mtype in ["XGBoost", "LSTM", "GRU"]:
        results[mtype] = walk_forward_validation(_df, proc, model_type=mtype, n_splits=5)

    champion = min(results, key=lambda k: results[k]["avg_mae"])

    output = (results, champion)
    try:
        joblib.dump(output, cache_path)
    except Exception:
        pass  # disk cache is a bonus, not required for the app to work

    return output


@st.cache_resource(show_spinner=False)
def run_champion_forecast(symbol: str, session_label: str, _df, champion: str):
    """
    Stage B - runs once per session checkpoint.

    Retrains ONLY the already-selected champion model on the latest
    available data and generates the 5-day forecast. The other two
    algorithms are not touched here; model selection is handled
    separately by select_champion_model() above.

    Note: prepare_data() fits a scaler on all available data - this is
    the one place that is appropriate, since the champion is being
    retrained on the full, latest dataset for forecasting rather than
    being evaluated fold-by-fold.

    Disk-cached per (symbol, session_label): if a checkpoint's result
    already exists on disk, it is loaded instead of retraining.
    """
    cache_path = _forecast_cache_path(symbol, session_label)

    if cache_path.exists():
        try:
            return joblib.load(cache_path)
        except Exception:
            pass  # corrupt/incompatible cache file - fall through and retrain

    proc = StockPreprocessor()
    X, y, scaled = proc.prepare_data(_df)
    champ = ChampionForecaster(proc)
    champ.fit(X, y, scaled, champion, raw_df=_df)
    forecast_5 = champ.forecast(n_days=5)

    output = (forecast_5, proc)
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
        results, champion = select_champion_model(symbol, df)
        fc5, proc = run_champion_forecast(symbol, session_label, df, champion)
        sentiment = load_sentiment(symbol, name, session_label)

        last_close = float(df["Close"].iloc[-1])
        exp_return = (float(fc5[-1]) - last_close) / last_close * 100
        sl_info    = stop_loss_advisory(last_close, fc5, 0.05)
        trend      = _trend_label(exp_return)
        sent_label = sentiment["sentiment_label"]
        action     = _suggested_action(
                         exp_return, sent_label, sl_info["breached"])
        best       = results[champion]

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
            "_champion":        champion,
            "_mae":             round(best["avg_mae"],    4),
            "_da":              round(best["avg_acc"]*100, 1),
            "_actuals_rm":      best["actuals_rm"],
            "_preds_rm":        best["predictions_rm"],
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
    "Model Evaluation and Explainability",
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

if st.sidebar.button("Rebuild model selection (all algorithms)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for f in ENGINE_CACHE_DIR.glob("*.joblib"):
        f.unlink()
    for f in MODEL_SELECTION_CACHE_DIR.glob("*.joblib"):
        f.unlink()
    st.rerun()
st.sidebar.caption(
    "Refresh forecast retrains only the selected champion model on "
    "the latest data. Rebuild model selection re-compares all three "
    "algorithms and should rarely be needed.")


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
        "All 10 stocks processed through the quantitative engine. "
        "Champion model selected per stock by lowest walk-forward MAE "
        "(leakage-free, per-fold scaling). Sentiment and its reason are "
        "generated locally by a Llama 3 agent reading live news and "
        "macro indicators. Suggested action combines forecast "
        "direction, sentiment, and stop-loss signal. Results refresh "
        "once per session checkpoint (morning / half-day / end-of-day).")

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
        results, champion = select_champion_model(symbol, df)
        fc5, proc = run_champion_forecast(symbol, session_label, df, champion)
        sentiment = load_sentiment(symbol, selected_name, session_label)

    last_close = float(df["Close"].iloc[-1])
    exp_return = (float(fc5[-1]) - last_close) / last_close * 100
    sl_info    = stop_loss_advisory(last_close, fc5, stop_loss_pct)
    sent_label = sentiment["sentiment_label"]
    action     = _suggested_action(
                     exp_return, sent_label, sl_info["breached"])
    best       = results[champion]
    da         = round(best["avg_acc"] * 100, 1)
    mae        = round(best["avg_mae"], 4)
    actuals_rm = best["actuals_rm"]
    preds_rm   = best["predictions_rm"]

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

    # Forecast continues from today
    x_fc = list(range(n_hist - 1, n_hist + len(fc5)))
    fc_y = np.concatenate([[hist_close[-1]], fc5])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_hist, y=hist_close,
        name="Actual Price",
        line=dict(color="#10b981", width=2)))

    fig.add_trace(go.Scatter(
        x=x_bt, y=preds_rm[-n_bt:],
        name=f"{champion} Back-test",
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Champion Model", champion)
    m2.metric("Directional Accuracy", f"{da}%",
              help="% of steps where model predicted correct direction "
                   "(leakage-free, RM-scale evaluation)")
    m3.metric("MAE", f"RM {mae}",
              help="Mean Absolute Error in Ringgit - lower is better")

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
        "Champion model selected by choosing the one with the lowest MAE across 5-fold walk-forward "
        "validation, with a fresh scaler fit per fold to avoid data "
        "leakage. Directional accuracy measures whether the model "
        "correctly predicted price direction (up/down) at each step, "
        "anchored to the previous actual close. Stop-loss threshold "
        "is adjustable in the sidebar.")

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