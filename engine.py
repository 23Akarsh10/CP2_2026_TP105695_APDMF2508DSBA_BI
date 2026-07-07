import logging
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import yfinance as yf
import tensorflow as tf
import feedparser
import json
import re
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from langchain_community.llms import Ollama

from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Optional, Callable, Dict, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

@dataclass
class EngineConfig:
    """
    Single source of truth for every previously hardcoded
    parameter. Pass a custom instance to any function via its
    config= argument, or edit the module-level CONFIG once at
    start-up to change behaviour everywhere at once.
    """
    # ── preprocessing ──────────────────────────────────────────
    window_size: int = 60

    # ── feature-engineering windows ────────────────────────────
    ma_short: int = 20
    ma_long: int = 50
    ema_span: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    vol_window: int = 20
    atr_period: int = 14

    # ── training ────────────────────────────────────────────────
    epochs: int = 10
    champion_epochs: int = 15
    batch_size: int = 32
    patience: int = 3
    validation_split: float = 0.1
    n_splits: int = 5

    # ── forecasting ─────────────────────────────────────────────
    forecast_horizon: int = 5
    range_lookback: int = 20

    # ── risk ────────────────────────────────────────────────────
    stop_pct: float = 0.05

    # ── sentiment ───────────────────────────────────────────────
    rss_feeds: tuple = (
        "https://www.theedgemalaysia.com/rss.html",  # The Edge Malaysia
        "https://www.bernama.com/en/rssfeed.php",    # Bernama
        "https://www.freemalaysiatoday.com/feeds/rss/headlines/",  # FMT
        "https://www.malaymail.com/feed/rss/malaysia",             # Malay Mail
        "https://thesun.my/feed/",                                 # The Sun
    )
    max_headlines: int = 8
    fuzzy_threshold: float = 0.82
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    # Set this to a callable returning {OPR, CPI, GDP_Growth}
    # to pull live macro data; None keeps the static placeholders.
    macro_provider: Optional[Callable[[], Dict[str, float]]] = None

    # ── refresh schedule ────────────────────────────────────────
    half_day_cutoff: dtime = dtime(13, 0)
    end_of_day_cutoff: dtime = dtime(17, 0)


CONFIG = EngineConfig()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Fixed random seeds for reproducibility.
# All three must be set to guarantee consistent results
# across repeated executions.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# Trained models are saved here so the app does not need
# to retrain from scratch on every session.
MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# The 10 highly liquid Bursa Malaysia stocks.
# Selected on the basis of high average daily trade volume.
# Section 3.4 of the methodology.
# The 10 highly liquid Bursa Malaysia stocks.
# Mapped using numeric stock codes for universal yfinance API stability.
BURSA_10 = {
    "Maybank":            "1155.KL",
    "Public Bank":        "1295.KL",
    "CIMB Group":         "1023.KL",
    "TNB":                "5347.KL",
    "Petronas Chemicals": "5183.KL",
    "IHH Healthcare":     "5225.KL",
    "Maxis Berhad":       "6012.KL",
    "Axiata Group":       "6888.KL",
    "Genting Berhad":     "3182.KL",
    "KL Kepong":          "2445.KL",
}

def fetch_data(symbol: str, period: str = "5y") -> pd.DataFrame:
    """
    Downloads 5 years of daily OHLCV data from Yahoo Finance.

    Applies forward imputation (ffill) to fill gaps caused
    by public holidays and market closure days. This preserves
    temporal continuity without introducing simulated prices.

    symbol : Yahoo Finance ticker e.g. '1155.KL' for Maybank
    period : how far back to go. '5y' = 5 years of daily data.
    """
    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    # Forward fill: copy the last valid price into any gap day.
    df = df.ffill()

    return df

def compute_indicators(df: pd.DataFrame,
                       config: EngineConfig = CONFIG) -> pd.DataFrame:
    """
    Single shared indicator engine used by BOTH add_features()
    and the recursive ChampionForecaster, guaranteeing the two
    never drift apart.

    All indicators are causal — each row uses only current and
    past values, never future ones — so computing them once over
    the whole series introduces no look-ahead. The only leakage
    concern in this project was scaling, fixed separately in
    walk_forward_validation() in the next step.

    Adds: Returns, MA20, MA50, EMA20, RSI(14),
    MACD(12,26)+Signal(9), Bollinger Bands(20, 2 std-dev),
    Volatility(20), ATR(14) and Volume.
    """
    df = df.copy()
    close = df["Close"]

    # Daily returns + moving averages
    df["Returns"] = close.pct_change()
    df["MA20"]    = close.rolling(config.ma_short).mean()
    df["MA50"]    = close.rolling(config.ma_long).mean()
    df["EMA20"]   = close.ewm(span=config.ema_span, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(config.rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(config.rsi_period).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD + Signal
    ema_fast    = close.ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow    = close.ewm(span=config.macd_slow, adjust=False).mean()
    df["MACD"]  = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=config.macd_signal, adjust=False).mean()

    # Bollinger Bands
    mid = close.rolling(config.bb_period).mean()
    std = close.rolling(config.bb_period).std()
    df["BB_Upper"] = mid + config.bb_std * std
    df["BB_Lower"] = mid - config.bb_std * std

    # Volatility (rolling std of returns)
    df["Volatility"] = df["Returns"].rolling(config.vol_window).std()

    # Average True Range (needs High/Low — degrades to 0.0 if absent)
    if {"High", "Low"}.issubset(df.columns):
        prev_close = close.shift(1)
        true_range = pd.concat([
            (df["High"] - df["Low"]),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(config.atr_period).mean()
    else:
        df["ATR"] = 0.0

    # Volume passthrough — default to 0.0 (not NaN) so dropna()
    # never wipes out rows just because Volume was missing.
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    return df


def add_features(df: pd.DataFrame,
                 config: EngineConfig = CONFIG) -> pd.DataFrame:
    """Computes all 13 indicators and drops warm-up rows containing NaN."""
    return compute_indicators(df, config).dropna().copy()

def _recompute_indicators(close_series: pd.Series) -> pd.DataFrame:
    """
    Recalculates all technical indicators from a Close series.
    Used during recursive forecasting to update indicators
    after each new predicted price is appended.

    Returns a single-row DataFrame with the latest values.
    """
    df = pd.DataFrame({"Close": close_series.values})

    df["MA20"]   = df["Close"].rolling(20).mean()
    df["MA50"]   = df["Close"].rolling(50).mean()
    delta        = df["Close"].diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    rs           = gain / (loss + 1e-9)
    df["RSI"]    = 100 - (100 / (1 + rs))
    ema12        = df["Close"].ewm(span=12, adjust=False).mean()
    ema26        = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Return only the last row — the most recently calculated values
    return df.iloc[[-1]]

class StockPreprocessor:
    """
    Handles Min-Max normalisation and 60-day sliding window
    sequence building.

    Close must remain at index 0 in FEATURES because it is
    the prediction target and inverse_transform uses column 0.
    """

    FEATURES: List[str] = [
    "Close", "Returns", "MA20", "MA50", "EMA20",
    "RSI", "MACD", "Signal", "BB_Upper", "BB_Lower",
    "Volatility", "ATR", "Volume",]

    def __init__(self, window_size: int = 60):
        # 60-day window as specified in the methodology
        self.window_size = window_size
        # MinMaxScaler transforms each feature to range [0, 1]
        self.scaler      = MinMaxScaler(feature_range=(0, 1))
        self.features    = self.FEATURES

    def prepare_data(self, df: pd.DataFrame):
        """
        Scales the features and builds sliding window sequences.

        Returns:
        X           : input sequences, shape (N, 60, 13)
                      N windows, each 60 days, 13 features
        y           : target values, shape (N,)
                      the next-day Close price in scaled form
        scaled_full : the full scaled dataset
                      kept for use in forecasting later
        """
        data        = df[self.features].values
        scaled_full = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.window_size, len(scaled_full)):
            X.append(scaled_full[i - self.window_size : i])
            y.append(scaled_full[i, 0])   # Close is column 0

        return np.array(X), np.array(y), scaled_full

    def inverse_transform(self, pred: np.ndarray) -> np.ndarray:
        """
        Converts scaled Close predictions back to actual RM prices.
        Needed to display real prices in the dashboard and to
        compute metrics in RM rather than in scaled values.
        """
        dummy       = np.zeros((len(pred), len(self.features)))
        dummy[:, 0] = pred.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]

    def scale_row(self, row: np.ndarray) -> np.ndarray:
        """
        Scales a single new feature row using the already-fitted
        scaler. Used during recursive forecasting.

        Clips output to [0, 1] to guard against scaler
        extrapolation instability. If a predicted price exceeds
        the historical min/max range the scaler was trained on,
        the resulting scaled value would be outside [0, 1] and
        would destabilise the LSTM/GRU layers. Clipping prevents
        this. This is documented as a known limitation.

        row shape: (1, 13)
        """
        scaled = self.scaler.transform(row)
        return np.clip(scaled, 0.0, 1.0)
    

def build_lstm(input_shape):
    """
    2-layer stacked LSTM as specified in section 3.6.
    input_shape: (60, 13) — 60 days, 13 features.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ], name="LSTM")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_gru(input_shape):
    """
    2-layer stacked GRU as specified in section 3.6.
    Same architecture as LSTM for a fair comparison.
    """
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ], name="GRU")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_xgboost():
    """
    XGBoost gradient boosting model as specified in section 3.6.
    Receives flat 2D input (360 values) not 3D sequences.
    """
    return XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        verbosity=0,
        random_state=RANDOM_SEED,
    )

def directional_accuracy(y_true_rm: np.ndarray,
                          y_pred_rm: np.ndarray) -> float:
    """
    Directional accuracy on actual RM prices.

    For each step i, compares whether the model correctly
    predicted the direction of price[i] relative to the
    previous ACTUAL close price[i-1]. This anchors each
    evaluation to real market data rather than to a previous
    prediction, avoiding compounding direction errors.

    Returns a float between 0 and 1. Multiply by 100 for %.
    """
    y_true = y_true_rm.flatten()
    y_pred = y_pred_rm.flatten()

    # Actual direction: did price go up from previous real close?
    actual_dir = y_true[1:] > y_true[:-1]

    # Predicted direction: does prediction imply rise from
    # the previous ACTUAL close? (not previous prediction)
    pred_dir = y_pred[1:] > y_true[:-1]

    return float(np.mean(actual_dir == pred_dir))


def sharpe_ratio(y_true_rm: np.ndarray,
                  y_pred_rm: np.ndarray,
                  risk_free: float = 0.0) -> float:
    """
    Approximate annualised Sharpe Ratio.
    Simulates a long/short strategy driven by model signals.

    NOTE: This is an approximation. It does not account for
    transaction costs, position sizing, slippage, leverage,
    or trade execution delays. Interpret as a directional
    statistical performance indicator only.
    """
    signal = (np.diff(y_pred_rm.flatten()) > 0).astype(float)*2-1
    ret    = signal * np.diff(y_true_rm.flatten()) / (
                 y_true_rm.flatten()[:-1] + 1e-9)
    excess = ret - risk_free / 252
    if excess.std() < 1e-9:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())

def walk_forward_validation(X:            np.ndarray,
                             y:            np.ndarray,
                             preprocessor: StockPreprocessor,
                             model_type:   str = "XGBoost",
                             n_splits:     int = 5,
                             epochs:       int = 10) -> dict:
    """
    Expanding-window walk-forward cross-validation.
    Implements section 3.6 of the methodology.

    All metrics are computed on inverse-transformed RM prices
    so MAE is interpretable as actual Ringgit error.

    Returns averaged MAE (RM), directional accuracy, Sharpe
    Ratio, and all predictions/actuals in RM for charting.
    """
    fold_size  = len(X) // (n_splits + 1)
    maes, accs, sharpes       = [], [], []
    all_preds_rm, all_acts_rm = [], []

    # EarlyStopping halts training when validation loss stops
    # improving for 3 consecutive epochs. Prevents overfitting.
    early_stop = EarlyStopping(patience=3,
                               restore_best_weights=True,
                               verbose=0)

    for i in range(n_splits):
        train_end  = fold_size * (i + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, len(X))

        X_tr, X_te = X[:train_end], X[test_start:test_end]
        y_tr, y_te = y[:train_end], y[test_start:test_end]

        if model_type == "XGBoost":
            model = build_xgboost()
            # Flatten 3D to 2D: (N, 60, 13) becomes (N, 780)
            model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
            preds_scaled = model.predict(
                               X_te.reshape(len(X_te), -1))

        elif model_type == "LSTM":
            model = build_lstm((X_tr.shape[1], X_tr.shape[2]))
            model.fit(X_tr, y_tr, epochs=epochs,
                      batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            preds_scaled = model.predict(X_te, verbose=0).flatten()

        elif model_type == "GRU":
            model = build_gru((X_tr.shape[1], X_tr.shape[2]))
            model.fit(X_tr, y_tr, epochs=epochs,
                      batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            preds_scaled = model.predict(X_te, verbose=0).flatten()

        # Inverse-transform BEFORE computing any metrics.
        # This ensures all metrics are in RM not scaled values.
        true_rm  = preprocessor.inverse_transform(y_te)
        preds_rm = preprocessor.inverse_transform(preds_scaled)

        maes.append(mean_absolute_error(true_rm, preds_rm))
        accs.append(directional_accuracy(true_rm, preds_rm))
        sharpes.append(sharpe_ratio(true_rm, preds_rm))
        all_preds_rm.extend(preds_rm.tolist())
        all_acts_rm.extend(true_rm.tolist())

    return {
        "avg_mae":        float(np.mean(maes)),
        "avg_acc":        float(np.mean(accs)),
        "avg_sharpe":     float(np.mean(sharpes)),
        "predictions_rm": np.array(all_preds_rm),
        "actuals_rm":     np.array(all_acts_rm),
        "predictions":    np.array(all_preds_rm),
        "actuals":        np.array(all_acts_rm),
    }

def compute_shap_importance(X:            np.ndarray,
                             y:            np.ndarray,
                             preprocessor: StockPreprocessor,
                             n_background: int = 100) -> pd.DataFrame:
    """
    XGBoost SHAP feature importance using the true target y.

    Trains XGBoost on (X, y) — the same pair used in
    walk_forward_validation — then applies SHAP TreeExplainer
    to quantify the contribution of each feature.

    Since the input is a flattened 60-day x 13-feature window
    (780 columns total), SHAP values are aggregated across
    all 60 time steps to produce one score per feature.

    Returns a DataFrame ['Feature', 'SHAP Importance'] sorted
    by importance descending.
    """
    # Flatten: (N, 60, 13) → (N, 780)
    X_flat = X.reshape(X.shape[0], -1)

    # Train XGBoost on the TRUE forecasting target y
    model = build_xgboost()
    model.fit(X_flat, y)

    # SHAP TreeExplainer for tree-based models
    n_bg       = min(n_background, len(X_flat))
    background = X_flat[:n_bg]
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(background)

    # shap_vals shape: (n_bg, 780)
    # Reshape to (n_bg, 60 days, 13 features) then average
    # across samples and days to get one score per feature
    n_features = len(preprocessor.features)
    window     = preprocessor.window_size
    shap_3d    = shap_vals.reshape(n_bg, window, n_features)
    importance = np.abs(shap_3d).mean(axis=(0, 1))    # shape (13,)

    df = pd.DataFrame({
        "Feature":         preprocessor.features,
        "SHAP Importance": np.round(importance, 6),
    }).sort_values("SHAP Importance",
                   ascending=False).reset_index(drop=True)

    return df

# ═══════════════════════════════════════════════════════════════════
# LOCAL AI SENTIMENT AGENT — Track A: Company News (RSS)
# ═══════════════════════════════════════════════════════════════════

# RSS feed sources for Malaysian financial news.
# These provide a multilingual mix of English and Bahasa
# Melayu corporate and market coverage.
RSS_FEEDS = [
    # 1. The Edge Malaysia — Malaysia's leading financial news outlet
    "https://www.theedgemalaysia.com/rss.html",
    # 2. Bernama — Malaysia's official national news agency
    "https://www.bernama.com/en/rssfeed.php",
    # 3. Free Malaysia Today — independent bilingual outlet (EN/BM)
    "https://www.freemalaysiatoday.com/feeds/rss/headlines/",
    # 4. Malay Mail — established Malaysian daily, fully digital
    "https://www.malaymail.com/feed/rss/malaysia",
    # 5. The Sun Malaysia — independent English daily
    "https://thesun.my/feed/",
]


def fetch_company_headlines(company_name: str,
                             max_headlines: int = 8) -> list:
    """
    Track A — Company News.

    Pulls the most recent headlines from Malaysian financial
    RSS feeds and filters them down to articles that mention
    the given company name. Accepts a multilingual mix of
    English and Bahasa Melayu headlines without any
    translation step — the language model itself reads and
    reasons in both languages in Step 18.

    company_name : the company name to filter for, e.g.
                    "Maybank" or "CIMB"
    max_headlines : maximum number of matching headlines
                    to return

    Returns a list of headline strings. Returns an empty
    list if no feed can be reached or no match is found —
    this is handled gracefully by run_sentiment_agent() later,
    not by raising an error here.
    """
    matched = []

    for url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                title = entry.get("title", "").strip()
                if title and company_name.lower() in title.lower():
                    matched.append(title)
        except Exception:
            # If one feed fails (site down, network issue),
            # continue trying the remaining feeds instead of
            # crashing the whole pipeline.
            continue

    return matched[:max_headlines]

def mock_macro_indicators() -> dict:
    """
    Track B — Macroeconomic Indicators.

    Provides Malaysia's current high-level macroeconomic
    context: Bank Negara's Overnight Policy Rate (OPR),
    CPI inflation percentage, and GDP growth rate.

    This is explicitly a placeholder function. Live versions
    of these figures are published periodically by Bank
    Negara Malaysia and the Department of Statistics
    Malaysia, but typically require either a paid data feed
    or manual quarterly updates rather than a real-time API.
    This function can be replaced with a live data source in
    future work without changing any other part of the
    sentiment pipeline, since run_sentiment_agent() only
    expects a dictionary with these three keys.

    Returns a dictionary with OPR, CPI and GDP_Growth as
    percentages.
    """
    return {
        "OPR":        3.00,   # Overnight Policy Rate, %
        "CPI":        2.10,   # Inflation rate, %
        "GDP_Growth": 4.40,   # GDP growth rate, %
    }

def build_agent_context(company_name: str) -> dict:
    """
    Bundles Track A (company headlines) and Track B (macro
    indicators) into a single unified context dictionary for
    a given company.

    This is the robust mock/placeholder ingestion function
    referenced in the methodology. It always returns a valid
    dictionary even if no headlines are found, so the calling
    function never has to handle a missing-data case itself.

    company_name : e.g. "Maybank" or "CIMB"

    Returns a dict with keys: headlines (list) and macro (dict)
    """
    headlines = fetch_company_headlines(company_name)
    macro     = mock_macro_indicators()

    return {
        "headlines": headlines,
        "macro":     macro,
    }


def _extract_json(raw_text: str) -> dict:
    """
    Extracts the first JSON object found in a block of text.
    Used because language models sometimes wrap their JSON
    output with extra sentences despite being told not to.

    Raises ValueError if no JSON object can be found, which
    run_sentiment_agent() catches and handles gracefully.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())


def _default_neutral_response() -> dict:
    """
    The safe fallback response used whenever the agent
    pipeline cannot produce a valid result — no headlines,
    no model connection, or a malformed response. Matches
    the exact structure specified in the methodology.
    """
    return {
        "sentiment_label":       "Neutral",
        "macro_impact":          "No recent news or macroeconomic "
                                  "coverage available.",
        "qualitative_rationale": "No recent news or macroeconomic "
                                  "coverage available.",
    }


def run_sentiment_agent(ticker: str, company_name: str) -> dict:
    """
    The core local AI sentiment pipeline. Replaces VADER and
    TextBlob entirely.

    Connects to a local Llama 3 model running through Ollama
    (completely free, no API key, runs on your own machine).
    Builds a unified context of company headlines and
    macroeconomic indicators, then asks the model to act as
    a financial analyst and return a strict JSON verdict.

    ticker        : the Yahoo Finance ticker, e.g. '1155.KL'
                    (kept for interface consistency with the
                    rest of engine.py, not directly used by
                    the language model)
    company_name  : the company name used to search RSS
                    feeds, e.g. "Maybank"

    Returns a dictionary with exactly three keys:
        sentiment_label       : "Positive", "Negative" or "Neutral"
        macro_impact           : 1-2 sentence assessment of how
                                  macro conditions affect this
                                  company's sector, in English
        qualitative_rationale  : 1-2 sentence plain-English
                                  justification for the label

    If anything fails (no headlines, Ollama not running, a
    malformed response) this function never raises an error
    upward to the dashboard. It always returns the same three
    keys with a safe Neutral default instead.
    """
    try:
        context   = build_agent_context(company_name)
        headlines = context["headlines"]
        macro     = context["macro"]

        # If there is nothing at all to analyse, do not even
        # call the model — return the documented default
        # immediately, exactly as the methodology specifies.
        if not headlines:
            return _default_neutral_response()

        headline_block = "\n".join(f"- {h}" for h in headlines)

        prompt = f"""You are an elite Financial Intelligence Analyst
specialising in the Malaysian stock market (Bursa Malaysia).

Analyse the following data for the company "{company_name}".

COMPANY HEADLINES (may include English, Bahasa Melayu or
Mandarin, and may reference corporate announcements):
{headline_block}

MACROECONOMIC INDICATORS (Malaysia, current):
- Overnight Policy Rate (OPR): {macro['OPR']}%
- CPI / Inflation: {macro['CPI']}%
- GDP Growth: {macro['GDP_Growth']}%

Cross-reference the headlines with the macroeconomic
indicators above. Consider how interest rates, inflation,
or any supply chain or shipping disruptions mentioned in
the headlines would affect this specific company's sector.

Respond with ONLY a JSON object in exactly this format,
with no extra text before or after it, and with all text
written fully in English:

{{
  "sentiment_label": "Positive" or "Negative" or "Neutral",
  "macro_impact": "1-2 sentences on how the macro indicators
                    or any supply chain/shipping shocks in
                    the headlines affect this company's sector",
  "qualitative_rationale": "1-2 sentences explaining the
                    overall sentiment label for an everyday
                    retail investor"
}}"""

        # Connect to the local Ollama server running llama3.
        # No API key required — this runs entirely on your
        # own machine at http://localhost:11434
        llm = Ollama(model="llama3")
        raw_response = llm.invoke(prompt)

        parsed = _extract_json(raw_response)

        # Defensive validation — make sure the model actually
        # returned all three required keys with a valid label.
        label = parsed.get("sentiment_label", "Neutral")
        if label not in ("Positive", "Negative", "Neutral"):
            label = "Neutral"

        return {
            "sentiment_label":      label,
            "macro_impact":         parsed.get(
                "macro_impact", "No macro impact assessment available."),
            "qualitative_rationale": parsed.get(
                "qualitative_rationale",
                "No qualitative rationale available."),
        }

    except Exception:
        # Any failure anywhere in the pipeline — RSS feeds down,
        # Ollama not running, malformed JSON — falls back here.
        # The dashboard always receives a usable result.
        return _default_neutral_response()
    

