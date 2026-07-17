import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Enable oneDNN optimizations
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # no usable TF GPU on native Windows anyway; kills the "GPU support not available" notice
import re
import json
import inspect
import logging
import warnings
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time as dtime
from difflib import SequenceMatcher
from typing import Optional, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message=".*input_shape.*")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    yf = None
    _HAS_YF = False

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    _HAS_FEEDPARSER = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    XGBRegressor = None
    _HAS_XGB = False

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    optuna = None
    _HAS_OPTUNA = False

try:
    from langchain_ollama import OllamaLLM as Ollama
    _HAS_OLLAMA = True
except ImportError:
    Ollama = None
    _HAS_OLLAMA = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    _HAS_TF = True
except ImportError:
    Sequential = None
    Input = LSTM = GRU = Dense = Dropout = None
    Adam = EarlyStopping = ReduceLROnPlateau = None
    _HAS_TF = False


def _require(flag, package, feature):
    if not flag:
        raise ImportError(f"'{package}' is required for {feature} but is not installed.")


@dataclass
class EngineConfig:
    window_size: int = 60
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
    epochs: int = 45
    champion_epochs: int = 60
    batch_size: int = 32
    patience: int = 6
    validation_split: float = 0.1
    n_splits: int = 5
    forecast_horizon: int = 5
    range_lookback: int = 20
    stop_pct: float = 0.05
    rss_feeds: tuple = (
        # 1. The Edge Malaysia — Malaysia's leading financial news
        # organisation. Covers Bursa Malaysia, corporates, economics.
        # Primary source for market-moving business news.
        "https://www.theedgemalaysia.com/rss.html",
        # 2. Bernama — Malaysia's official national news agency.
        # Authoritative source for government, economic and
        # corporate announcements.
        "https://www.bernama.com/en/rssfeed.php",
        # 3. Free Malaysia Today — independent bilingual (English/BM)
        # outlet. Covers business and economic policy news.
        "https://www.freemalaysiatoday.com/feeds/rss/headlines/",
        # 4. Malay Mail — established Malaysian daily, fully digital
        # since 2018. Covers local business and market news.
        "https://www.malaymail.com/feed/rss/malaysia",
        # 5. The Sun Malaysia — independent English daily with
        # dedicated business coverage of Bursa-listed companies.
        "https://thesun.my/feed/",
    )
    max_headlines: int = 8
    fuzzy_threshold: float = 0.82
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    macro_provider: Optional[Callable] = None
    half_day_cutoff: dtime = dtime(13, 0)
    end_of_day_cutoff: dtime = dtime(17, 0)


CONFIG = EngineConfig()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FEATURES = [
    "Returns", "Close", "MA20", "MA50", "EMA20",
    "RSI", "MACD", "Signal", "BB_Upper", "BB_Lower",
    "Volatility", "ATR", "Volume",
    "Returns_Lag1", "Returns_Lag5", "Index_Return", "FX_Return",
]

# 10 highly liquid Bursa Malaysia stocks. Keys are display names used
# throughout the dashboard (and by run_tournament.py) and passed to
# run_sentiment_agent() as the company_name for headline matching;
# values are Yahoo Finance tickers. This is the single source of
# truth for the stock universe - both app.py and run_tournament.py
# import it from here rather than keeping their own copies.
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

COMPANY_ALIASES = {
    "Maybank":            ["maybank", "malayan banking", "mbb", "1155.kl"],
    "CIMB":               ["cimb", "cimb group", "1023.kl"],
    "Public Bank":        ["public bank", "pbbank", "pbb", "1295.kl"],
    "TNB":                ["tenaga", "tenaga nasional", "tnb", "5347.kl"],
    "Petronas Chemicals": ["petronas chemicals", "petchem", "5183.kl"],
    "Maxis":              ["maxis", "maxis berhad", "6012.kl"],
    "Axiata":             ["axiata", "axiata group", "6888.kl"],
    "Genting":            ["genting", "genting berhad", "genting bhd", "3182.kl"],
    "KL Kepong":          ["kl kepong", "klk", "kuala lumpur kepong", "2445.kl"],
    "IHH Healthcare":     ["ihh", "ihh healthcare", "5225.kl"],
}


def fetch_data(symbol, period="5y", config=CONFIG):
    _require(_HAS_YF, "yfinance", "market data download")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df.ffill()


_AUX_SERIES_CACHE: Dict[str, pd.Series] = {}


def _fetch_aux_return_series(ticker, period="5y"):
    cache_key = f"{ticker}:{period}"
    if cache_key in _AUX_SERIES_CACHE:
        return _AUX_SERIES_CACHE[cache_key]
    series = pd.Series(dtype=float)
    if _HAS_YF:
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if not hist.empty:
                series = hist["Close"].pct_change()
                series.index = series.index.tz_localize(None) if series.index.tz is not None else series.index
        except Exception:
            series = pd.Series(dtype=float)
    _AUX_SERIES_CACHE[cache_key] = series
    return series


def _align_aux_series(df_index, ticker, period="5y"):
    aux = _fetch_aux_return_series(ticker, period=period)
    idx = df_index.tz_localize(None) if getattr(df_index, "tz", None) is not None else df_index
    if aux.empty:
        return np.zeros(len(df_index))
    aligned = aux.reindex(idx, method="ffill").fillna(0.0)
    return aligned.values


def compute_indicators(df, config=CONFIG):
    df = df.copy()
    close = df["Close"]
    df["Returns"] = close.pct_change()
    df["MA20"] = close.rolling(config.ma_short).mean()
    df["MA50"] = close.rolling(config.ma_long).mean()
    df["EMA20"] = close.ewm(span=config.ema_span, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(config.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(config.rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema_fast = close.ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=config.macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=config.macd_signal, adjust=False).mean()
    mid = close.rolling(config.bb_period).mean()
    std = close.rolling(config.bb_period).std()
    df["BB_Upper"] = mid + config.bb_std * std
    df["BB_Lower"] = mid - config.bb_std * std
    df["Volatility"] = df["Returns"].rolling(config.vol_window).std()
    df["Returns_Lag1"] = df["Returns"].shift(1)
    df["Returns_Lag5"] = df["Returns"].shift(5)
    df["Index_Return"] = _align_aux_series(df.index, "^KLSE")
    df["FX_Return"] = _align_aux_series(df.index, "MYR=X")
    df["Target_Return"] = close.shift(-config.forecast_horizon) / close - 1.0
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
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df


def add_features(df, config=CONFIG):
    df = compute_indicators(df, config)
    return df.dropna(subset=FEATURES).reset_index(drop=True).copy()


class StockPreprocessor:
    FEATURES = FEATURES
    TARGET_COLUMN = "Target_Return"

    def __init__(self, window_size=None, config=CONFIG):
        self.config = config
        self.window_size = window_size or config.window_size
        self.scaler = RobustScaler()
        self.features = list(self.FEATURES)

    def get_feature_matrix(self, df):
        return df[self.features].values

    def get_target_array(self, df):
        return df[self.TARGET_COLUMN].values

    def build_sequences(self, scaled, target_raw):

        X, y, anchor_idx = [], [], []
        for i in range(self.window_size, len(scaled)):
            anchor = i - 1
            t = target_raw[anchor]
            if np.isnan(t):
                continue
            X.append(scaled[i - self.window_size:i])
            y.append(t)
            anchor_idx.append(anchor)
        return np.array(X), np.array(y), np.array(anchor_idx, dtype=int)

    def prepare_data(self, df):
        scaled_full = self.scaler.fit_transform(self.get_feature_matrix(df))
        target_raw = self.get_target_array(df)
        X, y, _anchor_idx = self.build_sequences(scaled_full, target_raw)
        return X, y, scaled_full

    def inverse_transform_column(self, values, col_name, scaler=None):
        sc = scaler if scaler is not None else self.scaler
        idx = self.features.index(col_name)
        dummy = np.zeros((len(values), len(self.features)))
        dummy[:, idx] = np.asarray(values).flatten()
        return sc.inverse_transform(dummy)[:, idx]

    def transform_column(self, values, col_name, scaler=None):
        sc = scaler if scaler is not None else self.scaler
        idx = self.features.index(col_name)
        dummy = np.zeros((len(values), len(self.features)))
        dummy[:, idx] = np.asarray(values).flatten()
        return sc.transform(dummy)[:, idx]

    def get_last_window_3d(self, scaled_full):
        return scaled_full[-self.window_size:][np.newaxis, ...]

    def get_last_window_2d(self, scaled_full):
        return scaled_full[-self.window_size:].flatten()[np.newaxis, :]


def _compile(model, learning_rate=None):
    optimizer = Adam(learning_rate=learning_rate) if learning_rate else "adam"
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def build_lstm(input_shape, units=96, dropout=0.2, learning_rate=None):
    _require(_HAS_TF, "tensorflow", "the LSTM model")
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(50, activation="relu"),
        Dense(1),
    ], name="LSTM")
    return _compile(model, learning_rate)


def build_gru(input_shape, units=96, dropout=0.2, learning_rate=None):
    _require(_HAS_TF, "tensorflow", "the GRU model")
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=True),
        Dropout(dropout),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(50, activation="relu"),
        Dense(1),
    ], name="GRU")
    return _compile(model, learning_rate)


_SEQUENCE_BUILDERS = {
    "LSTM": build_lstm, "GRU": build_gru,
}
SEQUENCE_MODELS = tuple(_SEQUENCE_BUILDERS)


def build_sequence_model(model_type, input_shape, **kwargs):
    builder = _SEQUENCE_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"Unknown sequence model_type: {model_type!r}. Choose from {SEQUENCE_MODELS}.")
    sig = inspect.signature(builder)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return builder(input_shape, **accepted)


def xgb_engineer_features(X, feature_names):
    n, w, f = X.shape
    last = X[:, -1, :]
    momentum = last - X[:, 0, :]
    mean_all = X.mean(axis=1)
    std_all = X.std(axis=1)
    mean_5 = X[:, -min(5, w):, :].mean(axis=1)
    mean_10 = X[:, -min(10, w):, :].mean(axis=1)
    lag1 = X[:, -min(2, w), :]
    lag5 = X[:, -min(6, w), :]
    lag10 = X[:, -min(11, w), :]

    blocks = [last, lag1, lag5, lag10, mean_5, mean_10, mean_all, std_all, momentum]
    suffixes = ["last", "lag1", "lag5", "lag10", "mean5", "mean10", "mean_all", "std_all", "momentum"]
    engineered = np.concatenate(blocks, axis=1)
    names = [f"{name}_{suffix}" for suffix in suffixes for name in feature_names]
    return engineered, names


def build_xgboost(params=None):
    _require(_HAS_XGB, "xgboost", "the XGBoost model")
    defaults = dict(n_estimators=100, learning_rate=0.05, max_depth=5,
                    subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                    objective="reg:squarederror", verbosity=0, random_state=42)
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)


def _build_callbacks(config=CONFIG):
    return [
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True, min_delta=1e-5, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, config.patience - 1), min_lr=1e-5, verbose=0),
    ]


def tune_xgboost(X, y, n_iter=20, cv_splits=3, random_state=42, feature_names=None):
    feature_names = feature_names or [str(i) for i in range(X.shape[2])]
    X_flat, _ = xgb_engineer_features(X, feature_names)
    param_dist = {
        "n_estimators": [100, 200, 300, 400], "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0], "reg_lambda": [0.5, 1.0, 2.0],
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=random_state),
        param_distributions=param_dist, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=random_state, verbose=0)
    search.fit(X_flat, y)
    return search.best_params_, float(-search.best_score_)


def tune_keras(X, y, model_type="LSTM", config=CONFIG, n_trials=6, random_state=42):
    _require(_HAS_TF, "tensorflow", "Keras hyperparameter tuning")
    rng = np.random.default_rng(random_state)
    grid = {"units": [32, 50, 64, 96], "dropout": [0.1, 0.2, 0.3], "batch_size": [16, 32, 64]}
    split = int(len(X) * 0.85)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    best = None
    for _ in range(n_trials):
        units = int(rng.choice(grid["units"]))
        dropout = float(rng.choice(grid["dropout"]))
        batch_size = int(rng.choice(grid["batch_size"]))
        model = build_sequence_model(model_type, (X.shape[1], X.shape[2]), units=units, dropout=dropout)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=config.epochs,
                  batch_size=batch_size, callbacks=_build_callbacks(config), verbose=0)
        val_loss = float(model.evaluate(X_val, y_val, verbose=0))
        if best is None or val_loss < best["val_loss"]:
            best = {"units": units, "dropout": dropout, "batch_size": batch_size, "val_loss": val_loss}
    return best


def tune_xgboost_optuna(X, y, n_trials=30, cv_splits=3, timeout=None, random_state=42, feature_names=None):
    feature_names = feature_names or [str(i) for i in range(X.shape[2])]
    if not _HAS_OPTUNA:
        return tune_xgboost(X, y, n_iter=n_trials, cv_splits=cv_splits, random_state=random_state,
                             feature_names=feature_names)
    _require(_HAS_XGB, "xgboost", "XGBoost tuning")
    X_flat, _ = xgb_engineer_features(X, feature_names)
    splitter = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        fold_maes = []
        for tr_idx, val_idx in splitter.split(X_flat):
            model = build_xgboost(params)
            model.fit(X_flat[tr_idx], y[tr_idx])
            pred = model.predict(X_flat[val_idx])
            fold_maes.append(mean_absolute_error(y[val_idx], pred))
        return float(np.mean(fold_maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, float(study.best_value)


def tune_keras_optuna(X, y, model_type="LSTM", config=CONFIG, n_trials=20, timeout=None, random_state=42):
    if not (_HAS_OPTUNA and _HAS_TF):
        return tune_keras(X, y, model_type=model_type, config=config, random_state=random_state)
    split = int(len(X) * 0.85)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    input_shape = (X.shape[1], X.shape[2])

    def objective(trial):
        units = trial.suggest_categorical("units", [32, 50, 64, 96, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        model = build_sequence_model(model_type, input_shape, units=units, dropout=dropout, learning_rate=lr)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=config.epochs,
                  batch_size=batch_size, callbacks=_build_callbacks(config), verbose=0)
        return float(model.evaluate(X_val, y_val, verbose=0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    best["val_loss"] = float(study.best_value)
    return best


def optimize_hyperparameters(model_type, X, y, method="auto", config=CONFIG, feature_names=None, **kwargs):
    use_optuna = (method == "optuna") or (method == "auto" and _HAS_OPTUNA)
    if model_type == "XGBoost":
        if use_optuna:
            return tune_xgboost_optuna(X, y, feature_names=feature_names, **kwargs)
        return tune_xgboost(X, y, feature_names=feature_names, **kwargs)
    if model_type in SEQUENCE_MODELS:
        if use_optuna:
            return tune_keras_optuna(X, y, model_type=model_type, config=config, **kwargs)
        return tune_keras(X, y, model_type=model_type, config=config, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def get_tuning_window(df, preprocessor, n_splits=None, config=CONFIG):
    n_splits = n_splits or config.n_splits
    W = preprocessor.window_size
    raw = preprocessor.get_feature_matrix(df)
    target_raw = preprocessor.get_target_array(df)
    n_seq = len(raw) - W
    n_valid = max(0, n_seq - config.forecast_horizon)
    fold_size = n_valid // (n_splits + 1)
    train_end_seq = fold_size
    train_raw_end = W + train_end_seq
    scaler = RobustScaler()
    scaler.fit(raw[:train_raw_end])
    scaled_all = scaler.transform(raw)
    X_all, y_all, _anchor_idx = preprocessor.build_sequences(scaled_all, target_raw)
    return X_all[:train_end_seq], y_all[:train_end_seq]


def tune_model_params(df, preprocessor, model_type, config=CONFIG,
                       use_optuna=True, xgb_trials=15, keras_trials=10):
    X_tune, y_tune = get_tuning_window(df, preprocessor, config.n_splits, config)

    if model_type == "XGBoost":
        if use_optuna:
            best_params, _ = tune_xgboost_optuna(X_tune, y_tune, n_trials=xgb_trials, feature_names=preprocessor.features)
        else:
            best_params, _ = tune_xgboost(X_tune, y_tune, n_iter=xgb_trials, feature_names=preprocessor.features)
        return best_params

    if model_type in SEQUENCE_MODELS:
        if use_optuna:
            best = tune_keras_optuna(X_tune, y_tune, model_type=model_type,
                                      config=config, n_trials=keras_trials)
        else:
            best = tune_keras(X_tune, y_tune, model_type=model_type,
                               config=config, n_trials=keras_trials)
        return {k: v for k, v in best.items() if k != "val_loss"}

    raise ValueError(f"Unknown model_type: {model_type!r}")


def directional_accuracy(prev_close, y_true, y_pred):
    prev_close, y_true, y_pred = np.asarray(prev_close).flatten(), np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    actual_dir = y_true > prev_close
    pred_dir = y_pred > prev_close
    return float(np.mean(actual_dir == pred_dir))


def sharpe_ratio(prev_close, y_true, y_pred, risk_free=0.0):
    """Same anchoring fix as directional_accuracy: a long/short signal
    is taken based on whether the model predicted a rise or fall
    relative to the previous actual close, and realised return is the
    actual price change relative to that same anchor."""
    prev_close, y_true, y_pred = np.asarray(prev_close).flatten(), np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    signal = np.where(y_pred > prev_close, 1.0, -1.0)
    ret = signal * (y_true - prev_close) / (prev_close + 1e-9)
    excess = ret - risk_free / 252
    if excess.std() < 1e-9:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def walk_forward_validation(df, preprocessor, model_type="XGBoost", n_splits=None,
                            epochs=None, model_params=None, config=CONFIG):
    n_splits = n_splits or config.n_splits
    epochs = epochs or config.epochs
    horizon = config.forecast_horizon
    W = preprocessor.window_size
    raw = preprocessor.get_feature_matrix(df)
    target_raw = preprocessor.get_target_array(df)
    close_idx = preprocessor.features.index("Close")
    raw_close = raw[:, close_idx]

    n_seq = len(raw) - W
    n_valid = max(0, n_seq - horizon)
    if n_valid <= n_splits:
        raise ValueError(f"Not enough data ({n_valid} sequences with a known "
                          f"{horizon}-day forward target) for {n_splits} splits.")
    fold_size = n_valid // (n_splits + 1)
    maes, accs, sharpes = [], [], []
    all_preds_rm, all_acts_rm = [], []

    for i in range(n_splits):
        train_end_seq = fold_size * (i + 1)
        test_start_seq = train_end_seq
        test_end_seq = min(test_start_seq + fold_size, n_valid)
        train_raw_end = W + train_end_seq
        scaler = RobustScaler()
        scaler.fit(raw[:train_raw_end])
        scaled_all = scaler.transform(raw)
        X_all, y_all, anchor_idx = preprocessor.build_sequences(scaled_all, target_raw)
        X_tr, y_tr = X_all[:train_end_seq], y_all[:train_end_seq]
        X_te = X_all[test_start_seq:test_end_seq]
        y_te = y_all[test_start_seq:test_end_seq]
        anchor_te = anchor_idx[test_start_seq:test_end_seq]
        if len(X_te) == 0:
            continue

        if model_type == "XGBoost":
            model = build_xgboost(model_params)
            X_tr_eng, _ = xgb_engineer_features(X_tr, preprocessor.features)
            X_te_eng, _ = xgb_engineer_features(X_te, preprocessor.features)
            model.fit(X_tr_eng, y_tr)
            preds = model.predict(X_te_eng)
        elif model_type in SEQUENCE_MODELS:
            mp = model_params or {}
            seq_kw = {k: mp[k] for k in ("units", "dropout", "learning_rate") if k in mp}
            batch_size = mp.get("batch_size", config.batch_size)
            model = build_sequence_model(model_type, (X_tr.shape[1], X_tr.shape[2]), **seq_kw)
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      validation_split=config.validation_split, callbacks=_build_callbacks(config), verbose=0)
            preds = model.predict(X_te, verbose=0).flatten()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        prev_close = raw_close[anchor_te]
        actual_close = raw_close[anchor_te + horizon]
        pred_close = prev_close * (1.0 + preds)

        maes.append(mean_absolute_error(actual_close, pred_close))
        accs.append(directional_accuracy(prev_close, actual_close, pred_close))
        sharpes.append(sharpe_ratio(prev_close, actual_close, pred_close))
        all_preds_rm.extend(pred_close.tolist())
        all_acts_rm.extend(actual_close.tolist())

    return {
        "avg_mae": float(np.mean(maes)) if maes else float("nan"),
        "avg_acc": float(np.mean(accs)) if accs else float("nan"),
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else float("nan"),
        "predictions_rm": np.array(all_preds_rm), "actuals_rm": np.array(all_acts_rm),
        "predictions": np.array(all_preds_rm), "actuals": np.array(all_acts_rm),
    }


def select_champion(results, mae_weight=0.5, acc_weight=0.2, sharpe_weight=0.3):
    names = list(results.keys())
    maes = np.array([results[n]["avg_mae"] for n in names], dtype=float)
    accs = np.array([results[n]["avg_acc"] for n in names], dtype=float)
    sharpes = np.array([results[n]["avg_sharpe"] for n in names], dtype=float)

    def _normalise(values, higher_is_better):
        value_range = values.max() - values.min()
        if value_range <= 1e-12:
            return np.ones_like(values)
        norm = (values - values.min()) / value_range
        return norm if higher_is_better else 1.0 - norm

    mae_score = _normalise(maes, higher_is_better=False)
    acc_score = _normalise(accs, higher_is_better=True)
    sharpe_score = _normalise(sharpes, higher_is_better=True)

    composite = mae_weight * mae_score + acc_weight * acc_score + sharpe_weight * sharpe_score
    return names[int(np.argmax(composite))]


def run_model_tournament(df, config=CONFIG, model_types=("XGBoost", "LSTM", "GRU"),
                         n_splits=None, use_optuna=True, xgb_trials=15, keras_trials=10,
                         mae_weight=0.5, acc_weight=0.2, sharpe_weight=0.3):
    n_splits = n_splits or config.n_splits
    proc = StockPreprocessor(config=config)

    results = {}
    tuned_params = {}
    for mtype in model_types:
        params = tune_model_params(df, proc, mtype, config=config, use_optuna=use_optuna,
                                    xgb_trials=xgb_trials, keras_trials=keras_trials)
        tuned_params[mtype] = params
        results[mtype] = walk_forward_validation(df, proc, model_type=mtype, n_splits=n_splits,
                                                  model_params=params, config=config)

    champion = select_champion(results, mae_weight=mae_weight, acc_weight=acc_weight, sharpe_weight=sharpe_weight)
    return {
        "results": results,
        "tuned_params": tuned_params,
        "champion": champion,
        "champion_params": tuned_params[champion],
    }


class ChampionForecaster:
    def __init__(self, preprocessor, epochs=None, config=CONFIG):
        self.proc = preprocessor
        self.config = config
        self.epochs = epochs or config.champion_epochs
        self._model = None
        self._mtype = None
        self._scaled_full = None
        self._work = None

    def fit(self, X, y, scaled_full, model_type, raw_df=None, model_params=None):
        self._mtype = model_type
        if model_type == "XGBoost":
            self._model = build_xgboost(model_params)
            X_eng, _ = xgb_engineer_features(X, self.proc.features)
            self._model.fit(X_eng, y)
        elif model_type in SEQUENCE_MODELS:
            mp = model_params or {}
            seq_kw = {k: mp[k] for k in ("units", "dropout", "learning_rate") if k in mp}
            batch_size = mp.get("batch_size", self.config.batch_size)
            self._model = build_sequence_model(model_type, (X.shape[1], X.shape[2]), **seq_kw)
            self._model.fit(X, y, epochs=self.epochs, batch_size=batch_size,
                            validation_split=self.config.validation_split, callbacks=_build_callbacks(self.config), verbose=0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self._scaled_full = scaled_full
        if raw_df is not None:
            cols = [c for c in ("Close", "High", "Low", "Volume") if c in raw_df.columns]
            self._work = raw_df[cols].copy().reset_index(drop=True)
        else:
            self._work = None
        return self

    def forecast(self, n_days=None):
        horizon = n_days or self.config.forecast_horizon
        if horizon != self.config.forecast_horizon:
            raise ValueError(
                f"This model was trained to directly predict a "
                f"{self.config.forecast_horizon}-day forward return; it "
                f"can't be asked for a different horizon ({horizon}) "
                f"without retraining on a target computed for that "
                f"horizon (see compute_indicators/Target_Return)."
            )
        if self._scaled_full is None or self._work is None:
            raise ValueError(
                "ChampionForecaster.forecast() requires raw_df to have been "
                "passed to fit(): the model predicts a forward return, "
                "which needs a real last-known price to anchor to when "
                "converting that prediction into a Ringgit forecast."
            )

        current_price = float(self._work["Close"].iloc[-1])
        window = self._scaled_full[-self.proc.window_size:]

        if self._mtype == "XGBoost":
            x_eng, _ = xgb_engineer_features(window[np.newaxis, ...], self.proc.features)
            pred_return = float(self._model.predict(x_eng)[0])
        else:
            x = window[np.newaxis, ...]
            pred_return = float(self._model.predict(x, verbose=0).flatten()[0])

        forecast_price = current_price * (1.0 + pred_return)
        return np.array([forecast_price])


def _aliases_for(company_name, aliases=None):
    aliases = aliases if aliases is not None else COMPANY_ALIASES
    name_l = company_name.lower()
    terms = {name_l}
    for canon, al in aliases.items():
        al_l = [a.lower() for a in al]
        if name_l == canon.lower() or name_l in al_l:
            terms.update(al_l)
            terms.add(canon.lower())
    return list(terms)


def _fuzzy_contains(title_lower, term, threshold):
    # Word-boundary substring match, not a bare `in` check - otherwise a
    # short term like a ticker code can match as a coincidental fragment
    # of an unrelated number or word in the headline (e.g. "5183" inside
    # a Ringgit figure or resignation count that has nothing to do with
    # that company).
    if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", title_lower):
        return True
    # The fuzzy fallback is only meaningful for terms long enough that a
    # high SequenceMatcher ratio actually implies relatedness. Short
    # terms (e.g. 3-letter tickers) can hit that ratio against unrelated
    # short words by chance, so skip fuzzy matching for them entirely.
    if len(term) < 5:
        return False
    for token in re.findall(r"[a-z0-9.]+", title_lower):
        if SequenceMatcher(None, token, term).ratio() >= threshold:
            return True
    return False


def _headline_matches(title, company_name, threshold, aliases=None):
    title_lower = title.lower()
    return any(_fuzzy_contains(title_lower, term, threshold) for term in _aliases_for(company_name, aliases))


def fetch_company_headlines(company_name, max_headlines=None, config=CONFIG, aliases=None):
    max_headlines = max_headlines or config.max_headlines
    matched = []
    if not _HAS_FEEDPARSER:
        return matched
    for url in config.rss_feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                title = entry.get("title", "").strip()
                if title and _headline_matches(title, company_name, config.fuzzy_threshold, aliases):
                    matched.append(title)
        except Exception:
            continue
    return matched[:max_headlines]


def _static_macro_indicators():
    return {"OPR": 3.00, "CPI": 2.10, "GDP_Growth": 4.40}


def mock_macro_indicators():
    return _static_macro_indicators()


def get_macro_indicators(config=CONFIG):
    provider = config.macro_provider
    if callable(provider):
        try:
            data = provider()
            if all(k in data for k in ("OPR", "CPI", "GDP_Growth")):
                return {k: float(data[k]) for k in ("OPR", "CPI", "GDP_Growth")}
        except Exception:
            pass
    return _static_macro_indicators()


def _http_get_json(url, headers=None, timeout=10):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bnm_opr(timeout=10):
    url = "https://api.bnm.gov.my/public/opr"
    headers = {"Accept": "application/vnd.BNM.API.v1+json"}
    payload = _http_get_json(url, headers=headers, timeout=timeout)
    record = payload.get("data", payload)
    if isinstance(record, list):
        record = record[-1] if record else {}
    for key in ("new_opr_level", "opr_level", "rate"):
        if isinstance(record, dict) and key in record:
            return float(record[key])
    raise ValueError("Could not locate an OPR value in the BNM API response.")


def bnm_macro_provider(timeout=10, cpi_fn=None, gdp_fn=None):
    macro = dict(_static_macro_indicators())
    try:
        macro["OPR"] = fetch_bnm_opr(timeout=timeout)
    except Exception:
        pass
    for field, fn in (("CPI", cpi_fn), ("GDP_Growth", gdp_fn)):
        if callable(fn):
            try:
                macro[field] = float(fn())
            except Exception:
                pass
    return macro


def build_agent_context(company_name, config=CONFIG):
    return {"headlines": fetch_company_headlines(company_name, config=config), "macro": get_macro_indicators(config)}


def _extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())


def _default_neutral_response():
    msg = "No recent news or macroeconomic coverage available."
    return {"sentiment_label": "Neutral", "macro_impact": msg, "qualitative_rationale": msg}


_NEGATIVE_RATIONALE_WORDS = (
    "decline", "declining", "fall", "falling", "drop", "dropping", "loss",
    "losses", "weak", "weaken", "weakening", "risk", "risks", "risky",
    "shortage", "shortages", "exodus", "concern", "concerns", "negative",
    "negatively", "hurt", "hurting", "harm", "harmful", "downturn",
    "cut", "cuts", "reduced", "reduce", "reducing", "slowdown", "strain",
    "pressure", "pressured", "underperform", "warn", "warns", "warning",
)
_POSITIVE_RATIONALE_WORDS = (
    "growth", "growing", "grow", "rise", "rising", "rose", "gain",
    "gains", "strong", "strengthen", "strengthening", "positive",
    "positively", "boost", "boosting", "improve", "improving",
    "improvement", "record", "surge", "surging", "expansion",
    "expanding", "upgrade", "upgraded", "outperform", "beat", "beats",
)


def _label_consistent_with_rationale(label, rationale):
    """
    Cheap safety net for the (fairly common) failure mode where a small
    local LLM writes a rationale with a clear negative or positive tone
    but fills sentiment_label with an unrelated/default value. Not a
    substitute for a good prompt - just a backstop that flags obvious
    mismatches so we don't silently show contradictory text next to
    each other in the UI.
    """
    text = rationale.lower()
    neg_hits = sum(1 for w in _NEGATIVE_RATIONALE_WORDS if w in text)
    pos_hits = sum(1 for w in _POSITIVE_RATIONALE_WORDS if w in text)
    if neg_hits >= 2 and neg_hits > pos_hits and label != "Negative":
        return False
    if pos_hits >= 2 and pos_hits > neg_hits and label != "Positive":
        return False
    return True


def run_sentiment_agent(ticker, company_name, config=CONFIG):
    try:
        context = build_agent_context(company_name, config)
    except Exception:
        return _default_neutral_response()
    headlines = context["headlines"]
    macro = context["macro"]
    if not headlines:
        return _default_neutral_response()
    headline_block = "\n".join(f"- {h}" for h in headlines)
    # The label is asked for LAST and explicitly tied to the rationale,
    # rather than as an independent field the model can fill in without
    # regard to what it just wrote - this is what was causing the label
    # to disagree with the rationale's actual tone.
    prompt = f"""You are an elite Financial Intelligence Analyst.
Company: {company_name}
Headlines:
{headline_block}
Macro: OPR={macro['OPR']}% CPI={macro['CPI']}% GDP={macro['GDP_Growth']}%

Step 1: Write "macro_impact" - a short assessment of how these macro figures affect this specific company.
Step 2: Write "qualitative_rationale" - a short assessment of how these specific headlines affect this specific company.
Step 3: Read back what you wrote in Step 2. If it describes a net negative effect on the company, sentiment_label must be "Negative". If it describes a net positive effect, sentiment_label must be "Positive". Only use "Neutral" if the effect is genuinely mixed or unclear.

Respond with ONLY JSON with exactly these keys, in this order: macro_impact, qualitative_rationale, sentiment_label"""
    if not _HAS_OLLAMA:
        return _default_neutral_response()
    try:
        llm = Ollama(model=config.ollama_model, base_url=config.ollama_base_url)
        raw_response = llm.invoke(prompt)
    except Exception:
        return _default_neutral_response()
    try:
        parsed = _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError):
        return _default_neutral_response()
    label = parsed.get("sentiment_label", "Neutral")
    if label not in ("Positive", "Negative", "Neutral"):
        label = "Neutral"
    rationale = parsed.get("qualitative_rationale", "No qualitative rationale available.")
    if not _label_consistent_with_rationale(label, rationale):
        text = rationale.lower()
        neg_hits = sum(1 for w in _NEGATIVE_RATIONALE_WORDS if w in text)
        pos_hits = sum(1 for w in _POSITIVE_RATIONALE_WORDS if w in text)
        label = "Negative" if neg_hits > pos_hits else "Positive"
    return {"sentiment_label": label,
            "macro_impact": parsed.get("macro_impact", "No macro impact assessment available."),
            "qualitative_rationale": rationale}


HALF_DAY_CUTOFF = CONFIG.half_day_cutoff
END_OF_DAY_CUTOFF = CONFIG.end_of_day_cutoff


def get_current_session_label(now=None, config=CONFIG):
    now = now or datetime.now()
    current_time = now.time()
    if current_time < config.half_day_cutoff:
        return "morning"
    if current_time < config.end_of_day_cutoff:
        return "half_day"
    return "end_of_day"


def stop_loss_advisory(current_price, forecast_prices, stop_pct=None, config=CONFIG):

    stop_pct = config.stop_pct if stop_pct is None else stop_pct
    threshold = current_price * (1 - stop_pct)
    horizon = config.forecast_horizon
    end_price = float(np.asarray(forecast_prices, dtype=float).flatten()[-1])
    breached = end_price < threshold
    if breached:
        advisory = (f"The {horizon}-day forecast (RM {end_price:.4f}) breaches the "
                    f"{stop_pct*100:.0f}% stop-loss. Consider a sell order at RM {threshold:.4f}.")
    else:
        exp_ret = (end_price - current_price) / current_price * 100
        advisory = (f"No stop-loss breach in the {horizon}-day forecast. "
                    f"Forecast RM {end_price:.4f} ({exp_ret:+.2f}% vs today).")
    return {"threshold": round(threshold, 4), "breached": breached,
            "breach_day": (horizon if breached else None),
            "advisory": advisory, "stop_pct": stop_pct}


def build_ranking_table(stock_summaries):
    rows = []
    for name, s in stock_summaries.items():
        last = s["last_close"]
        fc_end = float(s["forecast"][-1])
        ret = (fc_end - last) / last * 100 if last else 0.0
        rows.append({"Stock": name, "Last Close (RM)": round(last, 4), "5-Day Forecast": round(fc_end, 4),
                    "Expected Return": round(ret, 2), "Champion": s["champion"], "MAE": round(s["mae"], 4),
                    "Dir. Acc (%)": round(s["da"] * 100, 1), "Sharpe": round(s["sharpe"], 3),
                    "Sentiment": s["sentiment_label"]})
    df = pd.DataFrame(rows).sort_values("Expected Return", ascending=False).reset_index(drop=True)
    df.index += 1
    return df