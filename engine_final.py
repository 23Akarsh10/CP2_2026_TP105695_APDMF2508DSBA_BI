import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Enable oneDNN optimizations
import re
import json
import inspect
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time as dtime
from difflib import SequenceMatcher
from typing import Optional, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

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
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    _HAS_TF = True
except ImportError:
    Sequential = None
    LSTM = GRU = Dense = Dropout = None
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
    epochs: int = 10
    champion_epochs: int = 15
    batch_size: int = 32
    patience: int = 3
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
    "Close", "Returns", "MA20", "MA50", "EMA20",
    "RSI", "MACD", "Signal", "BB_Upper", "BB_Lower",
    "Volatility", "ATR", "Volume",
]

COMPANY_ALIASES = {
    "Maybank":            ["maybank", "malayan banking", "mbb", "1155", "1155.kl"],
    "CIMB":               ["cimb", "cimb group", "1023", "1023.kl"],
    "Public Bank":        ["public bank", "pbbank", "pbb", "1295", "1295.kl"],
    "TNB":                ["tenaga", "tenaga nasional", "tnb", "5347", "5347.kl"],
    "Petronas Chemicals": ["petronas chemicals", "petchem", "5183", "5183.kl"],
    "Maxis":              ["maxis", "maxis berhad", "6012", "6012.kl"],
    "Axiata":             ["axiata", "axiata group", "6888", "6888.kl"],
    "Genting":            ["genting", "genting berhad", "genting bhd", "3182", "3182.kl"],
    "KL Kepong":          ["kl kepong", "klk", "kuala lumpur kepong", "2445", "2445.kl"],
    "IHH Healthcare":     ["ihh", "ihh healthcare", "5225", "5225.kl"],
}


def fetch_data(symbol, period="5y", config=CONFIG):
    _require(_HAS_YF, "yfinance", "market data download")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df.ffill()


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
    return compute_indicators(df, config).dropna().copy()


class StockPreprocessor:
    FEATURES = FEATURES

    def __init__(self, window_size=None, config=CONFIG):
        self.config = config
        self.window_size = window_size or config.window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = list(self.FEATURES)

    def get_feature_matrix(self, df):
        return df[self.features].values

    def build_sequences(self, scaled):
        X, y = [], []
        for i in range(self.window_size, len(scaled)):
            X.append(scaled[i - self.window_size:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def prepare_data(self, df):
        scaled_full = self.scaler.fit_transform(self.get_feature_matrix(df))
        X, y = self.build_sequences(scaled_full)
        return X, y, scaled_full

    def inverse_transform(self, pred, scaler=None):
        sc = scaler if scaler is not None else self.scaler
        dummy = np.zeros((len(pred), len(self.features)))
        dummy[:, 0] = np.asarray(pred).flatten()
        return sc.inverse_transform(dummy)[:, 0]

    def get_last_window_3d(self, scaled_full):
        return scaled_full[-self.window_size:][np.newaxis, ...]

    def get_last_window_2d(self, scaled_full):
        return scaled_full[-self.window_size:].flatten()[np.newaxis, :]

    def scale_row(self, row):
        return np.clip(self.scaler.transform(row), 0.0, 1.0)


def _compile(model, learning_rate=None):
    optimizer = Adam(learning_rate=learning_rate) if learning_rate else "adam"
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def build_lstm(input_shape, units=50, dropout=0.2, learning_rate=None):
    _require(_HAS_TF, "tensorflow", "the LSTM model")
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1),
    ], name="LSTM")
    return _compile(model, learning_rate)


def build_gru(input_shape, units=50, dropout=0.2, learning_rate=None):
    _require(_HAS_TF, "tensorflow", "the GRU model")
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
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


def tune_xgboost(X, y, n_iter=20, cv_splits=3, random_state=42):
    X_flat = X.reshape(len(X), -1)
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


def tune_xgboost_optuna(X, y, n_trials=30, cv_splits=3, timeout=None, random_state=42):
    if not _HAS_OPTUNA:
        return tune_xgboost(X, y, n_iter=n_trials, cv_splits=cv_splits, random_state=random_state)
    _require(_HAS_XGB, "xgboost", "XGBoost tuning")
    X_flat = X.reshape(len(X), -1)
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


def optimize_hyperparameters(model_type, X, y, method="auto", config=CONFIG, **kwargs):
    use_optuna = (method == "optuna") or (method == "auto" and _HAS_OPTUNA)
    if model_type == "XGBoost":
        return tune_xgboost_optuna(X, y, **kwargs) if use_optuna else tune_xgboost(X, y, **kwargs)
    if model_type in SEQUENCE_MODELS:
        if use_optuna:
            return tune_keras_optuna(X, y, model_type=model_type, config=config, **kwargs)
        return tune_keras(X, y, model_type=model_type, config=config, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def directional_accuracy(y_true, y_pred):
    actual_dir = np.diff(y_true.flatten()) > 0
    pred_dir = np.diff(y_pred.flatten()) > 0
    return float(np.mean(actual_dir == pred_dir))


def sharpe_ratio(y_true, y_pred, risk_free=0.0):
    signal = (np.diff(y_pred.flatten()) > 0).astype(float) * 2 - 1
    ret = signal * np.diff(y_true.flatten()) / (y_true.flatten()[:-1] + 1e-9)
    excess = ret - risk_free / 252
    if excess.std() < 1e-9:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def walk_forward_validation(df, preprocessor, model_type="XGBoost", n_splits=None,
                            epochs=None, model_params=None, config=CONFIG):
    n_splits = n_splits or config.n_splits
    epochs = epochs or config.epochs
    W = preprocessor.window_size
    raw = preprocessor.get_feature_matrix(df)
    n_seq = len(raw) - W
    if n_seq <= n_splits:
        raise ValueError(f"Not enough data ({n_seq} sequences) for {n_splits} splits.")
    fold_size = n_seq // (n_splits + 1)
    maes, accs, sharpes = [], [], []
    all_preds_rm, all_acts_rm = [], []

    for i in range(n_splits):
        train_end_seq = fold_size * (i + 1)
        test_start_seq = train_end_seq
        test_end_seq = min(test_start_seq + fold_size, n_seq)
        train_raw_end = W + train_end_seq
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(raw[:train_raw_end])
        scaled_all = scaler.transform(raw)
        X_all, y_all = preprocessor.build_sequences(scaled_all)
        X_tr, y_tr = X_all[:train_end_seq], y_all[:train_end_seq]
        X_te = X_all[test_start_seq:test_end_seq]
        y_te = y_all[test_start_seq:test_end_seq]
        if len(X_te) == 0:
            continue

        if model_type == "XGBoost":
            model = build_xgboost(model_params)
            model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
            preds_scaled = model.predict(X_te.reshape(len(X_te), -1))
        elif model_type in SEQUENCE_MODELS:
            mp = model_params or {}
            seq_kw = {k: mp[k] for k in ("units", "dropout", "learning_rate") if k in mp}
            batch_size = mp.get("batch_size", config.batch_size)
            model = build_sequence_model(model_type, (X_tr.shape[1], X_tr.shape[2]), **seq_kw)
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      validation_split=config.validation_split, callbacks=_build_callbacks(config), verbose=0)
            preds_scaled = model.predict(X_te, verbose=0).flatten()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        true_rm = preprocessor.inverse_transform(y_te, scaler=scaler)
        preds_rm = preprocessor.inverse_transform(preds_scaled, scaler=scaler)
        maes.append(mean_absolute_error(true_rm, preds_rm))
        accs.append(directional_accuracy(true_rm, preds_rm))
        sharpes.append(sharpe_ratio(true_rm, preds_rm))
        all_preds_rm.extend(preds_rm.tolist())
        all_acts_rm.extend(true_rm.tolist())

    return {
        "avg_mae": float(np.mean(maes)) if maes else float("nan"),
        "avg_acc": float(np.mean(accs)) if accs else float("nan"),
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else float("nan"),
        "predictions_rm": np.array(all_preds_rm), "actuals_rm": np.array(all_acts_rm),
        "predictions": np.array(all_preds_rm), "actuals": np.array(all_acts_rm),
    }


def compute_shap_importance(X, y, preprocessor, n_background=100, model_params=None):
    _require(_HAS_SHAP, "shap", "SHAP feature importance")
    X_flat = X.reshape(X.shape[0], -1)
    model = build_xgboost(model_params)
    model.fit(X_flat, y)
    n_bg = min(n_background, len(X_flat))
    background = X_flat[:n_bg]
    explainer = shap.TreeExplainer(model, background)
    shap_vals = explainer.shap_values(background)
    n_features = len(preprocessor.features)
    window = preprocessor.window_size
    shap_3d = shap_vals.reshape(n_bg, window, n_features)
    importance = np.abs(shap_3d).mean(axis=(0, 1))
    return pd.DataFrame({"Feature": preprocessor.features, "SHAP Importance": np.round(importance, 6)}
                        ).sort_values("SHAP Importance", ascending=False).reset_index(drop=True)


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
            self._model.fit(X.reshape(len(X), -1), y)
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
        n_days = n_days or self.config.forecast_horizon
        scaled = self._scaled_full.copy()
        work = self._work.copy() if self._work is not None else None
        preds_rm = []
        range_frac, avg_volume = 0.0, 0.0
        if work is not None:
            lb = self.config.range_lookback
            if {"High", "Low"}.issubset(work.columns):
                rng = ((work["High"] - work["Low"]) / work["Close"]).tail(lb)
                range_frac = float(rng.mean()) if rng.notna().any() else 0.0
            if "Volume" in work.columns:
                avg_volume = float(work["Volume"].tail(lb).mean())

        for _ in range(n_days):
            window = scaled[-self.proc.window_size:]
            if self._mtype == "XGBoost":
                x = window.flatten()[np.newaxis, :]
                pred_scaled = float(self._model.predict(x)[0])
            else:
                x = window[np.newaxis, ...]
                pred_scaled = float(self._model.predict(x, verbose=0).flatten()[0])
            pred_rm = float(self.proc.inverse_transform(np.array([pred_scaled]))[0])
            preds_rm.append(pred_rm)

            if work is not None:
                new_row = {"Close": pred_rm, "High": pred_rm * (1 + range_frac / 2),
                          "Low": pred_rm * (1 - range_frac / 2), "Volume": avg_volume}
                new_row = {k: v for k, v in new_row.items() if k in work.columns}
                work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
                feat_row = compute_indicators(work, self.config)[self.proc.features].iloc[[-1]].values
                new_scaled = scaled[[-1]] if np.isnan(feat_row).any() else self.proc.scale_row(feat_row)
            else:
                last = scaled[-1].copy()
                last[0] = float(np.clip(pred_scaled, 0.0, 1.0))
                new_scaled = last[np.newaxis, :]

            scaled = np.vstack([scaled, new_scaled])

        return np.array(preds_rm)


def direct_multistep_forecast_xgb(df, preprocessor, horizon=None, model_params=None, config=CONFIG):
    horizon = horizon or config.forecast_horizon
    raw = preprocessor.get_feature_matrix(df)
    W = preprocessor.window_size
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)
    X, _ = preprocessor.build_sequences(scaled)
    X_flat = X.reshape(len(X), -1)
    close_scaled = scaled[:, 0]
    last_window = scaled[-W:].flatten()[np.newaxis, :]
    preds_scaled = []
    for h in range(1, horizon + 1):
        max_j = len(X) - (h - 1)
        if max_j <= 0:
            break
        X_h = X_flat[:max_j]
        y_h = np.array([close_scaled[W + j - 1 + h] for j in range(max_j)])
        model = build_xgboost(model_params)
        model.fit(X_h, y_h)
        preds_scaled.append(float(model.predict(last_window)[0]))
    return preprocessor.inverse_transform(np.array(preds_scaled), scaler=scaler)


def _forecast_error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / (actual + 1e-9))) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def compare_forecasting_strategies(raw_df, preprocessor, horizon=None, n_origins=5, model_params=None, config=CONFIG):
    _require(_HAS_XGB, "xgboost", "forecast strategy comparison")
    horizon = horizon or config.forecast_horizon
    if "Close" not in raw_df.columns:
        raise ValueError("raw_df must contain a 'Close' column.")
    raw_df = raw_df.reset_index(drop=True)
    W = preprocessor.window_size
    min_history = W + max(config.ma_long, config.bb_period) + 20
    n = len(raw_df)
    latest_origin = n - horizon
    origins = [latest_origin - k * horizon for k in range(n_origins)]
    origins = sorted(o for o in origins if o >= min_history)
    if not origins:
        raise ValueError(f"Not enough data for comparison: need >= {min_history + horizon} rows, have {n}.")
    rec_rows, dir_rows, detail = [], [], []
    for origin in origins:
        hist_raw = raw_df.iloc[:origin].copy()
        actual = raw_df["Close"].iloc[origin:origin + horizon].to_numpy(dtype=float)
        if len(actual) < horizon:
            continue
        hist_feat = add_features(hist_raw, config)
        if len(hist_feat) <= W + 1:
            continue
        proc = StockPreprocessor(window_size=W, config=config)
        direct_pred = direct_multistep_forecast_xgb(hist_feat, proc, horizon=horizon, model_params=model_params, config=config)
        Xc, yc, scaledc = proc.prepare_data(hist_feat)
        champ = ChampionForecaster(proc, config=config).fit(Xc, yc, scaledc, "XGBoost", raw_df=hist_raw, model_params=model_params)
        recursive_pred = champ.forecast(horizon)
        h = min(horizon, len(direct_pred), len(recursive_pred))
        rec_rows.append((actual[:h], np.asarray(recursive_pred[:h], dtype=float)))
        dir_rows.append((actual[:h], np.asarray(direct_pred[:h], dtype=float)))
        for step in range(h):
            detail.append({"Origin": origin, "Step": step + 1, "Actual": round(float(actual[step]), 4),
                          "Recursive": round(float(recursive_pred[step]), 4), "Direct": round(float(direct_pred[step]), 4)})
    if not rec_rows:
        raise ValueError("No valid origins produced forecasts for comparison.")
    rec_actual = np.concatenate([a for a, _ in rec_rows])
    rec_pred = np.concatenate([p for _, p in rec_rows])
    dir_actual = np.concatenate([a for a, _ in dir_rows])
    dir_pred = np.concatenate([p for _, p in dir_rows])
    rec_metrics = _forecast_error_metrics(rec_actual, rec_pred)
    dir_metrics = _forecast_error_metrics(dir_actual, dir_pred)
    detail_df = pd.DataFrame(detail)
    per_step = (detail_df.assign(Recursive_AE=lambda d: (d["Recursive"] - d["Actual"]).abs(),
                                 Direct_AE=lambda d: (d["Direct"] - d["Actual"]).abs())
                .groupby("Step")[["Recursive_AE", "Direct_AE"]].mean().round(4)
                .rename(columns={"Recursive_AE": "Recursive_MAE", "Direct_AE": "Direct_MAE"}).reset_index())
    recommended = "direct" if dir_metrics["RMSE"] <= rec_metrics["RMSE"] else "recursive"
    return {"horizon": horizon, "n_origins_used": len(rec_rows), "recursive": rec_metrics,
            "direct": dir_metrics, "per_step": per_step, "detail": detail_df, "recommended": recommended}


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
    if term in title_lower:
        return True
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
    prompt = f"""You are an elite Financial Intelligence Analyst.
Company: {company_name}
Headlines:
{headline_block}
Macro: OPR={macro['OPR']}% CPI={macro['CPI']}% GDP={macro['GDP_Growth']}%
Respond with ONLY JSON: sentiment_label, macro_impact, qualitative_rationale"""
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
    return {"sentiment_label": label,
            "macro_impact": parsed.get("macro_impact", "No macro impact assessment available."),
            "qualitative_rationale": parsed.get("qualitative_rationale", "No qualitative rationale available.")}


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
    breach_idx = next((i for i, p in enumerate(forecast_prices) if p < threshold), None)
    breached = breach_idx is not None
    if breached:
        advisory = (f"Forecast breaches the {stop_pct*100:.0f}% stop-loss on Day {breach_idx + 1} "
                    f"(RM {forecast_prices[breach_idx]:.4f}). Consider a sell order at RM {threshold:.4f}.")
    else:
        max_fc = float(np.max(forecast_prices))
        exp_ret = (max_fc - current_price) / current_price * 100
        advisory = (f"No stop-loss breach forecast over {len(forecast_prices)} days. "
                    f"Peak forecast RM {max_fc:.4f} ({exp_ret:+.2f}% vs today).")
    return {"threshold": round(threshold, 4), "breached": breached, "breach_day": breach_idx,
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