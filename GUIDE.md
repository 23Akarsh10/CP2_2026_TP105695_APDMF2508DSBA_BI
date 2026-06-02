# Stock Predictor — Complete Step-by-Step Coding Guide

This guide tells you exactly what to type, where to type it,
and why it works. Follow it in order. Do not skip steps.
Test after every step before moving to the next one.

---

## Your Daily Routine — Do This Every Time

Open Anaconda Prompt from the Start Menu. Then run these
two commands one at a time:

```cmd
conda activate stock_predictor
cd "C:\Users\akars\OneDrive\Desktop\AK\Notes\CAPSTONE PROJECT\CP2\Second combined"
```

Open VS Code:
```cmd
code .
```

If code . does not work, open VS Code from the Start Menu
then go to File → Open Folder → select your project folder.

Check the bottom left corner of VS Code. It must say:
```
Python 3.10.x ('stock_predictor': conda)
```

If it says anything else, press Ctrl+Shift+P, type
"Python: Select Interpreter" and pick stock_predictor.

Open the terminal inside VS Code by pressing Ctrl + `

You are now ready to code.

---
---

# MILESTONE 1 — MAY 31
# Data Collection + Feature Engineering

What you are building:
The part of engine.py that defines the 10 stocks, downloads
5 years of price data for all of them, cleans it, calculates
the technical indicators, and normalises it into sequences
ready for machine learning.

This maps to sections 3.3, 3.4 and 3.5 of your methodology.

---

## Step 1 — Create engine.py

In VS Code, look at the Explorer panel on the left side.
Right-click in the empty space below your files.
Click New File.
Type: engine.py
Press Enter.

A blank file opens on the right side of VS Code.

---

## Step 2 — Add imports

What you are doing:
Imports load all the external tools Python needs before
your code runs. Think of it like gathering all your
equipment before starting a job. Every Python file starts
with imports at the very top.

What each import does:
- os: controls operating system settings
- random: Python's built-in random number generator
- warnings: lets you silence harmless messages
- numpy: mathematical arrays and calculations
- pandas: tables of data, like Excel in Python
- yfinance: downloads stock market data from Yahoo Finance
- MinMaxScaler: normalises numbers to a 0-1 range
- mean_absolute_error: measures how wrong predictions are
- XGBRegressor: the XGBoost machine learning model
- SentimentIntensityAnalyzer: VADER sentiment tool
- TextBlob: TextBlob sentiment tool
- joblib: saves and loads machine learning models to disk
- shap: explains what features drive XGBoost predictions
- tensorflow and keras layers: builds LSTM and GRU models
- EarlyStopping: stops training when it stops improving

Type this at the very top of engine.py:

```python
import os
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib
import shap

# These two lines must come BEFORE the tensorflow import.
# They suppress the long startup messages TensorFlow prints.
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

Press Ctrl+S to save.

---

## Step 3 — Add reproducibility seeds and constants

What you are doing:
Setting fixed random seeds so your model training produces
the same results every time you run it. This is required
for academic reproducibility — your supervisor or examiner
should be able to run the code and get the same numbers.

Also defining the model cache folder where trained models
will be saved to disk.

Type this below the imports:

```python
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
```

Press Ctrl+S to save.

---

## Step 4 — Define the 10 stock universe

What you are doing:
Defining the 10 highly liquid Bursa Malaysia stocks as a
dictionary. This is the complete stock universe for the
entire project. Section 3.4 of your methodology specifies
these stocks were selected on the basis of high average
daily trade volume.

The keys are the company names shown in the dashboard.
The values are Yahoo Finance ticker symbols. All Bursa
Malaysia tickers end in .KL

Type this below the constants:

```python
# The 10 highly liquid Bursa Malaysia stocks.
# Selected on the basis of high average daily trade volume.
# Section 3.4 of the methodology.
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
```

Press Ctrl+S to save.

---

## Step 5 — Write fetch_data()

What you are doing:
Writing a function that connects to Yahoo Finance and
downloads 5 years of daily OHLCV data for any stock ticker.

OHLCV means:
O = Open price (price when market opened that day)
H = High price (highest price reached that day)
L = Low price (lowest price reached that day)
C = Close price (price when market closed) — this is
    what the models predict
V = Volume (how many shares were traded)

Forward imputation is applied to fill any gaps caused by
public holidays where no trading occurred. Without it,
those gaps would appear as NaN values and confuse the
machine learning models.

Type this below BURSA_10:

```python
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
```

Press Ctrl+S to save.

Test — type this in the VS Code terminal and press Enter:

```cmd
python -c "from engine import fetch_data; df = fetch_data('1155.KL'); print(df[['Open','High','Low','Close','Volume']].tail())"
```

What you should see:
A table showing the last 5 rows of Maybank data.
Each row is one trading day.

Now test all 10 stocks at once:

```cmd
python -c "
from engine import fetch_data, BURSA_10
for name, ticker in BURSA_10.items():
    try:
        df = fetch_data(ticker)
        print(f'OK   {name:<22} {ticker}   rows={len(df)}')
    except Exception as e:
        print(f'FAIL {name:<22} {ticker}   {e}')
"
```

What you should see:
Ten lines all starting with OK, each showing around
1200 to 1260 rows (five years of trading days).

---

## Step 6 — Write add_features()

What you are doing:
Writing a function that calculates the five technical
indicators specified in section 3.5 of your methodology.

Why not just use raw prices?
Raw prices alone do not tell the model about momentum,
trend direction, or whether a stock is overbought.
Technical indicators summarise these patterns into numbers
the model can learn from.

What each indicator means:
MA20: Average of the last 20 closing prices at each step.
  Smooths short-term price noise.
MA50: Average of the last 50 closing prices.
  Captures longer-term trends.
RSI: Relative Strength Index. A number between 0 and 100.
  Above 70 means overbought. Below 30 means oversold.
MACD: Difference between 12-day and 26-day EMA.
  Positive means short-term momentum is stronger.
Signal: 9-day EMA of MACD. Used to confirm direction.

Type this below fetch_data():

```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates MA20, MA50, RSI(14), MACD(12/26), Signal(9).
    These form the feature matrix described in section 3.5.

    Drops the first rows where indicators cannot be calculated
    yet because not enough history exists. MA50 needs 50 days
    of history before it can produce a value, so the first
    ~50 rows are removed by dropna().
    """
    df = df.copy()

    # Moving averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI (14-day)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)    # 1e-9 prevents division by zero
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD and Signal line
    ema12        = df["Close"].ewm(span=12, adjust=False).mean()
    ema26        = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df.dropna()
```

Press Ctrl+S to save.

Test:

```cmd
python -c "
from engine import fetch_data, add_features
df = fetch_data('1155.KL')
df = add_features(df)
print(df[['Close','MA20','MA50','RSI','MACD','Signal']].tail())
print('Shape:', df.shape)
"
```

What you should see:
A table with 6 columns of numbers. No NaN values anywhere.
RSI will be between 0 and 100.

---

## Step 7 — Write _recompute_indicators()

What you are doing:
Writing a private helper function that recalculates all
five technical indicators from a given Close price series.

Why is this needed?
During the 5-day forecast, each new predicted price is
added to the running series. After each addition, all
indicators must be recalculated from scratch using the
updated series so the next forecast step sees realistic
feature values rather than stale historical ones.

The underscore at the start of the name means this function
is private — it is only used inside engine.py and is not
called directly from app.py.

Type this below add_features():

```python
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
```

Press Ctrl+S to save.

---

## Step 8 — Write StockPreprocessor class

What you are doing:
Writing a class that normalises the feature data and
builds the 60-day sliding window sequences that the
machine learning models learn from.

What is a class?
A class is a blueprint. You define it once and create
instances of it. Each instance remembers its own scaler
and settings. Classes in Python start with the word class.

Why normalise?
Close price is around 9.50 RM.
RSI is between 0 and 100.
Volume might be 5,000,000.
These are completely different scales. Neural networks
struggle to learn when features are on different scales
because the large numbers dominate the small ones.
Normalising to [0, 1] puts all features on equal footing.

What is a sliding window?
Instead of feeding the model one day at a time, we feed
it 60 consecutive days at once. The model looks at 60
days of history and predicts the next day's price.

Window 1: Days 1-60  → predict day 61
Window 2: Days 2-61  → predict day 62
Window 3: Days 3-62  → predict day 63
Each of these is one training sample.

Type this below _recompute_indicators():

```python
class StockPreprocessor:
    """
    Handles Min-Max normalisation and 60-day sliding window
    sequence building. Section 3.5 of the methodology.

    Close must remain at index 0 in FEATURES because it is
    the prediction target and inverse_transform uses column 0.
    """

    FEATURES = ["Close", "MA20", "MA50", "RSI", "MACD", "Signal"]

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
        X           : input sequences, shape (N, 60, 6)
                      N windows, each 60 days, 6 features
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

        row shape: (1, 6)
        """
        scaled = self.scaler.transform(row)
        return np.clip(scaled, 0.0, 1.0)
```

Press Ctrl+S to save.

Test:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor()
X, y, scaled = proc.prepare_data(df)
print('X shape:', X.shape)
print('y shape:', y.shape)
print('Expected X shape: (around 1150, 60, 6)')
"
```

What you should see:
X shape: (N, 60, 6) where N is around 1100 to 1200.

---

## Milestone 1 Git Commit

In the VS Code terminal:

```cmd
git add .
git commit -m "Milestone 1 - Data collection and feature engineering complete"
```

What to show your supervisor:
Run the all-10-stocks test from Step 5 live.
Run the features test from Step 6 live.
Run the shape test from Step 8 live.
Explain: fetch_data downloads real data for all 10 stocks.
add_features calculates the 5 technical indicators.
StockPreprocessor normalises and builds 60-day windows.

---
---

# MILESTONE 2 — JUNE 28
# Model Tournament + Walk-Forward Validation + Sentiment

What you are building:
The three model architectures, the walk-forward validation
system that trains and compares them, the evaluation metrics,
and the sentiment pipeline.

This maps to sections 3.6, 3.7 and the sentiment part of
3.4 in your methodology.

Do not delete anything from engine.py.
Keep adding below what is already there.

---

## Step 9 — Write the three model builders

What you are doing:
Writing three functions that build untrained machine learning
models ready to be trained on your data.

LSTM: Long Short-Term Memory. A recurrent neural network
that can learn long-term patterns in sequential data.
It has internal memory cells that decide what to remember
and what to forget. 2 layers with Dropout regularisation.

GRU: Gated Recurrent Unit. Similar to LSTM but with fewer
internal parameters. Trains faster and often performs just
as well. 2 layers with Dropout regularisation.

XGBoost: Extreme Gradient Boosting. A tree-based model that
builds many decision trees in sequence, each one correcting
the errors of the previous. Does not use sequences —
it receives a flat row of 360 numbers (60 days x 6 features).

Dropout(0.2) randomly disables 20% of neurons during
training. This prevents the model from memorising the
training data too precisely (called overfitting) and forces
it to learn more general patterns.

Type this below StockPreprocessor:

```python
def build_lstm(input_shape):
    """
    2-layer stacked LSTM as specified in section 3.6.
    input_shape: (60, 6) — 60 days, 6 features.
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
```

Press Ctrl+S to save.

---

## Step 10 — Write the evaluation metric functions

What you are doing:
Writing three functions that measure model performance.
These are computed on actual RM prices, not on scaled
values, so the numbers are directly interpretable.

MAE (Mean Absolute Error):
The average RM distance between predicted and actual price.
If MAE = 0.08, predictions are on average 8 sen from real.
Lower is better. This is the primary champion selector.

Directional Accuracy:
The percentage of steps where the model correctly predicted
whether price went up or down from the previous real close.
50% = same as a coin flip. Above 50% = better than random.
This uses the actual previous price as baseline, not the
previous prediction, to avoid compounding direction errors.

Sharpe Ratio:
Simulates a strategy that goes long when the model predicts
a rise and short when it predicts a fall. Measures how much
return is earned per unit of risk taken, annualised.
Above 1.0 is good. Documented as an approximation because
it does not account for transaction costs or slippage.

Type this below the model builders:

```python
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
```

Press Ctrl+S to save.

---

## Step 11 — Write walk_forward_validation()

What you are doing:
Writing the function that trains each model using
walk-forward cross-validation and returns averaged metrics.

Why walk-forward instead of a normal train/test split?
A normal split (train on 80%, test on last 20%) would
allow the model to train on data from 2023 and test on
data from 2020. That is look-ahead bias — the model has
effectively seen the future.

Walk-forward always trains on the past and tests on what
comes immediately after, exactly simulating real trading.

How expanding window walk-forward works:
The data is split into n_splits+1 equal chunks.
Fold 1: train on chunk 1,       test on chunk 2
Fold 2: train on chunks 1-2,    test on chunk 3
Fold 3: train on chunks 1-3,    test on chunk 4
Fold 4: train on chunks 1-4,    test on chunk 5
Fold 5: train on chunks 1-5,    test on chunk 6
Training window grows each fold. Metrics are averaged.

All metrics are computed after inverse-transforming
predictions back to RM prices, not on scaled values.
This makes MAE directly readable as a RM error value.

Type this below the metric functions:

```python
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
            # Flatten 3D to 2D: (N, 60, 6) becomes (N, 360)
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
```

Press Ctrl+S to save.

Test — XGBoost is fastest so test it first:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, walk_forward_validation
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor()
X, y, scaled = proc.prepare_data(df)
res = walk_forward_validation(X, y, proc, model_type='XGBoost', n_splits=3)
print('MAE (RM)         :', round(res['avg_mae'], 4))
print('Dir. Accuracy    :', round(res['avg_acc']*100, 1), '%')
print('Sharpe Ratio     :', round(res['avg_sharpe'], 3))
print('Actuals in RM    :', res['actuals_rm'][:3].round(4))
"
```

This takes around 30 to 60 seconds. That is normal.
MAE will be in RM — a number like 0.08 or 0.15.
Directional Accuracy will be between 40% and 70%.

---

## Step 12 — Write compute_shap_importance()

What you are doing:
Writing a function that uses SHAP to explain which features
drive the XGBoost model's predictions.

What is SHAP?
SHAP stands for SHapley Additive exPlanations. It assigns
an importance score to each input feature by calculating
how much each feature contributed to each prediction.
This makes the XGBoost model interpretable — you can show
which of the 6 features matters most.

Important note: SHAP is only applied to XGBoost here.
LSTM and GRU are recurrent neural networks whose internal
behaviour cannot be explained with SHAP TreeExplainer.
This asymmetry is acknowledged as a limitation.

This function trains XGBoost on the actual forecasting
target y (not a proxy) so the SHAP values explain the
real deployed model behaviour.

Type this below walk_forward_validation():

```python
def compute_shap_importance(X:            np.ndarray,
                             y:            np.ndarray,
                             preprocessor: StockPreprocessor,
                             n_background: int = 100) -> pd.DataFrame:
    """
    XGBoost SHAP feature importance using the true target y.

    Trains XGBoost on (X, y) — the same pair used in
    walk_forward_validation — then applies SHAP TreeExplainer
    to quantify the contribution of each feature.

    Since the input is a flattened 60-day x 6-feature window
    (360 columns total), SHAP values are aggregated across
    all 60 time steps to produce one score per feature.

    Returns a DataFrame ['Feature', 'SHAP Importance'] sorted
    by importance descending.
    """
    # Flatten: (N, 60, 6) → (N, 360)
    X_flat = X.reshape(X.shape[0], -1)

    # Train XGBoost on the TRUE forecasting target y
    model = build_xgboost()
    model.fit(X_flat, y)

    # SHAP TreeExplainer for tree-based models
    n_bg       = min(n_background, len(X_flat))
    background = X_flat[:n_bg]
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(background)

    # shap_vals shape: (n_bg, 360)
    # Reshape to (n_bg, 60 days, 6 features) then average
    # across samples and days to get one score per feature
    n_features = len(preprocessor.features)
    window     = preprocessor.window_size
    shap_3d    = shap_vals.reshape(n_bg, window, n_features)
    importance = np.abs(shap_3d).mean(axis=(0, 1))    # shape (6,)

    df = pd.DataFrame({
        "Feature":         preprocessor.features,
        "SHAP Importance": np.round(importance, 6),
    }).sort_values("SHAP Importance",
                   ascending=False).reset_index(drop=True)

    return df
```

Press Ctrl+S to save.

Test:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, compute_shap_importance
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor()
X, y, scaled = proc.prepare_data(df)
shap_df = compute_shap_importance(X, y, proc, n_background=50)
print('SHAP Feature Importance:')
print(shap_df.to_string(index=False))
"
```

What you should see:
A table with Feature and SHAP Importance columns.
Close will typically have the highest importance.

---

## Step 13 — Write the sentiment functions

What you are doing:
Writing two functions — a private helper that classifies
scores, and a main function that fetches live news headlines
from Yahoo Finance and scores them with VADER and TextBlob.

Why no model training for sentiment?
VADER and TextBlob are pre-trained lexicon tools. VADER
has a pre-built dictionary of 7,500 words each scored by
human raters. It looks up each word in the headline,
sums the scores, and returns a result. No training loop.

This is appropriate because:
1. We have no labelled Bursa Malaysia sentiment dataset
2. Sentiment is a qualitative overlay, not a predictor
3. VADER was designed for short financial text

Limitation to acknowledge: neither VADER nor TextBlob
handles Bahasa Malaysia vocabulary or Islamic finance
terms. Words not in their dictionaries score as 0.

Type this below compute_shap_importance():

```python
def _sentiment_label(score: float) -> str:
    """
    Converts a numerical score to a text classification.
    Positive if score > 0.05.
    Negative if score < -0.05.
    Neutral otherwise.
    """
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def benchmark_sentiment(symbol: str,
                         max_headlines: int = 10) -> dict:
    """
    Fetches the latest news headlines for a stock from Yahoo
    Finance and scores each one with VADER and TextBlob.

    Returns per-headline scores and an ensemble average.
    Ensemble = average of VADER and TextBlob scores.

    Limitation: lexicon tools score unknown words as 0.
    This affects Bahasa Malaysia terms and Islamic finance
    vocabulary. Acknowledged in the limitations chapter.
    """
    analyzer = SentimentIntensityAnalyzer()
    ticker   = yf.Ticker(symbol)

    try:
        news = ticker.news or []
    except Exception:
        news = []

    headlines = []
    for item in news[:max_headlines]:
        title = (item.get("title") or
                 item.get("headline") or "").strip()
        if not title:
            continue

        v   = analyzer.polarity_scores(title)["compound"]
        t   = TextBlob(title).sentiment.polarity
        avg = (v + t) / 2.0

        headlines.append({
            "title":    title,
            "vader":    round(v,   4),
            "textblob": round(t,   4),
            "avg":      round(avg, 4),
            "label":    _sentiment_label(avg),
        })

    if headlines:
        vader_avg    = np.mean([h["vader"]    for h in headlines])
        textblob_avg = np.mean([h["textblob"] for h in headlines])
    else:
        vader_avg = textblob_avg = 0.0

    ensemble = (vader_avg + textblob_avg) / 2.0

    return {
        "VADER":     round(float(vader_avg),    4),
        "TextBlob":  round(float(textblob_avg), 4),
        "Ensemble":  round(float(ensemble),     4),
        "V_Label":   _sentiment_label(vader_avg),
        "T_Label":   _sentiment_label(textblob_avg),
        "E_Label":   _sentiment_label(ensemble),
        "headlines": headlines,
        "n":         len(headlines),
    }
```

Press Ctrl+S to save.

Test:

```cmd
python -c "
from engine import benchmark_sentiment
result = benchmark_sentiment('1155.KL')
print('Overall label  :', result['E_Label'])
print('Ensemble score :', result['Ensemble'])
print('Headlines found:', result['n'])
for h in result['headlines'][:3]:
    print(f'  [{h[\"label\"]:8s}] {h[\"title\"][:65]}')
"
```

---

## Milestone 2 Git Commit

```cmd
git add .
git commit -m "Milestone 2 - Model tournament and sentiment pipeline complete"
```

What to show your supervisor:
Run the XGBoost walk-forward test live from Step 11.
Show MAE in RM, directional accuracy, and Sharpe Ratio.
Run the SHAP test from Step 12 and show feature ranking.
Run the sentiment test from Step 13 with live headlines.
Explain walk-forward vs normal train/test split.
Explain why VADER needs no training.
Explain all three evaluation metrics.

---
---

# MILESTONE 3 — JULY 12
# Champion Forecaster + Stop-Loss + UI + Integration

What you are building:
The champion forecaster that produces the 5-day price
forecast with full indicator recomputation, the stop-loss
advisory, the portfolio ranking, and the complete Streamlit
dashboard in app.py.

Do not delete anything from engine.py.
Keep adding below what is already there.

---

## Step 14 — Write ChampionForecaster class

What you are doing:
Writing a class that takes the champion model (the one
with the lowest MAE from walk-forward validation), retrains
it on all available data, and forecasts the next 5 days.

Why retrain on all data?
During walk-forward validation, some data was held back
for testing. Now that we know which model wins, we give
it every data point available before making a real forecast.
More data = better trained model.

Why iterative forecasting?
We do not know future prices. So to predict Day 2, we use
Day 1's prediction as input. To predict Day 3, we use Day
2's prediction. The 60-day window rolls forward one step
at a time.

After each predicted price, ALL technical indicators are
recomputed from the updated Close series using
_recompute_indicators(). This ensures MA20, MA50, RSI,
MACD and Signal are not stale during forecasting.

Model persistence: trained models are saved to the
model_cache folder. On the next session the app loads them
instead of retraining, making the dashboard much faster.

Type this below benchmark_sentiment():

```python
class ChampionForecaster:
    """
    Retrains the champion model on the full dataset then
    produces an iterative 5-day ahead forecast.

    After each predicted Close, all technical indicators are
    recomputed so the next step sees updated feature values.

    Trained models are saved to disk and reloaded on the
    next session to avoid unnecessary retraining.
    """

    def __init__(self, preprocessor: StockPreprocessor,
                 symbol: str, epochs: int = 15):
        self.proc    = preprocessor
        self.symbol  = symbol
        self.epochs  = epochs
        self._model  = None
        self._mtype  = None
        self._raw_close = None

    def _model_path(self, model_type: str) -> str:
        """Returns the file path for saving/loading this model."""
        safe = self.symbol.replace(".", "_")
        if model_type == "XGBoost":
            return os.path.join(MODEL_CACHE_DIR,
                                f"{safe}_{model_type}.joblib")
        return os.path.join(MODEL_CACHE_DIR,
                            f"{safe}_{model_type}.keras")

    def _save_model(self, model, model_type: str):
        """Saves trained model to disk."""
        path = self._model_path(model_type)
        if model_type == "XGBoost":
            joblib.dump(model, path)
        else:
            model.save(path)

    def _load_model(self, model_type: str):
        """Loads a previously saved model from disk."""
        path = self._model_path(model_type)
        if not os.path.exists(path):
            return None
        if model_type == "XGBoost":
            return joblib.load(path)
        return load_model(path)

    def fit(self, X: np.ndarray, y: np.ndarray,
            scaled_full: np.ndarray, model_type: str,
            raw_df: pd.DataFrame,
            force_retrain: bool = False):
        """
        Trains the champion model on the full dataset.
        Loads from cache if already trained and force_retrain
        is False. Saves to cache after training.

        raw_df: original unscaled DataFrame, needed for
                initialising the Close series for forecasting.
        """
        self._mtype       = model_type
        self._scaled_full = scaled_full
        self._raw_close   = (raw_df["Close"]
                             .copy()
                             .reset_index(drop=True))

        if not force_retrain:
            cached = self._load_model(model_type)
            if cached is not None:
                self._model = cached
                return

        es = EarlyStopping(patience=3,
                           restore_best_weights=True,
                           verbose=0)

        if model_type == "XGBoost":
            self._model = build_xgboost()
            self._model.fit(X.reshape(len(X), -1), y)

        elif model_type == "LSTM":
            self._model = build_lstm((X.shape[1], X.shape[2]))
            self._model.fit(X, y, epochs=self.epochs,
                            batch_size=32, validation_split=0.1,
                            callbacks=[es], verbose=0)

        elif model_type == "GRU":
            self._model = build_gru((X.shape[1], X.shape[2]))
            self._model.fit(X, y, epochs=self.epochs,
                            batch_size=32, validation_split=0.1,
                            callbacks=[es], verbose=0)

        self._save_model(self._model, model_type)

    def forecast(self, n_days: int = 5) -> np.ndarray:
        """
        Iterative n-day forecast with indicator recomputation.

        For each step:
        1. Build the current 60-day scaled window
        2. Predict next scaled Close
        3. Inverse-transform to RM
        4. Append to running Close series
        5. Recompute ALL indicators from updated series
        6. Scale new row (with clipping)
        7. Roll window forward and repeat

        Returns RM prices as an array of shape (n_days,).
        """
        close_series = self._raw_close.copy()
        scaled       = self._scaled_full.copy()
        preds_rm     = []

        for _ in range(n_days):
            window = scaled[-self.proc.window_size:]

            if self._mtype == "XGBoost":
                x           = window.flatten()[np.newaxis, :]
                pred_scaled = float(self._model.predict(x)[0])
            else:
                x           = window[np.newaxis, ...]
                pred_scaled = float(
                    self._model.predict(x, verbose=0).flatten()[0])

            pred_rm = float(self.proc.inverse_transform(
                np.array([pred_scaled]))[0])
            preds_rm.append(pred_rm)

            # Append predicted price to running Close series
            close_series = pd.concat(
                [close_series, pd.Series([pred_rm])],
                ignore_index=True)

            # Recompute all indicators from updated Close series
            new_row_df = _recompute_indicators(close_series)
            new_row    = new_row_df[self.proc.features].values

            if np.isnan(new_row).any():
                new_row_scaled = scaled[[-1]]
            else:
                new_row_scaled = self.proc.scale_row(new_row)

            scaled = np.vstack([scaled, new_row_scaled])

        return np.array(preds_rm)
```

Press Ctrl+S to save.

---

## Step 15 — Write stop_loss_advisory()

What you are doing:
Writing a function that checks whether any of the 5
forecasted prices falls below the investor's stop-loss.

A stop-loss is a pre-set price at which an investor sells
to limit losses. If you buy at RM 10.00 with a 5%
stop-loss, the threshold is RM 9.50.

Type this below ChampionForecaster:

```python
def stop_loss_advisory(current_price:   float,
                        forecast_prices: np.ndarray,
                        stop_pct:        float = 0.05) -> dict:
    """
    Checks if any forecasted price breaches the stop-loss.

    current_price   : today's closing price in RM
    forecast_prices : 5-day forecast in RM
    stop_pct        : threshold as decimal (0.05 = 5%)
    """
    threshold  = current_price * (1 - stop_pct)
    breach_idx = next(
        (i for i, p in enumerate(forecast_prices)
         if p < threshold), None)
    breached   = breach_idx is not None

    if breached:
        advisory = (
            f"Warning: Forecast breaches the "
            f"{stop_pct*100:.0f}% stop-loss on "
            f"Day {breach_idx + 1} "
            f"(RM {forecast_prices[breach_idx]:.4f}). "
            f"Consider a sell order at RM {threshold:.4f}.")
    else:
        max_fc  = float(np.max(forecast_prices))
        exp_ret = (max_fc - current_price) / current_price * 100
        advisory = (
            f"No stop-loss breach forecast over 5 days. "
            f"Peak forecast RM {max_fc:.4f} "
            f"({exp_ret:+.2f}% vs today).")

    return {
        "threshold":  round(threshold, 4),
        "breached":   breached,
        "breach_day": breach_idx,
        "advisory":   advisory,
        "stop_pct":   stop_pct,
    }
```

Press Ctrl+S to save.

---

## Step 16 — Write build_ranking_table()

What you are doing:
Writing a function that takes results from all 10 stocks
and produces a single table sorted by expected 5-day return.
This powers the Portfolio Ranking tab in the dashboard.

Type this below stop_loss_advisory():

```python
def build_ranking_table(stock_summaries: dict) -> pd.DataFrame:
    """
    Builds a ranked portfolio leaderboard sorted by expected
    5-day return. Section 3.3 of the methodology.

    stock_summaries: dict where each key is a stock name and
    each value contains: last_close, forecast, champion,
    mae, da, sharpe, sentiment_label
    """
    rows = []
    for name, s in stock_summaries.items():
        last   = s["last_close"]
        fc_end = float(s["forecast"][-1])
        ret    = (fc_end - last) / last * 100

        rows.append({
            "Stock":           name,
            "Last Close (RM)": round(last,        4),
            "5-Day Forecast":  round(fc_end,       4),
            "Expected Return": round(ret,          2),
            "Champion":        s["champion"],
            "MAE (RM)":        round(s["mae"],     4),
            "Dir. Acc (%)":    round(s["da"]*100,  1),
            "Sharpe†":         round(s["sharpe"],  3),
            "Sentiment":       s["sentiment_label"],
        })

    df = (pd.DataFrame(rows)
            .sort_values("Expected Return", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    return df
```

Press Ctrl+S to save.

engine.py is now complete.

---

## Step 17 — Create app.py

In VS Code, right-click in the Explorer panel.
Click New File. Type app.py. Press Enter.

app.py is the dashboard. It imports everything from
engine.py and displays it in a web browser using Streamlit.
It contains no machine learning logic — all of that is
in engine.py.

Copy the full app.py code from your reference file into
this blank file, then press Ctrl+S to save.

---

## Step 18 — Run the full app

In the VS Code terminal:

```cmd
streamlit run app.py
```

Your browser opens at http://localhost:8501

First load will take 2 to 5 minutes per stock while the
models train. This is normal. After the first load,
the same stock loads instantly from cache.

To stop the app press Ctrl+C in the terminal.

Verify every panel works:
- Top row: price, champion, directional accuracy, Sharpe
- Leaderboard: all 3 models with metrics
- Backtest chart: actual vs all 3 model predictions
- 5-Day Forecast: 5 day cards with RM prices and % change
- Stop-loss advisory updating with the sidebar slider
- Sentiment: VADER, TextBlob, Ensemble with headline expander
- Portfolio Ranking tab with all 10 stocks

---

## Milestone 3 Git Commit

```cmd
git add .
git commit -m "Milestone 3 - UI and system integration complete"
```

What to show your supervisor:
Live demo in the browser.
Select two different stocks and show results changing.
Move the stop-loss slider and show the advisory updating.
Open the sentiment headline expander.
Switch to the Portfolio Ranking tab.

---
---

# MILESTONE 4 — JULY 19
# Validation + Report Drafting

What you are doing:
Recording verified results for all 10 stocks and writing
the evaluation section of your capstone report.

---

## Step 19 — Record results for all 10 stocks

Open the app and select each stock one by one from the
sidebar dropdown. For each stock, copy the numbers from
the Model Leaderboard. The champion row has a trophy symbol.

| Stock | Champion | MAE (RM) | Dir. Acc % | Sharpe |
|-------|----------|----------|------------|--------|
| Maybank | | | | |
| Public Bank | | | | |
| CIMB Group | | | | |
| TNB | | | | |
| Petronas Chemicals | | | | |
| IHH Healthcare | | | | |
| Press Metal | | | | |
| CelcomDigi | | | | |
| MISC | | | | |
| Hong Leong Bank | | | | |

Take a screenshot of the Portfolio Ranking tab.
This screenshot goes in your report appendix.

---

## Step 20 — Answer these for your evaluation chapter

1. Which model wins most often across the 10 stocks?
   Does deep learning outperform XGBoost or not?

2. Is directional accuracy consistently above 50%?
   Above 50% means better than random guessing.

3. Is the Sharpe Ratio positive for most stocks?
   Positive means the strategy earns more than it risks.

4. Does sentiment label align with forecast direction?
   Note any stocks where they conflict.

5. Which stock has the lowest MAE?

6. Which stock has the highest expected 5-day return?

---

## Milestone 4 Git Commit

```cmd
git add .
git commit -m "Milestone 4 - Validation complete and report drafted"
```

---
---

# MILESTONE 5 — JULY 26
# Final Submission

Files to submit:

| File | What it contains |
|------|-----------------|
| engine.py | All data, model and sentiment code |
| app.py | Streamlit dashboard |
| requirements.txt | All Python dependencies |
| README.md | Project overview |
| GUIDE.md | This file |
| LIMITATIONS.md | Limitations chapter for the report |

Final commit and push:

```cmd
git add .
git commit -m "Milestone 5 - Final submission"
git push
```

Final checklist:
- All files are in the project folder
- App launches cleanly with streamlit run app.py
- All 10 stocks load without errors
- Portfolio Ranking tab works
- Git log shows all 5 milestone commits with dates
- Report references function names from engine.py correctly

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| ModuleNotFoundError | Wrong Python environment | conda activate stock_predictor |
| No data found for symbol | No internet | Check wifi connection |
| streamlit not recognised | Wrong environment | conda activate stock_predictor |
| Port already in use | App already running | streamlit run app.py --server.port 8502 |
| First load slow 2-5 min | Models training | Normal, wait for spinner |
| Portfolio tab slow 10-20 min | All 10 stocks training | Normal on first load, cached after |
| TensorFlow messages in terminal | Informational only | Safe to ignore |
