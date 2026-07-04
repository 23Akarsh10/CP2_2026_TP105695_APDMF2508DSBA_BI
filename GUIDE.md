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

Important — where this dictionary actually lives:
BURSA_10 is defined in app.py, NOT in engine.py. engine.py
is a general-purpose library — none of its functions know or
care which 10 stocks you track. It only ever processes one
stock at a time, whichever ticker and company name app.py
hands it. app.py is what loops over BURSA_10 and calls
engine.py functions once per stock. Keeping the stock list
out of engine.py means you could point the exact same engine
at a completely different set of 10 stocks just by editing
app.py, without touching engine.py at all.

For this step, since engine.py is what you are building right
now in Milestone 1, simply note this dictionary down — you
will type it into app.py in Milestone 3, Step 37 (Create
app.py), not here. The test commands below create a temporary
local copy purely so you can test fetch_data() against all 10
stocks while engine.py is still under construction.

The keys are the company names shown in the dashboard, and
these exact names are what get passed to the sentiment agent
later as company_name — so keep them short and recognisable.
The values are Yahoo Finance ticker symbols. All Bursa
Malaysia tickers end in .KL

```python
# The 10 highly liquid Bursa Malaysia stocks.
# Selected on the basis of high average daily trade volume.
# Section 3.4 of the methodology.
# This dictionary lives in app.py — engine.py never sees it.
BURSA_10 = {
    "Maybank":            "1155.KL",
    "CIMB":               "1023.KL",
    "Public Bank":        "1295.KL",
    "TNB":                "5347.KL",
    "Petronas Chemicals": "5183.KL",
    "Maxis":              "6012.KL",
    "Axiata":             "6888.KL",
    "Genting":            "3182.KL",
    "KL Kepong":          "2445.KL",
    "IHH Healthcare":     "5225.KL",
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

Type this below the FEATURES constant near the top of engine.py
(or directly below your imports if you have not yet reached the
feature-engineering steps):

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

Now test all 10 stocks at once. Since BURSA_10 will only be
defined inside app.py later (Milestone 3, Step 37), this test
defines the same dictionary directly inside the command itself,
purely so you can confirm fetch_data() works correctly for
every ticker before app.py exists:

```cmd
python -c "
from engine import fetch_data
BURSA_10 = {
    'Maybank':            '1155.KL',
    'CIMB':               '1023.KL',
    'Public Bank':        '1295.KL',
    'TNB':                '5347.KL',
    'Petronas Chemicals': '5183.KL',
    'Maxis':              '6012.KL',
    'Axiata':             '6888.KL',
    'Genting':            '3182.KL',
    'KL Kepong':          '2445.KL',
    'IHH Healthcare':     '5225.KL',
}
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
# Model Tournament + Walk-Forward Validation + Local AI Sentiment Agent

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

## Step 13 — Install and start Ollama with the llama3 model

What you are doing:
Before writing any Python code, you need a local AI engine
running on your machine. Ollama lets you run open-source
language models like Llama 3 completely free, with no API
key and no internet dependency once the model is downloaded.

This replaces VADER and TextBlob entirely. Instead of looking
up words in a fixed dictionary, a real language model reads
the full headline in context — including Bahasa Melayu and
Mandarin headlines — and reasons about sentiment the way a
human analyst would.

Step-by-step setup:

1. Download Ollama from https://ollama.com for Windows.
   Run the installer with default settings.

2. Open a new Anaconda Prompt (Ollama installs as a separate
   background service, not inside your conda environment).

3. Pull the llama3 model. This downloads several gigabytes
   so it only needs to run once:
   ```cmd
   ollama pull llama3
   ```

4. Verify Ollama is running:
   ```cmd
   ollama list
   ```
   You should see llama3 listed.

5. Test it directly from the command line before touching
   Python:
   ```cmd
   ollama run llama3 "Say hello in one sentence"
   ```
   If you see a one-sentence reply, Ollama is working
   correctly and engine.py will be able to reach it.

Ollama runs as a background service on
http://localhost:11434 automatically after installation.
You do not need to start it manually every time — it starts
with Windows. If a command ever fails with a connection
error, run `ollama serve` in a separate terminal window.

---

## Step 14 — Install the new Python packages

What you are doing:
Adding three new libraries to your environment:
feedparser for reading RSS news feeds, and
langchain-community plus langchain-core for connecting
Python to your local Ollama server.

Type this in your Anaconda Prompt with stock_predictor active:

```cmd
pip install feedparser langchain-community langchain-core
```

Also add these three lines to your requirements.txt file
so anyone else running your project installs them too:

```
feedparser>=6.0
langchain-community>=0.2
langchain-core>=0.2
```

---

## Step 15 — Write fetch_company_headlines() — Track A

What you are doing:
Writing the first half of the dual-track data ingestion
described in your updated methodology. This function
replaces the old yfinance news ticker call with a direct
RSS feed reader using feedparser.

Why RSS instead of yfinance news?
The yfinance news ticker only returns English headlines
from US-centric sources. Five RSS feeds from The Edge
Malaysia, Bernama, Free Malaysia Today, Malay Mail and
The Sun give you genuine Malaysian financial coverage,
including a multilingual mix of English and Bahasa Melayu
headlines, which is exactly what your methodology specifies.

How RSS feeds work:
An RSS feed is a structured XML file that every news site
publishes automatically, listing their most recent articles
with titles, links and timestamps. feedparser reads that
XML and turns it into a simple Python list you can loop
through, with no scraping or HTML parsing required.

Type this below your existing engine.py imports, in a new
section. First add the feedparser import at the top of the
file alongside your other imports:

```python
import feedparser
import json
import re
```

Now add this function. Place it where benchmark_sentiment()
used to be — you will replace that function entirely in
the next steps, so this is the start of the new sentiment
section of engine.py:

```python
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
```

Press Ctrl+S to save.

Test it on its own before moving forward:

```cmd
python -c "
from engine import fetch_company_headlines
headlines = fetch_company_headlines('Maybank')
print(f'Found {len(headlines)} headlines')
for h in headlines:
    print(' -', h)
"
```

What you should see:
A small number of headlines mentioning Maybank, or zero if
none of today's articles mention it by name. Zero is a
valid and expected result — not every company appears in
the news every single day. The next step handles that case.

---

## Step 16 — Write mock_macro_indicators() — Track B

What you are doing:
Writing the second half of the dual-track ingestion. This
function provides the macroeconomic context (interest
rates, inflation, GDP growth) that the AI agent needs to
judge how broader economic conditions affect a specific
company's sector.

Why macroeconomic data matters here:
A single company headline rarely tells the full story. If
Bank Negara raises the Overnight Policy Rate, every bank
stock is affected, regardless of what that bank's own press
release says that week. Including OPR, CPI and GDP growth
lets the agent reason about sector-wide pressure, not just
isolated company news.

Why "mock" data is acceptable here:
Your specification explicitly asks for a function that
mock-fetches or streams these indicators. Live macroeconomic
data normally requires a paid subscription to a service like
Bloomberg or a government statistics API with limited
historical granularity. For this capstone, a clearly labelled
placeholder function is the methodologically honest choice —
it demonstrates the architecture without requiring paid
infrastructure, and it can be swapped for a real data source
later without changing any other part of the pipeline.

Type this below fetch_company_headlines():

```python
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
```

Press Ctrl+S to save.

Test it:

```cmd
python -c "
from engine import mock_macro_indicators
print(mock_macro_indicators())
"
```

What you should see:
A dictionary with three keys: OPR, CPI and GDP_Growth.

---

## Step 17 — Write build_agent_context() — bundling both tracks

What you are doing:
Writing a function that combines Track A and Track B into
a single bundle. This is the unified context window your
methodology specifies — one object that contains both the
company headlines and the macroeconomic indicators, ready
to be handed to the language model in the next step.

Type this below mock_macro_indicators():

```python
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
```

Press Ctrl+S to save.

---

## Step 18 — Write run_sentiment_agent() — the core AI pipeline

What you are doing:
Writing the central function of the entire sentiment system.
This function sends the bundled context from Step 17 to your
local Llama 3 model running through Ollama, with a carefully
structured prompt that turns Llama 3 into a financial
analyst for the Malaysian market. It parses the model's
response into clean structured data your dashboard can use.

What is LangChain and why use it here?
LangChain is a Python library that simplifies connecting to
language models. Instead of manually building HTTP requests
to talk to Ollama's local server, langchain_community.llms.Ollama
gives you a simple Python object you call like a function.

What is a prompt, and why does the wording matter?
A prompt is the instruction text you send to the language
model. Unlike VADER's fixed dictionary, Llama 3 has no
built-in rule for what counts as Positive or Negative — it
infers this from how you ask. The prompt below explicitly:
1. Assigns it a role ("elite Financial Intelligence Analyst")
2. Gives it both data tracks (headlines + macro indicators)
3. Asks it to cross-reference them, not just summarise them
4. Forces a strict output format (JSON) so your Python code
   can reliably parse the response every time

Why parse with a regex fallback?
Language models occasionally add a sentence before or after
the JSON object even when told not to ("Here is the analysis:
{...}"). The regex pulls out just the {...} portion so your
code does not break on minor formatting variation.

Type this below build_agent_context(). This is the longest
function in the sentiment module, so read through the
comments carefully before typing:

```python
from langchain_community.llms import Ollama


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
```

Press Ctrl+S to save.

Test it — make sure Ollama is running first (Step 13):

```cmd
python -c "
from engine import run_sentiment_agent
result = run_sentiment_agent('1155.KL', 'Maybank')
print('Sentiment label :', result['sentiment_label'])
print('Macro impact    :', result['macro_impact'])
print('Rationale       :', result['qualitative_rationale'])
"
```

What you should see:
The first run will take 10 to 30 seconds because Llama 3
needs to load into memory. You should see a sentiment
label, a one-to-two sentence macro impact assessment, and
a one-to-two sentence plain-English rationale. If Maybank
has no matching headlines that day, you will see the exact
default Neutral response specified in the methodology
instead — this is correct behaviour, not a bug.

If you get a connection error, open a separate terminal and
run `ollama serve`, then try the test again.

---

## Step 19 — Remove the old VADER and TextBlob code

What you are doing:
Deleting the functions you no longer need. The old
benchmark_sentiment() and _sentiment_label() functions, and
the vaderSentiment and textblob imports at the top of the
file, are now fully replaced by Steps 15 through 18.

In engine.py, delete these from your imports at the top:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
```

Then delete the entire old benchmark_sentiment() function
and the old _sentiment_label() function if you still have
them from the previous milestone. Search for
"def benchmark_sentiment" in VS Code with Ctrl+F to find
and remove that whole block, including its docstring.

Also remove vaderSentiment and textblob from
requirements.txt since they are no longer used anywhere
in the project.

Press Ctrl+S to save engine.py after deleting.

Run a quick check that nothing else in engine.py still
refers to the deleted functions:

```cmd
python -c "import engine; print('engine.py loads cleanly')"
```

If this prints without error, the old sentiment code has
been fully and safely removed.

---

## Milestone 2 Git Commit

```cmd
git add .
git commit -m "Milestone 2 - Model tournament and local AI sentiment agent complete"
```

What to show your supervisor:
Run the XGBoost walk-forward test live from Step 11.
Show MAE in RM, directional accuracy, and Sharpe Ratio.
Run the SHAP test from Step 12 and show feature ranking.
Show `ollama list` in the terminal to prove llama3 is
installed locally with no external API key.
Run the run_sentiment_agent() test from Step 18 live and
show the structured JSON-style output appearing in the
terminal — the sentiment label, the macro impact line, and
the plain-English rationale.
Explain why this replaced VADER and TextBlob: a real
language model can read Bahasa Melayu and Mandarin
headlines directly and reason about macroeconomic context,
which a fixed English word-dictionary cannot do.
Explain the graceful fallback: if no headlines exist or
Ollama is not reachable, the system returns the documented
default Neutral response instead of crashing.

---
---

# MILESTONE 2.5 — REVIEW HARDENING PASS
# Fixing Data Leakage, Adding Real Features, Tuning, and Tests

What you are building:
Your supervisor's code review came back with one MAJOR issue,
several MODERATE issues, and a second round of "Areas for
Further Improvement" once the first round was already graded
Excellent. This milestone applies every one of those fixes to
engine.py. It sits between Milestone 2 and Milestone 3 because
it changes the *interface* of walk_forward_validation() — every
later step that calls it must use the new signature.

This is the single most important milestone for your viva,
because almost every question an examiner asks about
methodological rigour will be answered by something built here.

---

## Why this milestone exists — the MAJOR issue first

Before touching any code, understand the bug you are fixing.
In your original engine.py, StockPreprocessor.prepare_data()
fits one single MinMaxScaler on the ENTIRE dataset, then
walk_forward_validation() splits that already-scaled data into
folds. This means the scaler — and therefore every test fold —
was computed using statistics (the min and max of MA20, RSI,
MACD, etc.) drawn from data points that occur in the FUTURE
relative to that fold's training window. That is data leakage.
It means your reported MAE, directional accuracy and Sharpe
Ratio are all somewhat optimistic, because the test fold was
never truly "unseen" — the scaler had already peeked at it.

The fix is to fit a brand new scaler inside each fold, using
only that fold's training rows, and to throw it away after the
fold finishes. This is exactly what a real out-of-sample
deployment would experience: you never know tomorrow's price
range when you train today.

---

## Step 20 — Add the EngineConfig dataclass

What you are doing:
Centralising every hardcoded number in engine.py (window size,
epochs, batch size, RSI period, MACD spans, stop-loss
percentage, RSS feeds, Ollama URL, refresh cutoffs) into one
configuration object. Previously these numbers were scattered
across a dozen functions as literal values. Now they live in
one place, and any function can be called with a different
configuration without editing engine.py itself.

Why a dataclass and not a plain dictionary?
A Python dataclass gives you named, typed fields with default
values, and lets you write config.window_size instead of
config["window_size"]. It also lets VS Code autocomplete the
field names for you, which a plain dict cannot do.

Add this import near the top of engine.py, alongside your
other imports:

```python
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
```

Now add the configuration class itself, placed after your
imports near the top of engine.py:

```python
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
```

Press Ctrl+S to save.

Note: dtime must already be imported as
`from datetime import datetime, time as dtime` from your
Milestone 3 refresh-schedule step. If you have not yet reached
that step, add that import now alongside the other imports at
the top of the file. The order does not matter as long as it
is imported once before EngineConfig is defined.

---

## Step 21 — Add a module logger

What you are doing:
Replacing the practice of silently swallowing every error with
`except Exception: pass` with a proper Python logger. A logger
records what went wrong and why, without crashing the app or
printing noisy text to a user who is not interested in
debugging output.

Why not just use print()?
print() always shows, even in production, and cannot be turned
on or off, filtered by severity, or redirected to a file.
Logging can do all three from app.py without touching engine.py
at all.

Type this below CONFIG:

```python
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
```

You will also need `import logging` at the top of the file.

NullHandler means engine.py stays silent by default — it is a
library, not a script, so it should not print anything unless
the application using it (app.py) explicitly asks to see the
logs. If you want to see log messages while testing from the
terminal, add this to your test script, not to engine.py:

```python
logging.basicConfig(level=logging.INFO)
```

Press Ctrl+S to save.

---

## Step 22 — Expand the feature set from 6 to 13 indicators

What you are doing:
Your original feature set was Close, MA20, MA50, RSI, MACD,
Signal — six columns. Your reviewer's MODERATE feedback asked
for a richer feature set. You are now adding: Returns, EMA20,
Bollinger Bands (BB_Upper, BB_Lower), Volatility, ATR, and
Volume, bringing the total to 13 features.

What each new indicator means:
Returns: the day-to-day percentage change in Close price.
  Captures momentum in a way that raw price level cannot,
  because a RM 0.10 move means something very different for a
  RM 1.00 stock than a RM 50.00 stock.
EMA20: a 20-day Exponential Moving Average. Unlike a simple
  moving average, EMA weights recent days more heavily, so it
  reacts faster to new price action.
Bollinger Bands (BB_Upper, BB_Lower): two bands drawn 2 standard
  deviations above and below a 20-day moving average. When price
  approaches BB_Upper the stock may be statistically overbought;
  near BB_Lower it may be oversold. This is a different signal
  from RSI because it is anchored to the stock's own recent
  volatility, not a fixed 0-100 scale.
Volatility: the rolling 20-day standard deviation of Returns.
  Tells the model how turbulent the stock has recently been —
  useful context for how much weight to give the other signals.
ATR (Average True Range): a 14-day measure of how much a stock
  typically moves intraday, using High, Low and the previous
  Close. Unlike Volatility, ATR captures the *range* within a
  single day, not just the close-to-close change.
Volume: the number of shares traded. A price move on high
  volume is generally considered more reliable than the same
  move on thin volume.

Why one shared function for both training and forecasting?
The biggest risk when you write your own indicator formula
twice — once for historical training data, once inside the
recursive forecaster — is that the two versions drift apart
over time as you tweak one and forget the other. To prevent
that, you now write ONE function, compute_indicators(), and
both add_features() (training) and ChampionForecaster.forecast()
(prediction) call it. They can never disagree with each other.

Open engine.py and find your existing add_features() function
from Milestone 1. Replace it entirely with these two functions:

```python
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
```

Press Ctrl+S to save.

Now update your FEATURES list. Find the line in
StockPreprocessor (or near the top of the file if you keep it
as a module constant) that currently reads:

```python
FEATURES = ["Close", "MA20", "MA50", "RSI", "MACD", "Signal"]
```

Replace it with the full 13-feature list. Close MUST stay at
index 0 — every inverse_transform() and forecasting step
depends on that exact position:

```python
FEATURES: List[str] = [
    "Close", "Returns", "MA20", "MA50", "EMA20",
    "RSI", "MACD", "Signal", "BB_Upper", "BB_Lower",
    "Volatility", "ATR", "Volume",
]
```

Press Ctrl+S to save.

Test it:

```cmd
python -c "
from engine import fetch_data, add_features, FEATURES
df = fetch_data('1155.KL')
df = add_features(df)
print('Feature columns:', list(df.columns))
print('All 13 present:', set(FEATURES).issubset(df.columns))
print('Any NaN remaining:', df[FEATURES].isna().any().any())
"
```

What you should see:
A list including all 13 feature names, `All 13 present: True`,
and `Any NaN remaining: False`.

---

## Step 23 — Rewrite StockPreprocessor for the new feature count

What you are doing:
Updating StockPreprocessor so it works against the 13-feature
list automatically, and adding two small helper methods —
get_feature_matrix() and build_sequences() — that the new
leakage-free walk_forward_validation() needs in Step 24.

What changed and why:
get_feature_matrix() simply extracts the raw, UNSCALED feature
columns as a numpy array. This separates "select the columns"
from "scale the columns," which used to be tangled together
inside prepare_data(). Splitting them apart is what makes
per-fold scaling possible in the next step.

build_sequences() takes an already-scaled array and slides the
60-day window across it. prepare_data() still exists and still
fits the scaler on the whole dataset — it is now used ONLY by
the champion forecaster and the direct multi-step forecaster,
where you genuinely are forecasting the true future and there
is no evaluation metric being computed, so no leakage concern
applies there.

Replace your existing StockPreprocessor class entirely with this:

```python
class StockPreprocessor:
    """
    Builds sliding-window sequences and manages MinMax scaling.

    Scaling responsibility: prepare_data() fits the scaler on
    ALL data and is used only by the champion / direct
    forecasters, where the target is the genuine future and no
    evaluation metric is at stake. walk_forward_validation()
    does NOT use prepare_data() — it fits a fresh scaler per
    fold on training rows only, fixing the data-leakage issue
    raised in the academic review.
    """

    FEATURES = FEATURES

    def __init__(self, window_size: int = None, config: EngineConfig = CONFIG):
        self.config      = config
        self.window_size = window_size or config.window_size
        self.scaler      = MinMaxScaler(feature_range=(0, 1))
        self.features    = list(self.FEATURES)

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts the raw, UNSCALED feature columns as a numpy array."""
        return df[self.features].values

    def build_sequences(self, scaled: np.ndarray):
        """
        Slides the 60-day window across an already-scaled array.
        Sequence j uses rows [j, j+window) as input and the Close
        value at row (j+window) as the target.
        """
        X, y = [], []
        for i in range(self.window_size, len(scaled)):
            X.append(scaled[i - self.window_size:i])
            y.append(scaled[i, 0])               # Close is index 0
        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame):
        """
        Returns (X, y, scaled_full). Fits the scaler on the FULL
        dataset. Use only for champion training / true forecasting,
        never for walk-forward evaluation.
        """
        scaled_full = self.scaler.fit_transform(self.get_feature_matrix(df))
        X, y = self.build_sequences(scaled_full)
        return X, y, scaled_full

    def inverse_transform(self, pred: np.ndarray,
                          scaler: MinMaxScaler = None) -> np.ndarray:
        """
        Inverse-scales a 1-D array of Close predictions to RM
        prices. Accepts an optional per-fold scaler — this is what
        lets walk_forward_validation() invert each fold correctly
        using THAT fold's scaler, not the preprocessor's own one.
        """
        sc = scaler if scaler is not None else self.scaler
        dummy = np.zeros((len(pred), len(self.features)))
        dummy[:, 0] = np.asarray(pred).flatten()
        return sc.inverse_transform(dummy)[:, 0]

    def get_last_window_3d(self, scaled_full: np.ndarray) -> np.ndarray:
        return scaled_full[-self.window_size:][np.newaxis, ...]

    def get_last_window_2d(self, scaled_full: np.ndarray) -> np.ndarray:
        return scaled_full[-self.window_size:].flatten()[np.newaxis, :]

    def scale_row(self, row: np.ndarray) -> np.ndarray:
        """
        Scales a single new feature row with the fitted scaler,
        clipping to [0, 1] to guard against scaler extrapolation
        when a predicted price leaves the historical min/max
        range. Documented known limitation.
        """
        return np.clip(self.scaler.transform(row), 0.0, 1.0)
```

Press Ctrl+S to save.

---

## Step 24 — Fix the data-leakage bug in walk_forward_validation()

What you are doing:
This is the MAJOR fix. The function signature changes shape:
it now accepts the feature-engineered DataFrame directly
instead of pre-scaled (X, y) arrays, because the scaler can no
longer be fit once at the very start.

INTERFACE CHANGE — read this carefully:

```
OLD:  X, y, scaled = pre.prepare_data(df)
      res = walk_forward_validation(X, y, pre, model_type="LSTM")

NEW:  res = walk_forward_validation(df, pre, model_type="LSTM")
```

Anywhere in app.py or in your own test scripts that calls
walk_forward_validation(), you must update the call to pass
the feature-engineered DataFrame (the output of add_features())
together with the preprocessor object, and drop the separate
X, y arguments. The champion-training path is NOT affected —
ChampionForecaster.fit() still takes (X, y, scaled, model_type)
exactly as before, because that path correctly fits on all data.

Why does this fix the leakage?
Inside each fold, a brand-new MinMaxScaler is created and
.fit() is called only on the raw rows that belong to that
fold's training window — never on the test rows, and never on
rows from later folds. .transform() is then applied to the
whole series, but transform() does not learn anything; it only
applies the min/max it already knows. So the test rows get
scaled using parameters that have never seen them. That is the
textbook definition of leakage-free preprocessing for time
series.

Replace your existing walk_forward_validation() function
entirely with this:

```python
def walk_forward_validation(df: pd.DataFrame,
                            preprocessor: "StockPreprocessor",
                            model_type: str = "XGBoost",
                            n_splits: int = None,
                            epochs: int = None,
                            model_params: dict = None,
                            config: EngineConfig = CONFIG) -> dict:
    """
    Expanding-window walk-forward cross-validation with NO data
    leakage.

    For each fold a fresh MinMaxScaler is fit on that fold's
    TRAINING rows only, then applied to the whole series. Test
    rows are therefore scaled with parameters they never
    influenced. All metrics are computed on inverse-transformed
    RM prices, so MAE is a genuine Ringgit error.

    Takes the feature-engineered DataFrame (unscaled) instead of
    pre-scaled (X, y), because the scaler can no longer be fit
    globally up front.
    """
    n_splits = n_splits or config.n_splits
    epochs   = epochs or config.epochs
    W        = preprocessor.window_size

    raw   = preprocessor.get_feature_matrix(df)        # unscaled (T, F)
    n_seq = len(raw) - W
    if n_seq <= n_splits:
        raise ValueError(
            f"Not enough data ({n_seq} sequences) for {n_splits} splits."
        )
    fold_size = n_seq // (n_splits + 1)

    maes, accs, sharpes = [], [], []
    all_preds_rm, all_acts_rm = [], []

    for i in range(n_splits):
        train_end_seq  = fold_size * (i + 1)
        test_start_seq = train_end_seq
        test_end_seq   = min(test_start_seq + fold_size, n_seq)

        # Fit scaler on TRAINING rows only. Training sequences
        # target rows [W, W+train_end_seq); the last input window
        # ends just before train_raw_end, so this covers exactly
        # what the training set can see — nothing from the future.
        train_raw_end = W + train_end_seq
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(raw[:train_raw_end])
        scaled_all = scaler.transform(raw)            # transform-only on test

        X_all, y_all = preprocessor.build_sequences(scaled_all)
        X_tr, y_tr   = X_all[:train_end_seq], y_all[:train_end_seq]
        X_te         = X_all[test_start_seq:test_end_seq]
        y_te         = y_all[test_start_seq:test_end_seq]
        if len(X_te) == 0:
            continue

        if model_type == "XGBoost":
            model = build_xgboost(model_params)
            model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
            preds_scaled = model.predict(X_te.reshape(len(X_te), -1))

        elif model_type in SEQUENCE_MODELS:
            mp = model_params or {}
            seq_kw = {k: mp[k] for k in
                     ("units", "dropout", "learning_rate",
                      "num_heads", "num_blocks") if k in mp}
            batch_size = mp.get("batch_size", config.batch_size)
            model = build_sequence_model(
                model_type, (X_tr.shape[1], X_tr.shape[2]), **seq_kw)
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      validation_split=config.validation_split,
                      callbacks=_build_callbacks(config), verbose=0)
            preds_scaled = model.predict(X_te, verbose=0).flatten()

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Inverse-transform with THIS fold's scaler → metrics in RM.
        true_rm  = preprocessor.inverse_transform(y_te, scaler=scaler)
        preds_rm = preprocessor.inverse_transform(preds_scaled, scaler=scaler)

        maes.append(mean_absolute_error(true_rm, preds_rm))
        accs.append(directional_accuracy(true_rm, preds_rm))
        sharpes.append(sharpe_ratio(true_rm, preds_rm))
        all_preds_rm.extend(preds_rm.tolist())
        all_acts_rm.extend(true_rm.tolist())

    return {
        "avg_mae":        float(np.mean(maes)) if maes else float("nan"),
        "avg_acc":        float(np.mean(accs)) if accs else float("nan"),
        "avg_sharpe":     float(np.mean(sharpes)) if sharpes else float("nan"),
        "predictions_rm": np.array(all_preds_rm),
        "actuals_rm":     np.array(all_acts_rm),
        "predictions":    np.array(all_preds_rm),
        "actuals":        np.array(all_acts_rm),
    }
```

Press Ctrl+S to save.

Test it — note the new call shape:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, walk_forward_validation
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor(window_size=20)
res = walk_forward_validation(df, proc, model_type='XGBoost', n_splits=3)
print('MAE (RM)      :', round(res['avg_mae'], 4))
print('Dir. Accuracy :', round(res['avg_acc']*100, 1), '%')
print('Sharpe Ratio  :', round(res['avg_sharpe'], 3))
"
```

What you should see:
Three numbers, similar in shape to before, but now you can
confidently tell your supervisor these figures contain zero
data leakage — every test fold was scaled using only
information available before that fold began.

**Important note about your MAE changing slightly:** Do not be
alarmed if your MAE goes up a little compared to your earlier,
leaky results. A higher but honest number is strictly better
for your viva than a lower but leaked one — and you can explain
exactly why if asked.

---

## Step 25 — Add hyperparameter tuning (random search + Optuna)

What you are doing:
Your reviewer's MODERATE feedback noted that all hyperparameters
(neurons, learning rate, XGBoost depth) were hand-picked rather
than systematically searched. You are now adding two tiers of
search: a simple random search that needs no extra libraries
beyond scikit-learn, and a Bayesian (Optuna/TPE) search that is
more sample-efficient when installed.

What is the difference between random search and Bayesian
(TPE) search?
Random search tries random combinations and keeps the best one
it stumbled across. Bayesian search (using the Tree-structured
Parzen Estimator, or TPE) uses the results of EARLIER trials to
decide which combination to try NEXT — it learns where the
promising region of the search space is, so it usually finds a
better result in fewer trials.

Why TimeSeriesSplit instead of normal K-Fold for tuning?
The same look-ahead-bias reasoning from Step 24 applies here.
TimeSeriesSplit only ever trains on the past and validates on
what comes after, same as your walk-forward validation, so the
hyperparameters you select are not chosen using information
the model would never have during live trading.

Install Optuna (Bayesian search) alongside your other packages:

```cmd
pip install optuna
```

Add `optuna>=3.0` to your requirements.txt.

Add the optional import near the top of engine.py with your
other optional-dependency try/except blocks:

```python
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    optuna = None
    _HAS_OPTUNA = False
```

You will also need this import for TimeSeriesSplit:

```python
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
```

Now add the random-search tuner for XGBoost. Place this after
build_xgboost() and _build_callbacks() in engine.py — note this
guide assumes you have also added the _build_callbacks() helper
from the reviewed file, which wraps EarlyStopping together with
ReduceLROnPlateau; see Step 26 for that helper if you have not
added it yet:

```python
def tune_xgboost(X: np.ndarray, y: np.ndarray,
                 n_iter: int = 20, cv_splits: int = 3,
                 random_state: int = 42):
    """
    RandomizedSearchCV over XGBoost using TimeSeriesSplit
    (respects temporal order — no shuffling).
    Returns (best_params, best_cv_mae).
    """
    X_flat = X.reshape(len(X), -1)
    param_dist = {
        "n_estimators":     [100, 200, 300, 400],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0],
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror",
                               verbosity=0, random_state=random_state),
        param_distributions=param_dist, n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=random_state, verbose=0,
    )
    search.fit(X_flat, y)
    return search.best_params_, float(-search.best_score_)
```

Now add the Bayesian (Optuna) version, which falls back to the
random search above if Optuna is not installed:

```python
def tune_xgboost_optuna(X: np.ndarray, y: np.ndarray,
                        n_trials: int = 30, cv_splits: int = 3,
                        timeout: int = None, random_state: int = 42):
    """
    Bayesian XGBoost tuning with Optuna's TPE sampler, scored by
    mean MAE over a TimeSeriesSplit. More sample-efficient than
    tune_xgboost(). Falls back to that random search if Optuna
    is not installed.
    """
    if not _HAS_OPTUNA:
        return tune_xgboost(X, y, n_iter=n_trials, cv_splits=cv_splits,
                            random_state=random_state)

    X_flat   = X.reshape(len(X), -1)
    splitter = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
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
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, float(study.best_value)
```

Finally, add the unified entry point that picks Bayesian search
automatically when Optuna is available:

```python
def optimize_hyperparameters(model_type: str, X: np.ndarray, y: np.ndarray,
                             method: str = "auto", config: EngineConfig = CONFIG,
                             **kwargs):
    """
    Unified hyperparameter optimisation entry point.
    method="auto"   -> Bayesian (Optuna) when installed, else random search.
    method="optuna" -> force Bayesian (still falls back if absent).
    method="random" -> force the lightweight random search.
    """
    use_optuna = (method == "optuna") or (method == "auto" and _HAS_OPTUNA)
    if model_type == "XGBoost":
        if use_optuna:
            return tune_xgboost_optuna(X, y, **kwargs)
        return tune_xgboost(X, y, **kwargs)
    raise ValueError(f"Unknown model_type for tuning in this guide step: {model_type!r}")
```

Press Ctrl+S to save.

Test it:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, tune_xgboost
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor(window_size=20)
X, y, scaled = proc.prepare_data(df)
best_params, best_mae = tune_xgboost(X, y, n_iter=5, cv_splits=3)
print('Best params:', best_params)
print('Best CV MAE:', round(best_mae, 4))
"
```

What you should see:
A dictionary of tuned XGBoost hyperparameters and a best MAE
score. This will take 20-60 seconds depending on your machine.

---
## Step 26 — Add the shared EarlyStopping + ReduceLROnPlateau callback builder

What you are doing:
Your original code created a fresh EarlyStopping callback
inline, in several different functions, with slightly
inconsistent settings each time. You are now centralising this
into one _build_callbacks() helper, and adding
ReduceLROnPlateau, which automatically lowers the learning rate
when validation loss plateaus, helping the model converge in
fewer epochs.

Type this below build_xgboost() in engine.py. You will need
`from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau`
at the top of the file if you do not already have it:

```python
def _build_callbacks(config: EngineConfig = CONFIG):
    """
    Shared Keras callbacks. EarlyStopping restores the best
    weights (no on-disk checkpoint needed); ReduceLROnPlateau
    speeds up convergence, lowering the effective epoch count
    per fold.
    """
    return [
        EarlyStopping(monitor="val_loss", patience=config.patience,
                      restore_best_weights=True, min_delta=1e-5, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(1, config.patience - 1),
                          min_lr=1e-5, verbose=0),
    ]
```

Press Ctrl+S to save. This function is what Steps 32, 33 and 34
above already call as `callbacks=_build_callbacks(config)`.

---

## Step 27 — Add an optional, multi-step alternative to recursive forecasting

What you are doing:
Your reviewer flagged that recursive (iterative) forecasting
accumulates error across the 5-day horizon, because each day's
prediction feeds into the next day's input. You are adding a
DIRECT multi-step alternative: instead of one model predicting
one day ahead and being rolled forward five times, you train
FIVE separate XGBoost models, one specifically for "predict 1
day ahead," one for "predict 2 days ahead," and so on up to 5.
None of their errors can compound into each other, because none
of them ever sees another model's prediction as input.

Why not always use the direct method instead?
Direct multi-step needs 5x the training time (one model per
horizon step) and, with only XGBoost wired up here, does not
extend as naturally to the neural architectures without
significant extra engineering. The right academic answer is not
"always use direct" or "always use recursive" — it is "measure
both and let the evidence decide," which is exactly what Step
37 builds next.

Type this below ChampionForecaster (which you will build in
Milestone 3 if you have not already) or anywhere convenient
later in the file:

```python
def direct_multistep_forecast_xgb(df: pd.DataFrame,
                                  preprocessor: StockPreprocessor,
                                  horizon: int = None,
                                  model_params: dict = None,
                                  config: EngineConfig = CONFIG) -> np.ndarray:
    """
    Direct multi-step alternative to recursive forecasting:
    trains a separate XGBoost model per horizon step h (each
    predicting Close exactly h steps ahead from the same 60-day
    window), so errors do NOT compound across steps.

    Uses a scaler fit on all data (this is a true-future forecast,
    not an evaluation, so no leakage concern applies, matching the
    champion's scaling approach). XGBoost-only for tractability; a
    recurrent direct multi-step variant is future work.

    Returns array of shape (horizon,) in RM prices.
    """
    horizon = horizon or config.forecast_horizon
    raw = preprocessor.get_feature_matrix(df)
    W   = preprocessor.window_size

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)
    X, _   = preprocessor.build_sequences(scaled)
    X_flat = X.reshape(len(X), -1)
    close_scaled = scaled[:, 0]
    last_window  = scaled[-W:].flatten()[np.newaxis, :]

    preds_scaled = []
    for h in range(1, horizon + 1):
        max_j = len(X) - (h - 1)
        if max_j <= 0:
            break
        X_h = X_flat[:max_j]
        # Target for sequence j at horizon h is Close at row (W + j - 1 + h)
        y_h = np.array([close_scaled[W + j - 1 + h] for j in range(max_j)])
        model = build_xgboost(model_params)
        model.fit(X_h, y_h)
        preds_scaled.append(float(model.predict(last_window)[0]))

    return preprocessor.inverse_transform(np.array(preds_scaled), scaler=scaler)
```

Press Ctrl+S to save.

Test it:

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, direct_multistep_forecast_xgb
df = fetch_data('1155.KL')
df = add_features(df)
proc = StockPreprocessor(window_size=60)
forecast = direct_multistep_forecast_xgb(df, proc, horizon=5)
print('Direct 5-day forecast:', forecast.round(4))
"
```

What you should see:
Five RM prices, one per forecast day. This will take a few
seconds longer than a single XGBoost fit, because it is
training five separate models.

---

## Step 28 — Quantitatively compare recursive vs direct forecasting

What you are doing:
Your second-round feedback specifically asked you to justify
your forecasting strategy choice "from evidence, not
assumption." This function runs a rolling-origin backtest: it
picks several historical points in time, forecasts 5 days ahead
from each one using BOTH the recursive ChampionForecaster AND
the direct multi-step method, then compares both against what
actually happened using RMSE, MAE and MAPE.

What is a rolling-origin backtest?
Instead of testing your forecast just once, at just one point
in time, you repeat the same forecast-and-compare exercise from
several different "pretend today" points spaced `horizon` days
apart, moving backwards through history. This gives you several
independent samples of how each strategy performs, rather than
one potentially lucky or unlucky data point.

Type this below direct_multistep_forecast_xgb(). You will also
need `from typing import ...` already covering List, Callable
etc. from Step 20:

```python
def _forecast_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, MAE and MAPE (%) between aligned actual/predicted price vectors."""
    actual    = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err  = predicted - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / (actual + 1e-9))) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def compare_forecasting_strategies(raw_df: pd.DataFrame,
                                   preprocessor: "StockPreprocessor",
                                   horizon: int = None, n_origins: int = 5,
                                   model_params: dict = None,
                                   config: EngineConfig = CONFIG) -> dict:
    """
    Evidence-based comparison of recursive vs direct multi-step
    forecasting via a rolling-origin (walk-forward) backtest.

    For each of the last n_origins evaluation origins, BOTH
    strategies forecast `horizon` steps using only data strictly
    before that origin, and the predictions are compared to the
    realised closes. Both use XGBoost for a controlled
    like-for-like comparison.

    raw_df : the ORIGINAL OHLCV frame (not feature-engineered) —
             features are recomputed per slice, so each origin
             sees only genuinely historical data.

    Returns dict with: horizon, n_origins_used,
    recursive / direct : {RMSE, MAE, MAPE},
    per_step : DataFrame [Step, Recursive_MAE, Direct_MAE],
    detail   : DataFrame [Origin, Step, Actual, Recursive, Direct],
    recommended : 'recursive' or 'direct' (lower aggregate RMSE).
    """
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
        raise ValueError(
            f"Not enough data for comparison: need >= {min_history + horizon} "
            f"rows, have {n}."
        )

    rec_rows, dir_rows, detail = [], [], []
    for origin in origins:
        hist_raw = raw_df.iloc[:origin].copy()
        actual = raw_df["Close"].iloc[origin:origin + horizon].to_numpy(dtype=float)
        if len(actual) < horizon:
            continue

        hist_feat = add_features(hist_raw, config)
        if len(hist_feat) <= W + 1:
            continue

        # Fresh preprocessor per origin → independent scaler, no leakage.
        proc = StockPreprocessor(window_size=W, config=config)

        direct_pred = direct_multistep_forecast_xgb(
            hist_feat, proc, horizon=horizon, model_params=model_params, config=config)

        Xc, yc, scaledc = proc.prepare_data(hist_feat)
        champ = ChampionForecaster(proc, config=config).fit(
            Xc, yc, scaledc, "XGBoost", raw_df=hist_raw, model_params=model_params)
        recursive_pred = champ.forecast(horizon)

        h = min(horizon, len(direct_pred), len(recursive_pred))
        rec_rows.append((actual[:h], np.asarray(recursive_pred[:h], dtype=float)))
        dir_rows.append((actual[:h], np.asarray(direct_pred[:h], dtype=float)))
        for step in range(h):
            detail.append({
                "Origin":    origin, "Step": step + 1,
                "Actual":    round(float(actual[step]), 4),
                "Recursive": round(float(recursive_pred[step]), 4),
                "Direct":    round(float(direct_pred[step]), 4),
            })

    if not rec_rows:
        raise ValueError("No valid origins produced forecasts for comparison.")

    rec_actual = np.concatenate([a for a, _ in rec_rows])
    rec_pred   = np.concatenate([p for _, p in rec_rows])
    dir_actual = np.concatenate([a for a, _ in dir_rows])
    dir_pred   = np.concatenate([p for _, p in dir_rows])

    rec_metrics = _forecast_error_metrics(rec_actual, rec_pred)
    dir_metrics = _forecast_error_metrics(dir_actual, dir_pred)

    detail_df = pd.DataFrame(detail)
    per_step = (detail_df
                .assign(Recursive_AE=lambda d: (d["Recursive"] - d["Actual"]).abs(),
                        Direct_AE=lambda d: (d["Direct"] - d["Actual"]).abs())
                .groupby("Step")[["Recursive_AE", "Direct_AE"]]
                .mean().round(4)
                .rename(columns={"Recursive_AE": "Recursive_MAE",
                                 "Direct_AE": "Direct_MAE"})
                .reset_index())

    recommended = "direct" if dir_metrics["RMSE"] <= rec_metrics["RMSE"] else "recursive"

    return {
        "horizon":        horizon,
        "n_origins_used": len(rec_rows),
        "recursive":      rec_metrics,
        "direct":         dir_metrics,
        "per_step":       per_step,
        "detail":         detail_df,
        "recommended":    recommended,
    }
```

Note: this function calls ChampionForecaster, so it must be
placed AFTER that class in engine.py — if you have not yet
built ChampionForecaster (Milestone 3, Step 33 below), come
back to test this step once that class exists.

Press Ctrl+S to save.

Test it (after ChampionForecaster exists):

```cmd
python -c "
from engine import fetch_data, add_features, StockPreprocessor, compare_forecasting_strategies
df = fetch_data('1155.KL')
proc = StockPreprocessor(window_size=20)
result = compare_forecasting_strategies(df, proc, horizon=3, n_origins=3)
print('Recommended strategy:', result['recommended'])
print('Recursive RMSE/MAE/MAPE:', result['recursive'])
print('Direct    RMSE/MAE/MAPE:', result['direct'])
print(result['per_step'].to_string(index=False))
"
```

What you should see:
A printed recommendation ("recursive" or "direct"), two small
dictionaries of error metrics, and a per-step MAE breakdown
table. Whatever the result says, that is your evidence-based
answer for the viva question "why did you choose recursive (or
direct) forecasting?"

---

## Step 29 — Improve sentiment headline matching with aliases and fuzzy logic

What you are doing:
Your MODERATE feedback noted that matching headlines purely by
checking `company_name.lower() in title.lower()` misses a lot of
real coverage — a headline mentioning "1155" (Maybank's ticker),
"MBB" (a common abbreviation), or "Maybnk" (a typo) would never
match "Maybank" under the old exact-substring rule. You are
adding an alias dictionary plus fuzzy string matching so these
near-misses are correctly captured.

What is fuzzy matching, in plain terms?
SequenceMatcher (from Python's built-in difflib) compares two
strings and returns a similarity ratio between 0 and 1, based on
how many characters they have in common in the same relative
order. "Maybnk" vs "maybank" scores highly even though one
letter is missing, because almost every other character lines
up. Setting a threshold (0.82 here) means only genuinely close
matches pass, filtering out unrelated words that happen to share
a few letters.

Add the alias dictionary near your other module-level constants
(after FEATURES is a good spot). This version covers all 10
stocks in this project's universe (Maybank, CIMB, Public Bank,
TNB, Petronas Chemicals, Maxis, Axiata, Genting, KL Kepong,
IHH Healthcare) — note the dictionary keys must exactly match
the company names used as keys in BURSA_10 in app.py, since
those names are passed straight through to this dictionary
when matching headlines:

```python
COMPANY_ALIASES: Dict[str, List[str]] = {
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
```

If you ever swap one of these 10 stocks for a different Bursa
counter, add a matching entry here too — every company should
have its formal name, common abbreviation, and ticker number
listed. If a company has no entry in this dictionary at all,
matching does not break; _aliases_for() simply falls back to
matching the literal name you passed as a plain substring. What
you lose without an entry is the fuzzy/abbreviation/ticker
coverage — a headline saying just "6012" or a typo like
"Maxxis" would not be caught the way "1155" or "Maybnk" is
caught for Maybank.

Add `from difflib import SequenceMatcher` to your imports.

Now replace your existing fetch_company_headlines() entirely
with this alias-and-fuzzy-aware version:

```python
def _aliases_for(company_name: str, aliases: dict = None) -> List[str]:
    """Resolves a company name to the set of strings worth matching against."""
    aliases = aliases if aliases is not None else COMPANY_ALIASES
    name_l = company_name.lower()
    terms = {name_l}
    for canon, al in aliases.items():
        al_l = [a.lower() for a in al]
        if name_l == canon.lower() or name_l in al_l:
            terms.update(al_l)
            terms.add(canon.lower())
    return list(terms)


def _fuzzy_contains(title_lower: str, term: str, threshold: float) -> bool:
    """True if `term` appears in the title as a substring or a fuzzy token."""
    if term in title_lower:
        return True
    for token in re.findall(r"[a-z0-9.]+", title_lower):
        if SequenceMatcher(None, token, term).ratio() >= threshold:
            return True
    return False


def _headline_matches(title: str, company_name: str,
                      threshold: float, aliases: dict = None) -> bool:
    title_lower = title.lower()
    return any(_fuzzy_contains(title_lower, term, threshold)
               for term in _aliases_for(company_name, aliases))


def fetch_company_headlines(company_name: str, max_headlines: int = None,
                            config: EngineConfig = CONFIG,
                            aliases: dict = None) -> list:
    """
    Track A — Company News.

    Pulls recent headlines from Malaysian financial RSS feeds and
    keeps those matching the company via alias + fuzzy matching,
    so tickers, abbreviations and minor spelling variants are
    caught. Accepts a multilingual English/Bahasa Melayu mix
    without translation — the language model reads both.
    """
    max_headlines = max_headlines or config.max_headlines
    matched = []

    for url in config.rss_feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                title = entry.get("title", "").strip()
                if title and _headline_matches(
                        title, company_name, config.fuzzy_threshold, aliases):
                    matched.append(title)
        except Exception:
            continue

    return matched[:max_headlines]
```

Press Ctrl+S to save.

Test the matching logic directly, without needing internet.
This expanded version checks one example per stock across all
10 companies, plus two unrelated headlines that must correctly
fail to match:

```cmd
python -c "
from engine import _headline_matches

tests = [
    ('Maybank posts record profit', 'Maybank', True),
    ('1155 climbs after earnings', 'Maybank', True),
    ('Maybnk earnings beat estimates', 'Maybank', True),
    ('CIMB Group reports higher profit', 'CIMB', True),
    ('Public Bank Q3 earnings beat forecast', 'Public Bank', True),
    ('Tenaga Nasional unveils new solar plant', 'TNB', True),
    ('Petronas Chemicals reports strong Q2', 'Petronas Chemicals', True),
    ('Maxis Berhad rolls out 5G expansion', 'Maxis', True),
    ('Axiata Group announces restructuring', 'Axiata', True),
    ('Genting Bhd announces new resort', 'Genting', True),
    ('KLK shares rise on CPO prices', 'KL Kepong', True),
    ('IHH Healthcare expands hospital network', 'IHH Healthcare', True),
    ('Durian prices surge in Penang', 'Maybank', False),
    ('Unrelated tech news today', 'Genting', False),
]

for headline, company, expected in tests:
    result = _headline_matches(headline, company, 0.82)
    status = 'PASS' if result == expected else 'FAIL'
    print(f'{status}  [{company:20s}] {headline}')
"
```

What you should see:
Fourteen lines, every one starting with PASS. The first twelve
correctly match each company's coverage (formal name, ticker,
abbreviation, or a deliberate typo for Maybank), and the final
two unrelated headlines correctly fail to match. If any line
shows FAIL, check that the company name you tested matches a
key in COMPANY_ALIASES exactly, and that the alias list for
that company actually contains the term in the headline.

---

## Step 30 — Add a pluggable, live macro-data provider

What you are doing:
Your original mock_macro_indicators() always returned the same
three hardcoded numbers, with no way to ever swap in real data.
Your MODERATE feedback asked for a path to live macroeconomic
figures. You are now making the macro source pluggable via
config.macro_provider, and adding a real, working connector to
Bank Negara Malaysia's official public Overnight Policy Rate API
as one ready-to-use option.

Why keep the static fallback at all, even after adding a live
source?
Because live government APIs occasionally go down, change their
response format, or rate-limit you. A capstone DSS that crashes
the moment a third-party API has a bad day is a fragile system.
Falling back to documented, clearly-labelled static values keeps
the dashboard always working, while logging exactly when and why
the fallback was used.

Add `import urllib.request` and `import json` to your imports if
not already present (json was already needed for the sentiment
agent's JSON parsing).

Replace mock_macro_indicators() with this set of functions. Keep
the name mock_macro_indicators() too, as a backward-compatible
alias, so anything still calling it by the old name keeps
working:

```python
def _static_macro_indicators() -> dict:
    """Documented placeholder values for Malaysia's macro context."""
    return {
        "OPR":        3.00,   # Overnight Policy Rate, %
        "CPI":        2.10,   # Inflation rate, %
        "GDP_Growth": 4.40,   # GDP growth rate, %
    }


def mock_macro_indicators() -> dict:
    """Backward-compatible alias retained for older callers."""
    return _static_macro_indicators()


def get_macro_indicators(config: EngineConfig = CONFIG) -> dict:
    """
    Track B — Macroeconomic Indicators.

    If config.macro_provider is a callable returning
    {OPR, CPI, GDP_Growth}, its live values are used; otherwise
    the documented static placeholders are returned. Any provider
    error falls back cleanly, so the pipeline never breaks.
    """
    provider = config.macro_provider
    if callable(provider):
        try:
            data = provider()
            if all(k in data for k in ("OPR", "CPI", "GDP_Growth")):
                return {k: float(data[k]) for k in ("OPR", "CPI", "GDP_Growth")}
        except Exception:
            pass
    return _static_macro_indicators()


def _http_get_json(url: str, headers: dict = None, timeout: int = 10) -> dict:
    """Minimal stdlib JSON GET — no third-party 'requests' dependency."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bnm_opr(timeout: int = 10) -> float:
    """
    Live Overnight Policy Rate (%) from Bank Negara Malaysia's
    official Open API. Raises on network/parse failure so callers
    can decide how to fall back.
    """
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


def bnm_macro_provider(timeout: int = 10,
                       cpi_fn=None, gdp_fn=None) -> dict:
    """
    Ready-to-plug live macro provider for Malaysia.

    Wire it in with:  CONFIG.macro_provider = bnm_macro_provider

    Pulls OPR live from BNM's official Open API. CPI and GDP
    growth are served by DOSM rather than BNM, so pass optional
    cpi_fn / gdp_fn callables to source those live too — any field
    whose live fetch fails degrades to the documented static
    value. The result always contains all three keys.
    """
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
```

Press Ctrl+S to save.

Now update build_agent_context() (which you already built in
Milestone 2, Step 17) to call get_macro_indicators() instead of
the old mock_macro_indicators():

```python
def build_agent_context(company_name: str,
                        config: EngineConfig = CONFIG) -> dict:
    """
    Bundles Track A (headlines) and Track B (macro) into one
    context dict. Always returns a valid dict, even with no
    headlines.
    """
    return {
        "headlines": fetch_company_headlines(company_name, config=config),
        "macro":     get_macro_indicators(config),
    }
```

Press Ctrl+S to save.

Test the static path (works without internet):

```cmd
python -c "
from engine import get_macro_indicators
print(get_macro_indicators())
"
```

Test the live BNM connector (needs internet):

```cmd
python -c "
from engine import fetch_bnm_opr, bnm_macro_provider
print('Live OPR:', fetch_bnm_opr())
print('Provider output:', bnm_macro_provider())
"
```

What you should see:
The static test prints the three placeholder values. The live
test prints the actual current OPR pulled from Bank Negara
Malaysia's website, then a provider dict where OPR is now the
live number but CPI and GDP_Growth still show the static
fallback (since no cpi_fn / gdp_fn was supplied). If you want to
make the whole app use live OPR going forward, set
`CONFIG.macro_provider = bnm_macro_provider` once, near the top
of app.py.

---

## Step 31 — Add specific exception handling and logging throughout

What you are doing:
Your MINOR feedback flagged the pattern `except Exception: pass`
appearing repeatedly — it silently swallows ANY error, including
genuine bugs, making problems invisible until a user reports
something mysteriously not working. You are replacing blanket
catches with specific exception types plus a logged message,
everywhere it is practical to do so.

Go through fetch_company_headlines(), get_macro_indicators(),
bnm_macro_provider(), and run_sentiment_agent() and change the
generic catches to look like this pattern. For example, inside
fetch_company_headlines(), change:

```python
        except Exception:
            continue
```

to:

```python
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Could not reach feed %s: %s", url, exc)
            continue
        except Exception as exc:
            logger.exception("Unexpected error parsing feed %s: %s", url, exc)
            continue
```

The same pattern applies inside run_sentiment_agent() around the
Ollama call — split it into three clearly separate try blocks:
one around build_agent_context(), one around the llm.invoke()
call, and one around _extract_json(), each logging a specific
message so you can tell from your logs exactly which stage
failed, rather than only knowing "something, somewhere, failed."

This step has no single test command — instead, deliberately
break something (turn off your internet, or stop Ollama) and
confirm a clear, specific log line appears explaining what
happened, rather than a silent fallback with no trace at all.

---

## Step 32 — Write automated unit tests (test_engine.py)

What you are doing:
Creating a brand new file, test_engine.py, that automatically
checks dozens of small facts about your engine.py code every
time you run it — far more thorough and far faster than manually
typing test commands in the terminal each time, and something
your supervisor can point to as evidence of rigorous testing
practice.

Why does engine.py need optional, "lazy" imports for this to
work?
If engine.py hard-crashed the moment it tried to `import
tensorflow` on a machine that does not have TensorFlow
installed, you could never run ANY test, even ones that have
nothing to do with TensorFlow (like testing your feature
engineering math). The try/except import pattern you used for
Ollama, feedparser, shap and optuna throughout this milestone
applies here too — wrap TensorFlow and XGBoost imports the same
way, so the pure-Python parts of engine.py (feature math,
config, alias matching, risk calculations) stay testable even on
a bare machine with only numpy, pandas and scikit-learn.

Create a new file called test_engine.py in the same folder as
engine.py.

Install pytest, the standard Python testing framework:

```cmd
pip install pytest
```

Add `pytest>=7.0` to requirements.txt.

The full content of test_engine.py is long — it covers feature
engineering, preprocessing, metrics, the sentiment alias/fuzzy
matching, the macro provider fallback behaviour, risk and
ranking helpers, the model registry, and (where the relevant
optional library is installed) walk-forward validation, direct
multi-step forecasting, the strategy comparison, SHAP, and both
tuning paths. Rather than retype all of it here, copy the
test_engine.py file directly from your reviewer-provided
reference into your project folder, in the same folder as
engine.py.

Press Ctrl+S to save.

Run the test suite two ways:

```cmd
pytest test_engine.py -v
```

or, if pytest itself is not installed for some reason:

```cmd
python test_engine.py
```

What you should see:
A list of test names each marked PASSED (or, for tests needing
an optional library you have not installed, SKIPPED — never
FAILED). At the bottom, a summary line like
"38 passed, 2 skipped in 4.2s". If anything shows FAILED, the
test name tells you exactly which behaviour broke — fix that
specific function and re-run before moving on.

---

## Milestone 2.5 Git Commit

```cmd
git add .
git commit -m "Milestone 2.5 - Review hardening: fixed data leakage, expanded features to 13, added hyperparameter tuning, 5 model architectures, direct multi-step forecasting, fuzzy sentiment matching, live macro provider, and automated unit tests"
```

What to show your supervisor:
Run the walk-forward test from Step 24 and explicitly say out
loud: "this scaler is now fit fresh inside each fold, using only
that fold's training rows, which removes the data leakage you
flagged."
Run the 13-feature test from Step 22 and name each of the seven
new indicators and what it captures.
Run the five-architecture test from Step 26 and show all five
shapes printing correctly.
Run the strategy-comparison test from Step 28 and read out the
recommended strategy along with its RMSE/MAE numbers — this is
your evidence-based justification, not an assumption.
Run `pytest test_engine.py -v` live and let it scroll past with
everything PASSED or SKIPPED.
If you have internet in the room, run the live BNM OPR test
from Step 30 and show a real number coming back from Malaysia's
central bank.

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

## Step 33 — Write ChampionForecaster class

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

Type this below run_sentiment_agent() and _default_neutral_response()
(the last functions you added in Milestone 2):

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

## Step 34 — Write stop_loss_advisory()

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

## Step 35 — Write build_ranking_table()

What you are doing:
Writing a function that takes results from all 10 stocks
and produces a single table sorted by expected 5-day return.
This powers the Portfolio Ranking tab in the dashboard.

The Sentiment column in this table is now populated from
run_sentiment_agent()'s sentiment_label, not from the old
VADER/TextBlob ensemble label. No change is needed to this
function itself — it already accepts sentiment_label as a
plain string, so it works unchanged with the new agent.

Type this below stop_loss_advisory():

```python
def build_ranking_table(stock_summaries: dict) -> pd.DataFrame:
    """
    Builds a ranked portfolio leaderboard sorted by expected
    5-day return. Section 3.3 of the methodology.

    stock_summaries: dict where each key is a stock name and
    each value contains: last_close, forecast, champion,
    mae, da, sharpe, sentiment_label

    sentiment_label here comes from run_sentiment_agent(),
    e.g. stock_summaries['Maybank']['sentiment_label'] =
    result['sentiment_label']
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

## Step 36 — Add the half-day / end-of-day refresh schedule

What you are doing:
Adding a small scheduling helper that controls how often the
dashboard re-runs the full pipeline. Your updated scope asks
for the dashboard to behave like a football match scoreboard
that updates at fixed checkpoints rather than continuously —
specifically, at half-day and end-of-day, instead of on every
single page reload.

Why this matters:
Re-running the LSTM, GRU, XGBoost tournament and the Llama 3
sentiment agent for all 10 stocks is expensive — it can take
several minutes. Refreshing on every click would make the
dashboard unusably slow and would also call your local
Ollama model far more often than necessary. A fixed
half-day/end-of-day schedule mirrors how a real trading
desk only re-checks its full analysis a few times per day,
not continuously.

What "half-day" and "end-of-day" mean here:
Half-day refresh: triggers once after 1:00 PM (after the
Bursa Malaysia midday session closes).
End-of-day refresh: triggers once after 5:00 PM (after the
Bursa Malaysia market closes for the day).
Between those two checkpoints, the dashboard keeps showing
the last computed result instead of recalculating, exactly
like a scoreboard that only updates at half-time and full-time.

Type this near the top of engine.py, below your CONFIG and
FEATURES constants. This uses Python's built-in datetime module,
so add that import alongside your other imports at the top
of the file first:

```python
from datetime import datetime, time as dtime
```

Now add the scheduling helper:

```python
# ═══════════════════════════════════════════════════════════════════
# REFRESH SCHEDULE — Half-Day / End-of-Day Checkpoints
# ═══════════════════════════════════════════════════════════════════

# The two daily checkpoints, similar to half-time and full-time
# in a football match. The dashboard only recalculates after
# crossing into a new checkpoint window.
HALF_DAY_CUTOFF   = dtime(13, 0)   # 1:00 PM
END_OF_DAY_CUTOFF = dtime(17, 0)   # 5:00 PM


def get_current_session_label(now: datetime = None) -> str:
    """
    Returns which refresh checkpoint the current time falls
    into: 'morning', 'half_day' or 'end_of_day'.

    This label is used as part of the Streamlit cache key in
    app.py, so cached results automatically stay valid until
    the next checkpoint is crossed, then automatically
    recompute exactly once when crossed — without needing any
    manual refresh button or background scheduler process.

    now : optional datetime override, used for testing.
          Defaults to the current local time.
    """
    if now is None:
        now = datetime.now()

    current_time = now.time()

    if current_time < HALF_DAY_CUTOFF:
        return "morning"
    elif current_time < END_OF_DAY_CUTOFF:
        return "half_day"
    else:
        return "end_of_day"
```

Press Ctrl+S to save.

Test it:

```cmd
python -c "
from engine import get_current_session_label
from datetime import datetime
print('Right now       :', get_current_session_label())
print('10:00 AM example :', get_current_session_label(datetime(2026,7,1,10,0)))
print('2:00 PM example  :', get_current_session_label(datetime(2026,7,1,14,0)))
print('6:00 PM example  :', get_current_session_label(datetime(2026,7,1,18,0)))
"
```

What you should see:
Four lines. The 10 AM example should print "morning", the
2 PM example should print "half_day", and the 6 PM example
should print "end_of_day". Your "right now" result depends
on what time you are running the test.

How this plugs into app.py:
In Step 37 below, your Streamlit caching decorators will
include get_current_session_label() as part of what they
cache on. This means Streamlit automatically treats each
new checkpoint as a cache-miss and recomputes once, then
serves the cached scoreboard-style result to every user
until the next checkpoint — exactly the behaviour described
in your updated scope, with no extra scheduler library
required.

---

## Step 37 — Create app.py

In VS Code, right-click in the Explorer panel.
Click New File. Type app.py. Press Enter.

app.py is the dashboard. It imports everything from
engine.py and displays it in a web browser using Streamlit.
It contains no machine learning logic — all of that is
in engine.py. This is also the file where BURSA_10 finally
gets defined for real (you noted it down back in Milestone 1,
Step 4, but did not type it anywhere yet). Add it near the
top of app.py, right after your imports:

```python
# The 10 highly liquid Bursa Malaysia stocks. This dictionary
# lives here in app.py, not in engine.py — engine.py only ever
# processes one stock at a time, whichever ticker and company
# name this loop hands it.
BURSA_10 = {
    "Maybank":            "1155.KL",
    "CIMB":               "1023.KL",
    "Public Bank":        "1295.KL",
    "TNB":                "5347.KL",
    "Petronas Chemicals": "5183.KL",
    "Maxis":              "6012.KL",
    "Axiata":             "6888.KL",
    "Genting":            "3182.KL",
    "KL Kepong":          "2445.KL",
    "IHH Healthcare":     "5225.KL",
}
```

These exact key names ("Maybank", "CIMB", and so on) are what
get passed as company_name into run_sentiment_agent() for
every stock, which is in turn what gets matched against the
COMPANY_ALIASES dictionary you built in engine.py back in
Step 29 — so if you ever rename a key here, add a matching
entry to COMPANY_ALIASES too, or that stock's sentiment
matching quality will quietly degrade to plain substring
matching only.

Copy the full app.py code from your reference file into
this blank file. Two things will look different from the
version you may have seen before Milestone 2's sentiment
upgrade:

1. Wherever the dashboard previously called
   benchmark_sentiment(symbol), it now calls
   run_sentiment_agent(ticker, company_name) instead, and
   reads result['sentiment_label'], result['macro_impact']
   and result['qualitative_rationale'] rather than the old
   VADER/TextBlob ensemble fields.

2. The functions that load data and run the engine are now
   wrapped so that get_current_session_label() from Step 36
   is included in the cache key, so the dashboard only
   recomputes at the half-day and end-of-day checkpoints:

```python
@st.cache_data(show_spinner="Refreshing market data...")
def load_stock_data(ticker: str, session_label: str):
    """
    session_label comes from get_current_session_label().
    Streamlit treats a new session_label as a fresh cache
    key, so this function only re-executes once per
    checkpoint (morning / half_day / end_of_day) instead of
    on every page interaction.
    """
    df = fetch_data(ticker)
    df = add_features(df)
    return df
```

Every place in app.py that previously called
load_stock_data(ticker) now calls
load_stock_data(ticker, get_current_session_label()) instead,
and the same pattern applies to whichever function runs the
model tournament and the sentiment agent. Press Ctrl+S to
save once you have copied in the updated reference file.

---

## Step 38 — Run the full app

In the VS Code terminal:

```cmd
streamlit run app.py
```

Your browser opens at http://localhost:8501

The first load of the day will take several minutes per
stock while the models train and the Llama 3 agent runs for
the first time. This is normal. After that first
computation, the dashboard behaves like a scoreboard — it
keeps showing the same result instantly for every visit
until the next half-day or end-of-day checkpoint is crossed,
at which point it recomputes once automatically.

To stop the app press Ctrl+C in the terminal.

Verify every panel works:
- Top row: price, champion, directional accuracy, Sharpe
- Leaderboard: all 3 models with metrics
- Backtest chart: actual vs all 3 model predictions
- 5-Day Forecast: 5 day cards with RM prices and % change
- Stop-loss advisory updating with the sidebar slider
- Sentiment panel showing sentiment_label, macro_impact and
  qualitative_rationale from run_sentiment_agent() — not the
  old VADER/TextBlob bars
- Portfolio Ranking tab with all 10 stocks
- A small caption somewhere on the page showing the current
  session label (morning / half_day / end_of_day) so your
  supervisor can see which checkpoint the displayed data
  belongs to

---

## Milestone 3 Git Commit

```cmd
git add .
git commit -m "Milestone 3 - UI integration with local AI sentiment agent and half-day/end-of-day refresh complete"
```

What to show your supervisor:
Live demo in the browser.
Select two different stocks and show results changing.
Move the stop-loss slider and show the advisory updating.
Open the sentiment panel and show the sentiment_label,
macro_impact and qualitative_rationale text for a stock
that has matching headlines that day.
Point out the session label caption and explain that the
dashboard only recomputes at the half-day and end-of-day
checkpoints, like a scoreboard that updates at half-time
and full-time rather than continuously.
Switch to the Portfolio Ranking tab.

---
---

# MILESTONE 4 — JULY 19
# Validation + Report Drafting

What you are doing:
Recording verified results for all 10 stocks and writing
the evaluation section of your capstone report.

---

## Step 39 — Record results for all 10 stocks

Open the app and select each stock one by one from the
sidebar dropdown. For each stock, copy the numbers from
the Model Leaderboard. The champion row has a trophy symbol.
Also note the sentiment_label and macro_impact text shown
by run_sentiment_agent() for each stock, and which session
checkpoint (morning / half_day / end_of_day) the result
belongs to.

| Stock | Champion | MAE (RM) | Dir. Acc % | Sharpe | Sentiment Label |
|-------|----------|----------|------------|--------|------------------|
| Maybank | | | | | |
| CIMB | | | | | |
| Public Bank | | | | | |
| TNB | | | | | |
| Petronas Chemicals | | | | | |
| Maxis | | | | | |
| Axiata | | | | | |
| Genting | | | | | |
| KL Kepong | | | | | |
| IHH Healthcare | | | | | |

Take a screenshot of the Portfolio Ranking tab.
This screenshot goes in your report appendix.

---

## Step 40 — Answer these for your evaluation chapter

1. Which model wins most often across the 10 stocks?
   Does deep learning outperform XGBoost or not?

2. Is directional accuracy consistently above 50%?
   Above 50% means better than random guessing.

3. Is the Sharpe Ratio positive for most stocks?
   Positive means the strategy earns more than it risks.

4. Does the sentiment_label from run_sentiment_agent() align
   with the forecast direction? Note any stocks where the
   model is bullish but the agent's sentiment is Negative,
   or vice versa.

5. For stocks where headlines were found, does the
   macro_impact text returned by the agent make sense given
   that day's OPR, CPI and GDP_Growth mock values? Spot-check
   two or three examples manually.

6. Which stock has the lowest MAE?

7. Which stock has the highest expected 5-day return?

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
| engine.py | All data, model, local AI sentiment agent and refresh scheduling code |
| app.py | Streamlit dashboard |
| requirements.txt | All Python dependencies, including feedparser and langchain |
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
- Ollama is installed and `ollama list` shows llama3 locally
- run_sentiment_agent() returns the documented default
  Neutral response when a stock has no matching headlines
- The dashboard shows the current session label and only
  recomputes at the half-day and end-of-day checkpoints
- Git log shows all 5 milestone commits with dates
- Report references function names from engine.py correctly,
  including run_sentiment_agent(), fetch_company_headlines(),
  mock_macro_indicators() and get_current_session_label()

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
| run_sentiment_agent() hangs or times out | Ollama not running | Run `ollama serve` in a separate terminal, then retry |
| run_sentiment_agent() always returns Neutral | No headlines matched, or Ollama not reachable | Check `ollama list` shows llama3; try a different company name; this can also be a correct result if no news exists that day |
| ConnectionError to localhost:11434 | Ollama service not started | Open Ollama from the Start Menu, or run `ollama serve` |
| Dashboard never refreshes after editing engine.py | Streamlit cache still holding old session_label result | Stop the app with Ctrl+C and run streamlit run app.py again |

