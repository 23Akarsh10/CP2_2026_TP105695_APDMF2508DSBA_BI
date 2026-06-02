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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
    
if __name__ == "__main__":
    # Dual-format mapping to handle Yahoo API routing instability
    TICKER_LOOKUP = {
        "Maybank":            ["1155.KL", "MAYBANK.KL"],
        "Public Bank":        ["1295.KL", "PBBANK.KL"],
        "CIMB Group":         ["1023.KL", "CIMB.KL"],
        "TNB":                ["5347.KL", "TENAGA.KL"],
        "Petronas Chemicals": ["5183.KL", "PCHEM.KL"],
        "IHH Healthcare":     ["5225.KL", "IHH.KL"],
        "Maxis Berhad":       ["6012.KL", "MAXIS.KL"],
        "Axiata Group":       ["6888.KL", "AXIATA.KL"],
        "Genting Berhad":     ["3182.KL", "GENTING.KL"],
        "KL Kepong":          ["2445.KL", "KLK.KL"],
    }

    for stock_name, tickers in TICKER_LOOKUP.items():
        print(f"Exporting {stock_name}...")
        success = False
        
        for ticker in tickers:
            try:
                # Attempt to download using the current ticker format
                df_raw = fetch_data(ticker, period="5y")
                df_features = add_features(df_raw)
                
                filename = f"{stock_name.lower().replace(' ', '_')}_5y.csv"
                df_features.to_csv(filename)
                print(f" -> Saved successfully using symbol [{ticker}] as: {filename}")
                success = True
                break  # Exit the ticker fallback loop if successful
                
            except Exception:
                continue  # Try the alternative ticker format
                
        if not success:
            print(f" -> Failed to fetch {stock_name} using all known Yahoo symbols.")


