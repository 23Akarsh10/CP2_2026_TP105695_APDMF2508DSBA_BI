# Stock Market Decision Support System
## Bursa Malaysia — Quantitative Forecasting and Market Intelligence

A modular Python Decision Support System (DSS) for Bursa Malaysia stock
prediction, built using the **CRISP-DM** methodology.

Evaluates three machine learning models — **LSTM**, **GRU**, and **XGBoost** —
across 10 highly liquid Bursa Malaysia stocks using leakage-free walk-forward
validation, overlays a locally-run **AI sentiment agent** (live news + macro
indicators via Ollama and Llama 3), and surfaces everything through an
interactive Streamlit dashboard with loss-prevention and suggested-action
signals.

---

## Project Timeline

| Milestone | Deadline | Description |
|-----------|----------|-------------|
| M1 | 31 May | Data ingestion pipeline + feature engineering complete |
| M2 | 28 June | Model tournament + sentiment engine complete |
| M3 | 12 July | UI + back-end integration + live testing complete |
| M4 | 19 July | Walk-forward validation verified + report draft complete |
| M5 | 26 July | Final submission — source code + capstone report |

---

## CRISP-DM Phase Mapping

| Phase | Module | What it does |
|-------|--------|--------------|
| Business Understanding | `README.md`, `app.py` | Defines the DSS objective and stock universe |
| Data Understanding | `engine_final.py → fetch_data()` | Downloads 5 years of OHLCV data from Yahoo Finance, plus KLSE index and MYR/USD FX rates for auxiliary features |
| Data Preparation | `engine_final.py → compute_indicators(), add_features(), StockPreprocessor` | Computes 17 features, applies RobustScaler per fold, builds 60-day sliding windows |
| Modelling | `run_tournament.py → run_model_tournament()` | Tunes hyperparameters (Optuna) and runs leakage-free walk-forward validation for XGBoost, LSTM, GRU; champion is selected via `engine_final.py → select_champion()`, a weighted composite of MAE, Directional Accuracy, and Sharpe Ratio |
| Evaluation | `engine_final.py → directional_accuracy(), sharpe_ratio()` | MAE, directional accuracy, Sharpe Ratio |
| Deployment | `app.py` | Streamlit dashboard using only the pre-selected champion model |

---

## File Structure

```
stock_predictor/
├── app.py                    # Streamlit dashboard (UI layer)
├── engine_final.py           # ML backend — data, features, models,
│                             #   sentiment agent, forecasting
├── run_tournament.py         # Manual model comparison script (run once offline)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── GUIDE.md                  # Step-by-step setup and milestone guide
├── tournament_reports/       # JSON reports from run_tournament.py (auto-created)
└── engine_cache/             # Forecast cache (auto-created, safe to delete)
```

---

## Two-Stage Workflow

```
STAGE 1 — Offline (run once, manually)
────────────────────────────────────────
  python run_tournament.py

  For each of the 10 stocks × 3 model architectures:
    1. Tune hyperparameters (Optuna TPE, or RandomizedSearchCV fallback)
    2. Walk-forward validation using tuned params (5 folds, leakage-free)
    3. Report MAE, Directional Accuracy, Sharpe Ratio

  Aggregated results saved to tournament_reports/*.json
  Champion architecture is selected on a weighted composite of average
  MAE, average Directional Accuracy, and average Sharpe Ratio across
  all stocks (select_champion() in
  engine_final.py), not on MAE alone.
  Ready-to-paste CHAMPION_MODEL and CHAMPION_PARAMS printed to terminal

  You look at the report, confirm the champion, and paste the values
  into the CHAMPION MODEL CONFIGURATION block at the top of app.py.


STAGE 2 — Live dashboard (run anytime after stage 1)
─────────────────────────────────────────────────────
  streamlit run app.py

  For each stock, using ONLY the pre-selected champion model:
    1. Download latest data + compute 17 features
    2. Walk-forward validation (champion model + tuned params only)
    3. Retrain champion on full dataset → 5-day forecast
    4. Run local AI sentiment agent (Ollama + Llama 3)
    5. Compute stop-loss advisory + suggested action
    6. Display results in the dashboard

  No tournament, no tuning, no model comparison runs inside the app.
  Results are cached per session checkpoint (morning / half-day / end-of-day).
```

---

## How to Run

```bash
# 1. Create and activate environment
conda create -n stock_predictor python=3.10
conda activate stock_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama and pull Llama 3 (for sentiment analysis)
#    Download from https://ollama.com, then:
ollama pull llama3

# 4. Run the model tournament (takes 20–60 minutes for all 10 stocks)
python run_tournament.py

#    Read the printed report, choose the champion model and params,
#    then update CHAMPION_MODEL and CHAMPION_PARAMS in app.py.

# 5. Launch the dashboard
streamlit run app.py
```

If Ollama is not installed or not running, the sentiment agent falls back
to a documented "Neutral — no data available" response. All other
dashboard features (forecasts, model metrics, stop-loss) are unaffected.

---

## Stock Universe

| Company | Ticker |
|---------|--------|
| Maybank | 1155.KL |
| CIMB | 1023.KL |
| Public Bank | 1295.KL |
| TNB | 5347.KL |
| Petronas Chemicals | 5183.KL |
| Maxis | 6012.KL |
| Axiata | 6888.KL |
| Genting | 3182.KL |
| KL Kepong | 2445.KL |
| IHH Healthcare | 5225.KL |

`BURSA_10` is defined once in `engine_final.py` and imported by both
`app.py` and `run_tournament.py` — it is never duplicated.

---

## Features (17 total)

| Feature | Description |
|---------|-------------|
| Close | Closing price (RM) |
| Returns | Daily percentage change |
| MA20 | 20-day Simple Moving Average |
| MA50 | 50-day Simple Moving Average |
| EMA20 | 20-day Exponential Moving Average |
| RSI | 14-day Relative Strength Index |
| MACD | 12/26-day MACD |
| Signal | 9-day Signal Line (EMA of MACD) |
| BB_Upper | Bollinger Band Upper (20-day, 2σ) |
| BB_Lower | Bollinger Band Lower (20-day, 2σ) |
| Volatility | 20-day rolling std of Returns |
| ATR | 14-day Average True Range |
| Volume | Daily trading volume |
| Returns_Lag1 | 1-day lagged daily return |
| Returns_Lag5 | 5-day lagged daily return |
| Index_Return | Daily return of the KLSE Composite Index (`^KLSE`) |
| FX_Return | Daily return of MYR/USD exchange rate (`MYR=X`) |

**Scaling:** RobustScaler (robust to outliers), fit on training rows only
within each walk-forward fold — never on the full dataset before splitting.

**XGBoost feature engineering:** `xgb_engineer_features()` adds momentum,
rolling means, rolling std, and lagged views across the 60-day window on
top of the 17 base features, giving XGBoost richer time-series structure
than a plain flatten would provide.

---

## Prediction Target

The models predict **5-day forward return** (`Target_Return`):

```
Target_Return = (Close[t+5] / Close[t]) - 1
```

The dashboard converts this back to a Ringgit price for display:

```
Forecast Price = Current Close × (1 + Predicted Return)
```

This avoids recursive multi-step error accumulation entirely — the model
makes a single direct prediction for the 5-day horizon, not five
consecutive 1-day predictions fed back into each other.

---

## Models

| Model | Architecture | Input |
|-------|-------------|-------|
| XGBoost | Gradient Boosting with engineered features | Flattened + enriched 60-day window |
| LSTM | 2-layer stacked + Dropout (96 units) | 60-day sequence (3D) |
| GRU | 2-layer stacked + Dropout (96 units) | 60-day sequence (3D) |

Champion is selected via `select_champion()` in `engine_final.py`, a
weighted composite score (default 50% MAE / 20% Directional Accuracy /
30% Sharpe Ratio, all three min-max normalised across candidates)
computed after reviewing
`run_tournament.py` output. Once set in `app.py`, only that model is
used in the live dashboard.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error in RM — primary error metric; largest weight (50%) in the champion selection score |
| Directional Accuracy | % of predictions with correct up/down direction, anchored to previous actual close; smallest weight (20%) in the champion selection score |
| Sharpe Ratio | Annualised risk-adjusted return of a long/short strategy driven by the model signal (approximation — ignores transaction costs); 30% weight in the champion selection score |

All metrics computed in RM space on **actual closing prices** (not on scaled
or transformed values) so they are directly interpretable.

---

## Sentiment and Market Intelligence Pipeline

1. **RSS scraping** — `feedparser` pulls headlines from 5 Malaysian sources:
   The Edge Malaysia, Bernama, Free Malaysia Today, Malay Mail, The Sun Malaysia
2. **Fuzzy matching** — company aliases + `SequenceMatcher` catches tickers
   (`1155`), abbreviations (`MBB`, `KLK`), and near-typos (`Maybnk`)
3. **Macro context** — Bank Negara Malaysia OPR fetched live via BNM Open API;
   CPI and GDP growth use documented static fallbacks
4. **AI agent** — Llama 3 via local Ollama returns `sentiment_label`
   (Positive / Negative / Neutral), `macro_impact`, and `qualitative_rationale`
5. **Graceful fallback** — if no headlines are matched or Ollama is unreachable,
   a documented Neutral default is returned; the dashboard never crashes

---

## Decision Support Layer

- **5-day price forecast** — champion model predicts forward return, converted
  to a single RM target price
- **Stop-loss advisory** — flags whether the forecast breaches a user-adjustable
  threshold (default 5%) and suggests a sell price if so
- **Suggested action (Buy / Sell / Hold)** — combines expected return,
  sentiment label, and stop-loss breach into one signal
- **Signal conflict warning** — shown when the model is bullish but sentiment
  is Negative, or vice versa

---

## Dashboard Layout

Two-tab Streamlit interface:

**Market Overview** — all 10 stocks in one table:
`Sl. No. | Stock Name | Expected Return % | Trend | Sentiment | Sentiment Reason | Suggested Action`

**Stock Deep Dive** (selected stock):
- Projected vs Actual price chart (back-test + forecast point)
- Model information (champion name, directional accuracy, MAE)
- Active hyperparameters expander
- Loss prevention panel (suggested action + stop-loss advisory)
- Sentiment and macro rationale (Llama 3 agent output)

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras — LSTM, GRU |
| Gradient Boosting | XGBoost with custom feature engineering |
| Scaling | RobustScaler (scikit-learn) |
| Hyperparameter Tuning | Optuna (TPE sampler) with RandomizedSearchCV fallback |
| Data Source | Yahoo Finance (yfinance) — prices, KLSE index, MYR/USD FX |
| News Ingestion | feedparser — 5 Malaysian RSS feeds |
| Sentiment AI | Llama 3 via Ollama (langchain-ollama) — local, no paid API |
| Macro Data | BNM Open API (OPR live), static CPI/GDP fallback |
| Dashboard | Streamlit + Plotly |
| Caching | joblib (disk) + Streamlit cache_data / cache_resource (in-memory) |

---

## Disclaimer

This tool is a research and educational Decision Support System. It does not
constitute financial advice. Suggested actions are model outputs for research
purposes only. Past model performance is not indicative of future results.
All data sourced from Yahoo Finance, Bank Negara Malaysia, and public RSS feeds
under their respective terms of service.

*Developed by Akarsh Sriram (Student ID: TP105695), under the guidance of
Dr. Preethi Subramanian, as part of the MSc Data Science and Business Analytics
programme, Intake APDMF2508DSBA(BI), Asia Pacific University of Technology and
Innovation (APU), in collaboration with De Montfort University.*
