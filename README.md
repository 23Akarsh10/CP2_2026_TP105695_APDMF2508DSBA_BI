# Stock Predictor — Decision Support System

A modular Python Decision Support System (DSS) for Bursa Malaysia stock market
prediction, built using the **CRISP-DM** methodology.

Compares three machine learning models — **LSTM**, **GRU**, and **XGBoost** —
against 10 highly liquid Bursa stocks, overlays a locally-run **AI sentiment
agent** (live news + macro indicators), and surfaces everything through an
interactive Streamlit dashboard with loss-prevention and suggested-action
signals.

---

## Project Timeline & Milestones

| Milestone | Deadline | Description |
|-----------|----------|-------------|
| **M1** | 31 May | Automated data ingestion pipeline + feature engineering complete |
| **M2** | 28 June | Full model tournament + sentiment engine complete |
| **M3** | 12 July | UI + back-end integration + live testing complete |
| **M4** | 19 July | Walk-forward validation verified + report draft complete |
| **M5** | 26 July | Final submission — source code + capstone report |

---

## CRISP-DM Phase Mapping

| Phase | Module | Gantt Task |
|-------|--------|------------|
| Business Understanding | `README.md`, `app.py` | Project Initialisation (May W3) |
| Data Understanding | `engine_final.py → fetch_data()` | Data Collection + Cleaning (May W3–W4) |
| Data Preparation | `engine_final.py → add_features(), compute_indicators(), StockPreprocessor` | Data Collection + Cleaning (May W3–W4) |
| Modelling | `engine_final.py → build_lstm(), build_gru(), build_xgboost(), walk_forward_validation()` | Model Development & Comparison (June W1–W4) |
| Evaluation | `engine_final.py → directional_accuracy(), sharpe_ratio(), compute_shap_importance(), compare_forecasting_strategies()` | Validation (July W2–W4) |
| Deployment | `app.py` (Streamlit dashboard, two-stage joblib/session caching) | UI Development + System Integration (July W1–W2) |

---

## File Structure

```
stock_predictor/
├── app.py              # Streamlit dashboard (UI layer)
├── engine_final.py     # ML backend (data, features, models, sentiment agent, forecasting)
├── requirements.txt    # Python dependencies
├── README.md            # This file
└── GUIDE.md             # Step-by-step setup and milestone guide
```

---

## Stock Universe (Bursa Malaysia — 10 Highly Liquid Stocks)

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

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras (LSTM, GRU) |
| Gradient Boosting | XGBoost |
| Hyperparameter Tuning | Optuna (TPE sampler) with RandomizedSearchCV / manual grid fallback |
| Explainability | SHAP (TreeExplainer on XGBoost) |
| Data Source | Yahoo Finance API (yfinance) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| News Ingestion | feedparser (RSS: The Edge Malaysia, Bernama, Free Malaysia Today, Malay Mail, The Sun Malaysia) |
| Sentiment Analysis | Local LLM agent (Llama 3 via Ollama/LangChain) reasoning over matched headlines + macro data |
| Macro Data | Bank Negara Malaysia (BNM) OPR API, with static CPI/GDP fallback |
| Caching | joblib (disk) + Streamlit `cache_data` / `cache_resource` (in-memory) |
| Dashboard | Streamlit, Plotly |

---

## Models

| Model | Type | Input |
|-------|------|-------|
| LSTM | 2-layer stacked + Dropout | 60-day sequence (3D) |
| GRU | 2-layer stacked + Dropout | 60-day sequence (3D) |
| XGBoost | Gradient Boosting | Flattened 60-day window (2D) |

Champion model per stock = lowest walk-forward MAE (5-fold, leakage-free,
scaler refit per fold). Once selected, only the champion is retrained on
each session refresh; re-comparing all three models is a manual, separate
action in the dashboard ("Rebuild model selection").

---

## Features Engineered

- Close Price & Daily Returns
- MA20 (20-day Moving Average)
- MA50 (50-day Moving Average)
- EMA20 (20-day Exponential Moving Average)
- RSI (14-day Relative Strength Index)
- MACD (12/26-day) + Signal Line (9-day EMA of MACD)
- Bollinger Bands (20-day, ±2σ Upper/Lower)
- Rolling Volatility (20-day std of returns)
- ATR (14-day Average True Range)
- Volume

All features normalised via Min-Max scaling to [0, 1], fit only on training
data within each walk-forward fold to prevent leakage.

---

## Sentiment & Market Intelligence

A local AI agent (Llama 3, served via Ollama — no paid API) analyses each
stock independently:

1. Live RSS headlines are pulled from five Malaysian news sources and
   fuzzy-matched to each company (including ticker/alias matching).
2. Malaysia's Overnight Policy Rate (OPR) is fetched live from the BNM API,
   with CPI and GDP growth using documented static fallbacks.
3. The LLM is prompted to return a structured `sentiment_label`
   (Positive / Negative / Neutral), `macro_impact`, and
   `qualitative_rationale`.
4. If no matching headlines are found, or the LLM/parse step fails, a
   documented Neutral default is returned instead of a guess.

---

## Decision Support Layer

- **5-day forecast** — the champion model recursively forecasts 5 trading
  days ahead, regenerating technical indicators at each step.
- **Stop-loss advisory** — flags whether the forecast path breaches a
  user-adjustable stop-loss threshold (default 5%) and on which day.
- **Suggested action (Buy / Sell / Hold)** — combines expected 5-day return,
  sentiment label, and stop-loss breach status into one signal, with a
  signal-conflict warning when sentiment and forecast direction disagree.
- **Recursive vs. direct multistep comparison** — `compare_forecasting_strategies()`
  benchmarks recursive single-step forecasting against direct multistep
  XGBoost forecasting across multiple historical origins (RMSE/MAE/MAPE).

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — primary champion selector |
| Directional Accuracy | % of periods with correct up/down prediction |
| Sharpe Ratio | Annualised risk-adjusted return of forecast strategy |
| RMSE / MAPE | Used when comparing recursive vs. direct multistep forecasting |
| SHAP Importance | Mean absolute SHAP value per feature (XGBoost) |

---

## Dashboard (`app.py`)

Two-tab Streamlit interface:

- **Market Overview** — all 10 stocks processed and ranked by expected
  5-day return, with champion model, sentiment, and suggested action per
  stock.
- **Stock Deep Dive** — projected-vs-actual price chart (back-test +
  forecast), model information (champion, directional accuracy, MAE),
  loss-prevention panel (suggested action + stop-loss advisory), and the
  full sentiment/macro rationale for the selected stock.

Results refresh automatically once per session checkpoint (morning /
half-day / end-of-day, based on Malaysian trading hours), backed by a
two-stage cache:

- **Stage A — model selection** (all 3 algorithms compared): disk-cached
  per symbol, only reruns when explicitly rebuilt.
- **Stage B — champion forecast**: disk-cached per (symbol, session
  checkpoint), retrains only the champion model on the latest data.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but required for live sentiment) Install Ollama and pull Llama 3
#    https://ollama.com  →  ollama pull llama3

# 3. Launch dashboard
streamlit run app.py
```

If Ollama is not installed or unreachable, the sentiment agent gracefully
falls back to a documented "Neutral — no data available" response; the
rest of the dashboard (forecasts, technical models, stop-loss) is
unaffected.

See `GUIDE.md` for full step-by-step setup instructions.

---

## Disclaimer

This tool is a research and educational Decision Support System.
It does not constitute financial advice. Suggested actions are model
outputs for research purposes only. Past model performance is
not indicative of future results. All data sourced from Yahoo Finance,
Bank Negara Malaysia, and public RSS feeds under their respective terms
of service.

Developed by Akarsh Sriram (Student ID: TP105695), under the guidance of
Dr. Preethi Subramanian, as part of the MSc Data Science and Business
Analytics programme, Intake APDMF2508DSBA(BI), Asia Pacific University of
Technology and Innovation (APU), in collaboration with De Montfort
University.
