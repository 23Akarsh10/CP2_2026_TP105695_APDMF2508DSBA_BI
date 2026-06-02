# Stock Predictor — Decision Support System

A modular Python Decision Support System (DSS) for Bursa Malaysia stock market
prediction, built using the **CRISP-DM** methodology.

Compares three machine learning models — **LSTM**, **GRU**, and **XGBoost** —
against 10 highly liquid Bursa stocks, with a live sentiment analysis overlay
and an interactive Streamlit dashboard.

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
| Data Understanding | `engine.py → fetch_data()` | Data Collection + Cleaning (May W3–W4) |
| Data Preparation | `engine.py → add_features(), StockPreprocessor` | Data Collection + Cleaning (May W3–W4) |
| Modelling | `engine.py → walk_forward_validation()` | Model Development & Comparison (June W1–W4) |
| Evaluation | `engine.py → sharpe_ratio(), directional_accuracy()` | Validation (July W2–W4) |
| Deployment | `app.py` (Streamlit dashboard) | UI Development + System Integration (July W1–W2) |

---

## File Structure

```
stock_predictor/
├── app.py              # Streamlit dashboard (UI layer)
├── engine.py           # ML backend (data, models, sentiment, forecasting)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── GUIDE.md            # Step-by-step setup and milestone guide
```

---

## Stock Universe (Bursa Malaysia — 10 Highly Liquid Stocks)

| Company | Ticker |
|---------|--------|
| Maybank | 1155.KL |
| Public Bank | 1295.KL |
| CIMB Group | 1023.KL |
| TNB | 5347.KL |
| Petronas Chemicals | 5183.KL |
| IHH Healthcare | 5225.KL |
| Press Metal | 8869.KL |
| CelcomDigi | 6947.KL |
| MISC | 3816.KL |
| Hong Leong Bank | 5819.KL |

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras (LSTM, GRU) |
| Gradient Boosting | XGBoost |
| Data Source | Yahoo Finance API (yfinance) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Sentiment Analysis | VADER, TextBlob |
| Dashboard | Streamlit, Plotly |

---

## Models

| Model | Type | Input |
|-------|------|-------|
| LSTM | 2-layer stacked + Dropout | 60-day sequence (3D) |
| GRU | 2-layer stacked + Dropout | 60-day sequence (3D) |
| XGBoost | Gradient Boosting | Flattened 60-day window (2D) |

Champion model per stock = lowest walk-forward MAE.

---

## Features Engineered

- Close Price
- MA20 (20-day Moving Average)
- MA50 (50-day Moving Average)
- RSI (14-day Relative Strength Index)
- MACD (12/26-day)
- Signal Line (9-day EMA of MACD)

All features normalised via Min-Max scaling to [0, 1].

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — primary champion selector |
| Directional Accuracy | % of periods with correct up/down prediction |
| Sharpe Ratio | Annualised risk-adjusted return of forecast strategy |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run app.py
```

See `GUIDE.md` for full step-by-step setup instructions.

---

## Disclaimer

This tool is a research and educational Decision Support System.
It does not constitute financial advice. Past model performance is
not indicative of future results. All data sourced from Yahoo Finance
under their terms of service.
