# 📈 LSTM Stock Price Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/status-deployed-brightgreen.svg">
</p>

A deep-learning dashboard that trains a stacked **LSTM (PyTorch)** on **live
Yahoo Finance** data to forecast stock prices and predict price direction — and,
crucially, evaluates it **honestly** with return-space metrics and walk-forward
validation instead of a flattering price-level accuracy number.

### 🔗 Live demo
**▶️ [stock-price-lstm.streamlit.app](https://stock-price-lstm-eaqfzb6hbgmjdf9el5thnh.streamlit.app/)**  ·  📂 [GitHub repo](https://github.com/govindturkar69-crypto/stock-price-lstm)

> ⚠️ Educational / portfolio project — **not financial advice.** The dashboard is
> designed to show *why daily stock prediction is hard* and how to measure it
> honestly, not to make money.

---

## 🎯 Why this project is different

Most "stock predictor" projects report a 95%+ accuracy that is an artifact of
predicting price *levels* (where tomorrow ≈ today). This project deliberately
avoids that trap:

1. It first shows that absolute-price prediction scores a **negative R²** (can't
   extrapolate beyond the training price range).
2. It reframes the target to **stationary log returns** and reconstructs prices.
3. It uses only **scale-free features** so the model never sees raw price levels.
4. It evaluates in **return space** (IC, R² vs. mean) and with **walk-forward
   folds**, then reports the honest conclusion: daily returns are near a random
   walk and carry little reliable signal.

That intellectual honesty — plus a clean, deployed, well-engineered pipeline — is
the point.

## ✨ Features

- **Live data** via `yfinance` — any ticker: US (`AAPL`, `MSFT`) or Indian
  (`TCS.NS`, `RELIANCE.NS`), with a synthetic fallback for offline use.
- **Stationary feature engineering** — multi-horizon log returns, price/MA
  ratios, RSI, Bollinger %B, price-relative MACD & ATR, volume ratio, rolling
  volatility (pure pandas, no TA-Lib).
- **Stacked LSTM** (optionally bidirectional) with dropout, Huber loss, early
  stopping, LR-on-plateau, weight decay and gradient clipping.
- **Two tasks** — return regression (`main.py`) and a class-balanced up/down
  direction classifier (`classify.py`).
- **Honest evaluation** — price / return / direction metrics + expanding-window
  walk-forward validation.
- **Professional Streamlit dashboard** (`app.py`) — interactive Plotly charts,
  metric cards, horizon comparison, walk-forward view, confusion matrix, feature
  importance, and CSV download.
- **Security-reviewed** — input validation + sanitized errors; see
  [`SECURITY.md`](SECURITY.md).
- Runs on **CUDA / Apple MPS / CPU** automatically.

## 🖥️ The dashboard

The Streamlit app wraps the whole pipeline in an interactive UI:

- Pick a ticker (US or `.NS` Indian), horizon (1-day / 5-day), lookback, epochs.
- 🟢 **Live data** badge + "last fetched" timestamp (proves data is fresh).
- Actual-vs-predicted chart with the hold-out region shaded.
- Metric cards: **IC, Return R², Directional Accuracy, RMSE** — read honestly.
- Horizon comparison, walk-forward backtest, direction-classifier confusion
  matrix, permutation feature importance, and a **Download predictions (CSV)**
  button.

## 🚀 Quickstart

```bash
git clone https://github.com/govindturkar69-crypto/stock-price-lstm.git
cd stock-price-lstm
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

streamlit run app.py                 # launch the dashboard
# or run the pipelines directly:
python main.py                       # regression + honest metrics
python classify.py                   # class-balanced direction classifier
```

If pip can't find a `torch` wheel for your Python, get the install command from
https://pytorch.org/get-started/locally/ (a CPU build is fine), then re-run
`pip install -r requirements.txt`.

## 📊 How to read the metrics

| Space | Metric | Honest reading |
|-------|--------|----------------|
| Price | R² ~0.98 | **Not skill** — predictions are anchored to yesterday's actual price. |
| Return | IC, R² vs. mean | **The real test.** ~0 = no signal; >0 = genuine edge. |
| Direction | Balanced acc, ROC-AUC | 0.5 / base-rate = coin flip. Beat it across folds = real. |

**Key finding:** across price regression, return regression (1-day & 5-day), and
a balanced classifier, the model shows **no reliable signal** on daily data —
IC ≈ 0, ROC-AUC ≈ 0.5, sign flips across folds. Expected for near-random-walk
prices, and reported honestly.

## 🗂 Project structure

```
stock-price-lstm/
├── app.py                  # Streamlit dashboard
├── main.py                 # regression pipeline
├── classify.py             # direction classifier
├── config.yaml             # all settings
├── requirements.txt
├── README.md · SECURITY.md · LICENSE
├── .streamlit/config.toml  # dashboard theme
└── src/
    ├── config.py · data.py · features.py · dataset.py
    ├── model.py · train.py · evaluate.py · predict.py
    ├── classify_data.py · classify_train.py · plots.py
```

## 🔒 Security

Reviewed against the five Emergent security checks (Gitleaks, Bearer, ECC,
Trail of Bits). No secrets, no personal data, no database/auth/payments. Applied
hardenings: **input validation** (ticker allowlist + ISO date) and **sanitized
client-facing errors**. Full report: [`SECURITY.md`](SECURITY.md).

## ☁️ Deployment

Deployed on **Streamlit Community Cloud** from this repo (`app.py`). To deploy
your own fork: push to GitHub → [share.streamlit.io](https://share.streamlit.io)
→ select repo + `app.py` → Deploy.

## 🛠 Tech stack

Python · PyTorch · Streamlit · Plotly · NumPy · pandas · scikit-learn · yfinance

## 📄 License

[MIT](LICENSE) © Govind

---
<p align="center"><sub>For educational purposes only — not financial advice.</sub></p>
