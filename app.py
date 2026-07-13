"""Streamlit dashboard for the LSTM stock price prediction project.

Wraps the existing pipeline (src/) in an interactive, professional UI: live
yfinance data, LSTM training, honest return-space evaluation, walk-forward
backtest, a class-balanced direction classifier, and feature importance.

Run:  streamlit run app.py
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import load_config, set_seed
from src.data import load_prices
from src.features import add_features
from src.dataset import prepare
from src.train import train_model
from src.model import torch_predict
from src.evaluate import (
    predict_prices, predict_returns, regression_metrics,
    return_metrics, directional_metrics, naive_baseline, walk_forward,
)
from src.classify_data import prepare_classification
from src.classify_train import train_classifier, predict_proba

REPO_URL = "https://github.com/govindturkar69-crypto/stock-price-lstm"
ACCENT, GREEN, RED, MUTED = "#3b82f6", "#22c55e", "#ef4444", "#8b97a8"
CARD, BORDER = "#151b28", "#232c3d"
POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY",
           "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]

# --- Input validation (defence-in-depth: only clean values reach yfinance) ---
_TICKER_RE = re.compile(r'^[A-Za-z0-9^][A-Za-z0-9.\-^=]{0,14}$')

def valid_ticker(t: str) -> bool:
    """Allowlist ticker chars (letters, digits, . - ^ =). Blocks injection / junk."""
    return bool(_TICKER_RE.match(t or ''))

def valid_date(d: str) -> bool:
    """Require an ISO YYYY-MM-DD start date."""
    try:
        datetime.strptime(d, '%Y-%m-%d')
        return True
    except Exception:
        return False

st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    """Inject custom CSS for a clean dark financial-dashboard aesthetic."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
    #MainMenu, footer {{ visibility:hidden; }}
    .block-container {{ padding-top:1.4rem; padding-bottom:2rem; max-width:1320px; }}
    .hero {{ background:linear-gradient(120deg,#111a2e 0%,#0b0f19 65%);
      border:1px solid {BORDER}; border-radius:18px; padding:24px 30px; margin-bottom:8px; }}
    .hero h1 {{ margin:0; font-size:29px; font-weight:800; color:#f4f7fb; letter-spacing:-.4px; }}
    .hero p {{ margin:6px 0 0; color:{MUTED}; font-size:14px; max-width:840px; }}
    .badge {{ display:inline-block; background:{ACCENT}22; color:{ACCENT}; border:1px solid {ACCENT}55;
      padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin:10px 6px 0 0; }}
    .sec {{ font-size:19px; font-weight:700; margin:4px 0 2px; color:#eef2f8;
      border-left:3px solid {ACCENT}; padding-left:10px; }}
    .verdict {{ border-radius:14px; padding:14px 20px; font-weight:600; margin-top:6px; border:1px solid; }}
    .stButton>button {{ border-radius:10px; font-weight:700; height:44px; }}
    div[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER};
      border-radius:14px; padding:14px 16px; }}
    </style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker: str, start: str) -> dict:
    """Fetch fresh OHLCV from yfinance. synthetic_fallback off so a bad ticker
    raises (handled gracefully). TTL keeps it live without re-hitting the API
    on every widget change."""
    cfg = load_config("config.yaml")
    cfg["data"].update({"ticker": ticker, "start": start, "end": None,
                        "synthetic_fallback": False})
    try:
        raw, source = load_prices(cfg)
        return {"ok": True, "df": raw, "source": source,
                "ts": datetime.now().strftime("%d %b %Y, %H:%M:%S"), "err": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "df": None, "source": None, "ts": None, "err": str(e)}


def _permutation_importance(model, prep, n_repeats: int = 2) -> list:
    """Shuffle each feature and measure the rise in test error (no retraining)."""
    base = torch_predict(model, prep.X_test)
    base_err = float(np.mean((base - prep.y_test) ** 2))
    rng = np.random.default_rng(0)
    imps = []
    for j in range(prep.X_test.shape[2]):
        errs = []
        for _ in range(n_repeats):
            Xp = prep.X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, j] = Xp[perm][:, :, j]
            errs.append(float(np.mean((torch_predict(model, Xp) - prep.y_test) ** 2)))
        imps.append(max(0.0, np.mean(errs) - base_err))
    total = sum(imps) or 1.0
    return sorted(zip(prep.feature_names, [i / total for i in imps]),
                 key=lambda x: x[1], reverse=True)


@st.cache_data(show_spinner=False)
def train_and_eval(ticker, start, horizon, lookback, epochs) -> dict:
    """Train + evaluate one horizon; return a picklable results dict."""
    fetched = fetch_prices(ticker, start)
    if not fetched["ok"]:
        return {"ok": False, "err": fetched["err"]}
    cfg = load_config("config.yaml")
    cfg["data"].update({"ticker": ticker, "start": start, "synthetic_fallback": False})
    cfg["window"].update({"horizon": int(horizon), "lookback": int(lookback)})
    cfg["train"]["epochs"] = int(epochs)
    cfg["walk_forward"]["enabled"] = False
    set_seed(cfg["output"]["seed"])
    feat = add_features(fetched["df"], cfg)
    prep = prepare(feat, cfg)
    model, history = train_model(prep, cfg, cfg["output"]["dir"])
    y_true, y_pred = predict_prices(model, prep)
    true_ret, pred_ret = predict_returns(model, prep)
    ctx = feat["Close"].iloc[-300:]
    return {
        "ok": True, "source": fetched["source"], "ts": fetched["ts"],
        "n_features": len(prep.feature_names),
        "price": regression_metrics(y_true, y_pred), "naive": naive_baseline(y_true),
        "ret": return_metrics(true_ret, pred_ret), "dir": directional_metrics(true_ret, pred_ret),
        "ctx_dates": [d.strftime("%Y-%m-%d") for d in ctx.index], "ctx_close": ctx.values.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in prep.test_index],
        "test_true": y_true.tolist(), "test_pred": y_pred.tolist(),
        "history": {k: list(map(float, v)) for k, v in history.history.items()},
        "importance": _permutation_importance(model, prep),
    }


@st.cache_data(show_spinner=False)
def run_walk_forward(ticker, start, horizon, lookback, epochs) -> dict:
    """Expanding-window walk-forward; per-fold return-space skill."""
    fetched = fetch_prices(ticker, start)
    if not fetched["ok"]:
        return {"ok": False, "err": fetched["err"]}
    cfg = load_config("config.yaml")
    cfg["data"].update({"ticker": ticker, "start": start, "synthetic_fallback": False})
    cfg["window"].update({"horizon": int(horizon), "lookback": int(lookback)})
    cfg["train"]["epochs"] = int(epochs)
    set_seed(cfg["output"]["seed"])
    feat = add_features(fetched["df"], cfg)

    def _fold(df, c):
        p = prepare(df, c)
        m, _ = train_model(p, c, c["output"]["dir"])
        tr, pr = predict_returns(m, p)
        out = return_metrics(tr, pr)
        out["DirAcc_pct"] = directional_metrics(tr, pr)["DirAcc_pct"]
        return out

    wf = walk_forward(feat, cfg, _fold)
    return {"ok": True, "folds": wf.index.tolist(),
            "ic": wf["IC_corr"].tolist(), "diracc": wf["DirAcc_pct"].tolist()}


@st.cache_data(show_spinner=False)
def run_classifier(ticker, start, horizon, lookback, epochs) -> dict:
    """Class-balanced up/down classifier; confusion matrix + AUC/balanced-acc."""
    fetched = fetch_prices(ticker, start)
    if not fetched["ok"]:
        return {"ok": False, "err": fetched["err"]}
    cfg = load_config("config.yaml")
    cfg["data"].update({"ticker": ticker, "start": start, "synthetic_fallback": False})
    cfg["window"].update({"horizon": int(horizon), "lookback": int(lookback)})
    cfg["train"]["epochs"] = int(epochs)
    set_seed(cfg["output"]["seed"])
    feat = add_features(fetched["df"], cfg)
    prep = prepare_classification(feat, cfg)
    model, _ = train_classifier(prep, cfg, cfg["output"]["dir"])
    proba = predict_proba(model, prep.X_test)
    pred = (proba >= 0.5).astype(int)
    y = prep.y_test.astype(int)
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    up_rec = tp / (tp + fn) if tp + fn else 0.0
    dn_rec = tn / (tn + fp) if tn + fp else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        auc = float("nan")
    return {"ok": True, "cm": [[tn, fp], [fn, tp]],
            "balanced_acc": (up_rec + dn_rec) / 2, "roc_auc": auc}


def _layout(fig, height, title=""):
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02, x=0),
                      xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
                      font=dict(color="#c8d2e0"))
    return fig


def chart_predictions(res, height=380):
    """Actual vs predicted line chart with the hold-out region shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["ctx_dates"], y=res["ctx_close"], name="Actual price",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_pred"], name="Predicted (test)",
                             line=dict(color=GREEN, width=2, dash="dash")))
    fig.add_vrect(x0=res["test_dates"][0], x1=res["test_dates"][-1], fillcolor=GREEN,
                  opacity=0.06, line_width=0, annotation_text="Hold-out (test)",
                  annotation_position="top left")
    return _layout(fig, height)


def chart_bar(x, y, color_by_sign=False, height=320, title="", ylab=""):
    colors = ([GREEN if v >= 0 else RED for v in y] if color_by_sign else ACCENT)
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.update_yaxes(title=ylab)
    return _layout(fig, height, title)


def chart_confusion(cm, height=340):
    labels = ["Down", "Up"]
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {l}" for l in labels],
                               y=[f"True {l}" for l in labels], colorscale="Blues",
                               text=cm, texttemplate="%{text}", showscale=False))
    return _layout(fig, height)


def render_header():
    st.markdown("""
    <div class="hero">
      <h1>📈 LSTM Stock Price Predictor</h1>
      <p>A deep-learning dashboard that trains a stacked LSTM on <b>live Yahoo Finance</b>
      data, then evaluates it <b>honestly</b> with return-space metrics and walk-forward
      validation — not a flattering price-level accuracy number.</p>
      <div>
        <span class="badge">Python</span><span class="badge">PyTorch · LSTM</span>
        <span class="badge">yfinance</span><span class="badge">Walk-forward eval</span>
      </div>
    </div>""", unsafe_allow_html=True)


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🎛️ Control panel")
        st.markdown("**Ticker**")
        pick = st.selectbox("Popular", POPULAR, index=0, label_visibility="collapsed")
        custom = st.text_input("Or any ticker (US, or .NS for India)", "").strip().upper()
        ticker = custom or pick
        st.caption(f"🟢 **Live data** — {ticker} fetched fresh from yfinance")
        st.markdown("---")
        st.markdown("**Model settings**")
        start = st.text_input("History start", load_config("config.yaml")["data"]["start"])
        horizon = st.radio("Prediction horizon", [1, 5], horizontal=True,
                           format_func=lambda d: f"{d}-day")
        lookback = st.slider("Lookback window", 20, 120, 60, step=10)
        epochs = st.slider("Max epochs", 5, 100, 30, step=5,
                           help="Fewer = faster demo; early stopping still applies.")
        st.markdown("---")
        run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        if st.session_state.get("last_ts"):
            st.caption(f"Last fetched: {st.session_state['last_ts']}")
    return {"ticker": ticker, "start": start, "horizon": int(horizon),
            "lookback": int(lookback), "epochs": int(epochs), "run": run}


def render_metrics(res):
    st.markdown('<div class="sec">🎯 Performance metrics</div>', unsafe_allow_html=True)
    st.caption("Daily returns are near a random walk — read these honestly. IC/R² near 0 "
               "and directional accuracy near the base rate mean *no edge*, which is the "
               "expected, truthful result (not a failure).")
    rm, dm, pm, nv = res["ret"], res["dir"], res["price"], res["naive"]
    c = st.columns(4)
    c[0].metric("Information Coefficient", f"{rm['IC_corr']:+.3f}", f"{rm['IC_corr']:+.3f} vs 0",
                help="Correlation of predicted vs actual returns. >0 = signal.")
    c[1].metric("Return R²", f"{rm['ret_R2']:+.3f}", f"{rm['ret_R2']:+.3f} vs 0",
                help="Vs a 'predict the mean return' baseline. >0 = signal.")
    c[2].metric("Directional accuracy", f"{dm['DirAcc_pct']:.1f}%",
                f"{dm['DirAcc_pct'] - dm['base_up_rate_pct']:+.1f}% vs base",
                help="Correct up/down calls vs the base up-rate.")
    c[3].metric("Price RMSE", f"{pm['RMSE']:.2f}", f"{pm['RMSE'] - nv['RMSE']:+.2f} vs naive",
                delta_color="inverse", help="Lower is better; vs naive persistence.")
    signal = (rm["IC_corr"] > 0) or (rm["ret_R2"] > 0) or (dm["DirAcc_pct"] > dm["base_up_rate_pct"])
    if signal:
        st.markdown(f'<div class="verdict" style="background:{GREEN}15;border-color:{GREEN}55;color:{GREEN}">'
                    '✓ Some predictive signal detected on this run.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict" style="background:{RED}12;border-color:{RED}44;color:#f6a5a5">'
                    '✕ No reliable signal — daily prices behave near-random. This is the expected, honest result.</div>',
                    unsafe_allow_html=True)


def render_footer():
    st.divider()
    st.markdown(
        f"<div style='color:{MUTED};font-size:13px'>"
        f"🔗 <a href='{REPO_URL}' style='color:{ACCENT}'>GitHub repository</a> &nbsp;·&nbsp; "
        "⚠️ <b>For educational purposes only — not financial advice.</b><br>"
        "Built with PyTorch, Streamlit &amp; yfinance · LSTM stock prediction project.</div>",
        unsafe_allow_html=True)


def main():
    inject_css()
    render_header()
    p = render_sidebar()

    if p["run"]:
        st.session_state["ran"] = True
        st.session_state["params"] = p
    if not st.session_state.get("ran"):
        st.divider()
        st.info("👈 Ticker aur settings chuno, phir **Run Prediction** dabao. "
                "Empty-state — abhi tak koi model train nahi hua.")
        render_footer()
        return

    p = st.session_state["params"]

    # Validate untrusted user input before it reaches any external call.
    if not valid_ticker(p["ticker"]):
        st.error("⚠️ Invalid ticker format. Use letters/digits only, e.g. AAPL or TCS.NS.")
        render_footer(); return
    if not valid_date(p["start"]):
        st.error("⚠️ Invalid start date. Use YYYY-MM-DD, e.g. 2015-01-01.")
        render_footer(); return

    with st.spinner("📥 Fetching live data from yfinance..."):
        fetched = fetch_prices(p["ticker"], p["start"])
    if not fetched["ok"]:
        print(f"[fetch-error] {p['ticker']}: {fetched['err']}")  # server log only
        st.error("❌ Could not fetch data for **" + p["ticker"] + "**. "
                 "Check the ticker (use .NS for Indian stocks, e.g. TCS.NS) or try a different date range.")
        render_footer()
        return
    st.session_state["last_ts"] = fetched["ts"]

    with st.spinner(f"🧠 Training LSTM on {p['ticker']} ({p['horizon']}-day horizon)..."):
        res = train_and_eval(p["ticker"], p["start"], p["horizon"], p["lookback"], p["epochs"])
    if not res["ok"]:
        print(f"[train-error] {p['ticker']}: {res['err']}")  # server log only
        st.error("❌ Model training failed for this input. Try a different ticker or date range.")
        render_footer()
        return

    s = st.columns(4)
    s[0].metric("Ticker", p["ticker"])
    s[1].metric("Data source", "🟢 Live")
    s[2].metric("Features", res["n_features"])
    s[3].metric("Fetched at", fetched["ts"].split(",")[-1].strip())

    st.divider()
    st.markdown('<div class="sec">📊 Actual vs Predicted</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_predictions(res), use_container_width=True)

    st.divider()
    render_metrics(res)

    st.divider()
    st.markdown('<div class="sec">⚖️ Horizon comparison (1-day vs 5-day)</div>', unsafe_allow_html=True)
    if st.button("Run 1-day vs 5-day comparison"):
        st.session_state["cmp"] = True
    if st.session_state.get("cmp"):
        with st.spinner("Training both horizons..."):
            r1 = train_and_eval(p["ticker"], p["start"], 1, p["lookback"], p["epochs"])
            r5 = train_and_eval(p["ticker"], p["start"], 5, p["lookback"], p["epochs"])
        if r1["ok"] and r5["ok"]:
            cc = st.columns(2)
            cc[0].plotly_chart(chart_bar(["1-day", "5-day"], [r1["ret"]["IC_corr"], r5["ret"]["IC_corr"]],
                               color_by_sign=True, title="Information Coefficient", ylab="IC"),
                               use_container_width=True)
            cc[1].plotly_chart(chart_bar(["1-day", "5-day"], [r1["dir"]["DirAcc_pct"], r5["dir"]["DirAcc_pct"]],
                               title="Directional accuracy (%)", ylab="%"), use_container_width=True)
    else:
        st.caption("Click to train both horizons and compare their skill side by side.")

    st.divider()
    st.markdown('<div class="sec">🔁 Walk-forward backtest</div>', unsafe_allow_html=True)
    st.caption("Expanding-window folds, each trained on the past and tested on the next block. "
               "Sign flips across folds = noise, not signal — the project's honest core.")
    if st.button("Run walk-forward validation"):
        st.session_state["wf"] = True
    if st.session_state.get("wf"):
        with st.spinner("Running walk-forward folds (trains several models)..."):
            wf = run_walk_forward(p["ticker"], p["start"], p["horizon"], p["lookback"], p["epochs"])
        if wf["ok"]:
            st.plotly_chart(chart_bar([f"Fold {f}" for f in wf["folds"]], wf["ic"],
                            color_by_sign=True, title="IC per fold", ylab="IC"), use_container_width=True)
            st.caption(f"Mean IC across folds: **{np.mean(wf['ic']):+.3f}** "
                       "(near 0 with sign flips → no reliable edge).")
    else:
        st.caption("Click to run the expanding-window backtest.")

    st.divider()
    st.markdown('<div class="sec">🧭 Direction classifier (class-balanced)</div>', unsafe_allow_html=True)
    if st.button("Run up/down classifier"):
        st.session_state["clf"] = True
    if st.session_state.get("clf"):
        with st.spinner("Training class-balanced classifier..."):
            clf = run_classifier(p["ticker"], p["start"], p["horizon"], p["lookback"], p["epochs"])
        if clf["ok"]:
            cc = st.columns([1, 1])
            cc[0].plotly_chart(chart_confusion(clf["cm"]), use_container_width=True)
            cc[1].metric("ROC-AUC", f"{clf['roc_auc']:.3f}", help="0.5 = no skill")
            cc[1].metric("Balanced accuracy", f"{clf['balanced_acc']*100:.1f}%", help="50% = no skill")
            cc[1].caption("Both recalls > 0 = balancing worked; AUC ~0.5 = little directional signal.")
    else:
        st.caption("Click to train the balanced up/down classifier and see its confusion matrix.")

    st.divider()
    st.markdown('<div class="sec">🔬 Feature importance</div>', unsafe_allow_html=True)
    st.caption("Permutation importance: how much test error rises when each feature is shuffled.")
    names = [n for n, _ in res["importance"]][::-1]
    vals = [v for _, v in res["importance"]][::-1]
    st.plotly_chart(_layout(go.Figure(go.Bar(x=vals, y=names, orientation="h",
                    marker_color=ACCENT)), 30 + 26 * len(names)), use_container_width=True)

    st.divider()
    st.markdown('<div class="sec">⬇️ Download predictions</div>', unsafe_allow_html=True)
    out_df = pd.DataFrame({"date": res["test_dates"], "actual": res["test_true"],
                           "predicted": res["test_pred"]})
    st.download_button("Download predictions (CSV)", out_df.to_csv(index=False),
                       file_name=f"{p['ticker']}_predictions.csv", mime="text/csv")

    render_footer()


if __name__ == "__main__":
    main()
