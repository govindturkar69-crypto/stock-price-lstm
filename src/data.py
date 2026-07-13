from __future__ import annotations
import datetime as dt
from typing import Any
import numpy as np
import pandas as pd

def _synthetic_ohlcv(start: str, end: str | None, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp(end) if end else pd.Timestamp(dt.date.today())
    idx = pd.bdate_range(start_d, end_d)
    n = len(idx)
    (mu, sigma) = (0.0004, 0.015)
    shocks = rng.normal(mu, sigma, n)
    close = 100 * np.exp(np.cumsum(shocks))
    intraday = np.abs(rng.normal(0, 0.01, n)) * close
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    volume = rng.integers(5000000, 50000000, n).astype(float)
    df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=idx)
    df.index.name = 'Date'
    return df

def load_prices(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    d = cfg['data']
    (ticker, start, end) = (d['ticker'], d['start'], d.get('end'))
    interval = d.get('interval', '1d')
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise ValueError('Empty frame returned from yfinance')
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        raw = raw[[c for c in cols if c in raw.columns]].dropna()
        raw.index.name = 'Date'
        return (raw, f'yfinance:{ticker}')
    except Exception as exc:
        if not d.get('synthetic_fallback', True):
            raise
        print(f'[data] yfinance download failed ({exc!r}); using synthetic data.')
        return (_synthetic_ohlcv(start, end, seed=cfg.get('output', {}).get('seed', 42)), 'synthetic')
