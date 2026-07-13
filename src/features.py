from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd(close: pd.Series, fast: int=12, slow: int=26, signal: int=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd, macd_signal, macd - macd_signal)

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    (high, low, close) = (df['High'], df['Low'], df['Close'])
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

def _stationary_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    f = cfg['features']
    close = df['Close']
    out = pd.DataFrame(index=df.index)
    out['Close'] = close
    for w in f.get('return_windows', [1, 5, 10, 20]):
        out[f'ret_{w}'] = np.log(close / close.shift(w))
    out['log_return'] = out.get('ret_1', np.log(close / close.shift(1)))
    for w in f.get('sma_windows', [10, 20, 50]):
        out[f'close_over_sma_{w}'] = close / close.rolling(w).mean() - 1.0
    if f.get('rsi_period'):
        out['rsi'] = _rsi(close, f['rsi_period']) / 100.0
    (macd, sig, hist) = _macd(close)
    out['macd_hist_rel'] = hist / close
    if f.get('bollinger_period'):
        w = f['bollinger_period']
        ma = close.rolling(w).mean()
        sd = close.rolling(w).std()
        (upper, lower) = (ma + 2 * sd, ma - 2 * sd)
        out['bb_pctb'] = (close - lower) / (upper - lower)
    if f.get('atr_period'):
        out['atr_rel'] = _atr(df, f['atr_period']) / close
    vw = f.get('vol_window', 20)
    if 'Volume' in df.columns:
        out['vol_ratio'] = df['Volume'] / df['Volume'].rolling(vw).mean()
    out['ret_vol'] = out['log_return'].rolling(vw).std()
    return out

def _legacy_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    f = cfg['features']
    out = df.copy()
    close = out['Close']
    out['log_return'] = np.log(close / close.shift(1))
    for w in f.get('sma_windows', []):
        out[f'sma_{w}'] = close.rolling(w).mean()
        out[f'close_over_sma_{w}'] = close / out[f'sma_{w}']
    if f.get('rsi_period'):
        out['rsi'] = _rsi(close, f['rsi_period'])
    (macd, sig, hist) = _macd(close)
    (out['macd'], out['macd_signal'], out['macd_hist']) = (macd, sig, hist)
    if f.get('bollinger_period'):
        w = f['bollinger_period']
        ma = close.rolling(w).mean()
        sd = close.rolling(w).std()
        out['bb_upper'] = ma + 2 * sd
        out['bb_lower'] = ma - 2 * sd
        out['bb_pctb'] = (close - out['bb_lower']) / (out['bb_upper'] - out['bb_lower'])
    if f.get('atr_period'):
        out['atr'] = _atr(out, f['atr_period'])
    return out

def add_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if cfg['features'].get('stationary', True):
        out = _stationary_features(df, cfg)
    else:
        out = _legacy_features(df, cfg)
    return out.replace([np.inf, -np.inf], np.nan).dropna()
