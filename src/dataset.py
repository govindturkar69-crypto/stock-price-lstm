from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class Prepared:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: list[str]
    target_name: str
    test_index: pd.DatetimeIndex
    target_mode: str
    test_anchor: np.ndarray
    test_true_price: np.ndarray
    last_close: float
    horizon: int

def _sequences_with_index(features, target, lookback, horizon):
    (X, y, idx) = ([], [], [])
    n = len(features)
    for i in range(lookback, n - horizon + 1):
        X.append(features[i - lookback:i])
        t = i + horizon - 1
        y.append(target[t])
        idx.append(t)
    return (np.asarray(X), np.asarray(y), np.asarray(idx))

def prepare(df: pd.DataFrame, cfg) -> Prepared:
    target_name = cfg['data']['target']
    mode = cfg['data'].get('target_mode', 'log_return')
    exclude = set(cfg['data'].get('exclude_features', []))
    lookback = cfg['window']['lookback']
    horizon = cfg['window']['horizon']
    test_size = cfg['split']['test_size']
    val_size = cfg['split']['val_size']
    feature_names = [c for c in df.columns if c not in exclude]
    feats = df[feature_names].values.astype('float32')
    close = df[target_name].values.astype('float64')
    n = len(df)
    if mode == 'log_return':
        target_raw = np.full(n, np.nan, dtype='float64')
        target_raw[horizon:] = np.log(close[horizon:] / close[:-horizon])
    elif mode == 'volatility':
        lr = np.full(n, np.nan, dtype='float64')
        lr[1:] = np.log(close[1:] / close[:-1])
        target_raw = np.full(n, np.nan, dtype='float64')
        for t in range(horizon, n):
            target_raw[t] = np.std(lr[t - horizon + 1:t + 1])
    else:
        target_raw = close.copy()
    n_test = int(n * test_size)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_size)
    n_train = n_trainval - n_val
    fsc = StandardScaler().fit(feats[:n_train])
    tt = target_raw[:n_train]
    tsc = StandardScaler().fit(tt[np.isfinite(tt)].reshape(-1, 1))
    feats_s = fsc.transform(feats).astype('float32')
    target_s = tsc.transform(target_raw.reshape(-1, 1)).ravel().astype('float32')
    (X, y, tidx) = _sequences_with_index(feats_s, target_s, lookback, horizon)
    train_m = tidx < n_train
    val_m = (tidx >= n_train) & (tidx < n_train + n_val)
    test_m = tidx >= n_train + n_val
    tpos = tidx[test_m]
    return Prepared(X[train_m], y[train_m], X[val_m], y[val_m], X[test_m], y[test_m], fsc, tsc, feature_names, target_name, df.index[tpos], mode, close[tpos - horizon].astype('float64'), close[tpos].astype('float64'), float(close[-1]), int(horizon))
