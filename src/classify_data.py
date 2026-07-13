from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .dataset import _sequences_with_index

@dataclass
class ClsPrepared:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    feature_names: list[str]
    test_index: pd.DatetimeIndex
    pos_weight: float
    train_up_rate: float
    test_up_rate: float

def prepare_classification(df: pd.DataFrame, cfg) -> ClsPrepared:
    target = cfg['data']['target']
    exclude = set(cfg['data'].get('exclude_features', []))
    lookback = cfg['window']['lookback']
    horizon = cfg['window']['horizon']
    test_size = cfg['split']['test_size']
    val_size = cfg['split']['val_size']
    feature_names = [c for c in df.columns if c not in exclude]
    feats = df[feature_names].values.astype('float32')
    close = df[target].values.astype('float64')
    n = len(df)
    label = np.full(n, np.nan, dtype='float32')
    label[horizon:] = (close[horizon:] > close[:-horizon]).astype('float32')
    n_test = int(n * test_size)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_size)
    n_train = n_trainval - n_val
    fsc = StandardScaler().fit(feats[:n_train])
    feats_s = fsc.transform(feats).astype('float32')
    (X, y, tidx) = _sequences_with_index(feats_s, label, lookback, horizon)
    train_m = tidx < n_train
    val_m = (tidx >= n_train) & (tidx < n_train + n_val)
    test_m = tidx >= n_train + n_val
    y_train = y[train_m]
    n_pos = float(np.sum(y_train == 1.0))
    n_neg = float(np.sum(y_train == 0.0))
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    return ClsPrepared(X[train_m], y_train, X[val_m], y[val_m], X[test_m], y[test_m], fsc, feature_names, df.index[tidx[test_m]], float(pos_weight), float(np.mean(y_train)), float(np.mean(y[test_m])))
