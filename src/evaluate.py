from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .model import torch_predict

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100)
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    da = float(np.mean(true_dir == pred_dir) * 100) if len(true_dir) else float('nan')
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'MAPE_pct': mape, 'DirAcc_pct': da, 'R2': r2}

def invert(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()

def return_metrics(true_ret: np.ndarray, pred_ret: np.ndarray) -> dict[str, float]:
    t = np.asarray(true_ret).ravel()
    p = np.asarray(pred_ret).ravel()
    err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    ic = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 and p.std() > 0 else float('nan')
    return {'ret_RMSE': rmse, 'ret_MAE': mae, 'ret_R2': r2, 'IC_corr': ic}

def directional_metrics(true_ret: np.ndarray, pred_ret: np.ndarray) -> dict[str, float]:
    t = np.asarray(true_ret).ravel()
    p = np.asarray(pred_ret).ravel()
    true_up = t > 0
    pred_up = p > 0
    acc = float(np.mean(true_up == pred_up) * 100)
    tp = int(np.sum(pred_up & true_up))
    fp = int(np.sum(pred_up & ~true_up))
    fn = int(np.sum(~pred_up & true_up))
    tn = int(np.sum(~pred_up & ~true_up))
    precision = float(tp / (tp + fp)) if tp + fp else float('nan')
    recall = float(tp / (tp + fn)) if tp + fn else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (not np.isnan(precision)) and (not np.isnan(recall)) else float('nan')
    base_up_rate = float(np.mean(true_up) * 100)
    return {'DirAcc_pct': acc, 'up_precision': precision, 'up_recall': recall, 'up_F1': f1, 'base_up_rate_pct': base_up_rate, 'confusion': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}}

def predict_returns(model, prep) -> tuple[np.ndarray, np.ndarray]:
    pred_ret = invert(prep.target_scaler, torch_predict(model, prep.X_test))
    if prep.target_mode in ('log_return', 'volatility'):
        true_ret = invert(prep.target_scaler, prep.y_test)
    else:
        true_ret = np.log(prep.test_true_price / prep.test_anchor)
        pred_ret = np.log(np.maximum(pred_ret, 1e-09) / prep.test_anchor)
    return (true_ret, pred_ret)

def predict_prices(model, prep) -> tuple[np.ndarray, np.ndarray]:
    y_pred_t = invert(prep.target_scaler, torch_predict(model, prep.X_test))
    if prep.target_mode == 'log_return':
        y_pred = prep.test_anchor * np.exp(y_pred_t)
        y_true = prep.test_true_price
    else:
        y_pred = y_pred_t
        y_true = invert(prep.target_scaler, prep.y_test)
    return (y_true, y_pred)

def naive_baseline(y_true: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.concatenate([[y_true[0]], y_true[:-1]])
    return regression_metrics(y_true, y_pred)

def walk_forward(df, cfg: dict[str, Any], build_and_train) -> pd.DataFrame:
    n_splits = cfg['walk_forward']['n_splits']
    n = len(df)
    fold_size = n // (n_splits + 1)
    rows = []
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n)
        sub = df.iloc[:test_end].copy()
        local_cfg = {**cfg, 'split': {**cfg['split']}}
        local_cfg['split']['test_size'] = (test_end - train_end) / test_end
        local_cfg['split']['val_size'] = 0.1
        metrics = build_and_train(sub, local_cfg)
        metrics['fold'] = k
        rows.append(metrics)
    return pd.DataFrame(rows).set_index('fold')
