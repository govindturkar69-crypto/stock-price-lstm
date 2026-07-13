from __future__ import annotations
import numpy as np
from .evaluate import invert
from .model import torch_predict

def forecast_future(model, prep, steps: int=30) -> np.ndarray:
    window = prep.X_test[-1].copy()
    names = prep.feature_names
    fsc = prep.feature_scaler
    if prep.target_mode == 'log_return':
        lr_i = names.index('log_return') if 'log_return' in names else None
        close_i = names.index(prep.target_name) if prep.target_name in names else None
        price = prep.last_close
        h = max(1, getattr(prep, 'horizon', 1))
        prices = []
        for _ in range(steps):
            ret = float(invert(prep.target_scaler, torch_predict(model, window[np.newaxis, ...]))[0])
            ret /= h
            price *= float(np.exp(ret))
            prices.append(price)
            new_row = window[-1].copy()
            if lr_i is not None:
                new_row[lr_i] = (ret - fsc.mean_[lr_i]) / fsc.scale_[lr_i]
            if close_i is not None:
                new_row[close_i] = (price - fsc.mean_[close_i]) / fsc.scale_[close_i]
            window = np.vstack([window[1:], new_row])
        return np.asarray(prices)
    target_col = names.index(prep.target_name)
    preds_scaled = []
    for _ in range(steps):
        yhat = float(torch_predict(model, window[np.newaxis, ...])[0])
        preds_scaled.append(yhat)
        new_row = window[-1].copy()
        new_row[target_col] = yhat
        window = np.vstack([window[1:], new_row])
    return invert(prep.target_scaler, np.asarray(preds_scaled))
