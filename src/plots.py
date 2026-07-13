from __future__ import annotations
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_history(history, out_dir: str) -> str:
    path = os.path.join(out_dir, 'training_history.png')
    plt.figure(figsize=(9, 4))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training history')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path

def plot_predictions(index, y_true, y_pred, out_dir: str) -> str:
    path = os.path.join(out_dir, 'test_predictions.png')
    plt.figure(figsize=(11, 5))
    plt.plot(index, y_true, label='Actual', linewidth=1.5)
    plt.plot(index, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Test set: actual vs predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path

def plot_forecast(hist_index, hist_prices, future_index, future_prices, out_dir: str) -> str:
    path = os.path.join(out_dir, 'future_forecast.png')
    plt.figure(figsize=(11, 5))
    plt.plot(hist_index, hist_prices, label='History')
    plt.plot(future_index, future_prices, label='Forecast', linestyle='--', marker='o', ms=3)
    plt.axvline(hist_index[-1], color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Recursive multi-step forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
