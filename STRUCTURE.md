# Project Structure — stock-lstm-v3

```
stock-lstm-v3/
├── README.md              # Overview, v3 improvements, usage
├── STRUCTURE.md           # This file
├── requirements.txt       # torch-based dependencies
├── config.yaml            # Knobs: target_mode, features.stationary, exclude_features, reg
├── .gitignore
├── main.py                # Pipeline entry point (CLI)
└── src/
    ├── __init__.py
    ├── config.py          # Config loading + reproducible seeding
    ├── data.py            # yfinance download + synthetic fallback
    ├── features.py        # Stationary (default) or legacy feature engineering
    ├── dataset.py         # Return/price target, exclude_features, split, windowing
    ├── model.py           # Stacked LSTM (PyTorch) + torch_predict
    ├── train.py           # Training + early stop / LR sched / weight decay / grad clip
    ├── evaluate.py        # Metrics, baseline, price reconstruction, walk-forward
    ├── predict.py         # Recursive multi-step forecast (returns/price)
    └── plots.py           # Matplotlib charts

artifacts/                 # Created at runtime (git-ignored)
```
