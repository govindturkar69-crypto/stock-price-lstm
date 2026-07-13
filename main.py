from __future__ import annotations
import argparse
import json
import os
import pandas as pd
from src.config import load_config, set_seed
from src.data import load_prices
from src.features import add_features
from src.dataset import prepare
from src.train import train_model
from src.evaluate import regression_metrics, predict_prices, predict_returns, return_metrics, directional_metrics, naive_baseline, walk_forward
from src.predict import forecast_future
from src import plots

def parse_args():
    p = argparse.ArgumentParser(description='LSTM stock price prediction (PyTorch)')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--ticker', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--target-mode', choices=['log_return', 'price'], default=None)
    p.add_argument('--horizon', type=int, default=None, help='Forward return horizon in days')
    p.add_argument('--forecast-days', type=int, default=30)
    p.add_argument('--no-walk-forward', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic run for testing')
    return p.parse_args()

def apply_overrides(cfg, args):
    if args.ticker:
        cfg['data']['ticker'] = args.ticker
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.target_mode:
        cfg['data']['target_mode'] = args.target_mode
    if args.horizon:
        cfg['window']['horizon'] = args.horizon
    if args.no_walk_forward:
        cfg['walk_forward']['enabled'] = False
    if args.smoke:
        cfg['data']['synthetic_fallback'] = True
        cfg['data']['start'] = '2021-01-01'
        cfg['window']['lookback'] = 20
        cfg['model']['lstm_units'] = [16]
        cfg['model']['dense_units'] = []
        cfg['train']['epochs'] = 2
        cfg['train']['patience'] = 2
        cfg['walk_forward']['enabled'] = False
    return cfg

def _fold_metrics(df, cfg):
    prep = prepare(df, cfg)
    (model, _) = train_model(prep, cfg, out_dir=cfg['output']['dir'])
    (tr, pr) = predict_returns(model, prep)
    m = return_metrics(tr, pr)
    m['DirAcc_pct'] = directional_metrics(tr, pr)['DirAcc_pct']
    return m

def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg['output']['seed'])
    mode = cfg['data'].get('target_mode', 'log_return')
    print(f"[1/6] Loading data for {cfg['data']['ticker']} (target_mode={mode}) ...")
    (raw, source) = load_prices(cfg)
    print(f'      source={source} rows={len(raw)} range={raw.index.min().date()}..{raw.index.max().date()}')
    print('[2/6] Engineering features ...')
    feat = add_features(raw, cfg)
    print(f'      {feat.shape[1]} features, {len(feat)} usable rows')
    print('[3/6] Windowing & scaling ...')
    prep = prepare(feat, cfg)
    print(f'      train={prep.X_train.shape} val={prep.X_val.shape} test={prep.X_test.shape}')
    print('[4/6] Training LSTM ...')
    (model, history) = train_model(prep, cfg, out_dir)
    print('[5/6] Evaluating ...')
    (y_true, y_pred) = predict_prices(model, prep)
    price_m = regression_metrics(y_true, y_pred)
    baseline = naive_baseline(y_true)
    (true_ret, pred_ret) = predict_returns(model, prep)
    ret_m = return_metrics(true_ret, pred_ret)
    dir_m = directional_metrics(true_ret, pred_ret)
    print('  -- PRICE space (anchored; ~baseline for any model) --')
    print('     LSTM :', {k: round(v, 4) for (k, v) in price_m.items()})
    print('     Naive:', {k: round(v, 4) for (k, v) in baseline.items()})
    print('  -- RETURN space (the honest test of skill) --')
    print('     ', {k: round(v, 4) for (k, v) in ret_m.items()})
    print('  -- DIRECTION (up-day = positive class) --')
    print('     ', {k: round(v, 4) if isinstance(v, float) else v for (k, v) in dir_m.items() if k != 'confusion'})
    print('      confusion:', dir_m['confusion'])
    print(f"      (An IC>0 or ret_R2>0 or DirAcc>{dir_m['base_up_rate_pct']:.1f}% means real signal.)")
    report = {'source': source, 'target_mode': mode, 'price_space': {'lstm': price_m, 'naive_baseline': baseline}, 'return_space': ret_m, 'direction': dir_m}
    if cfg['walk_forward']['enabled']:
        print('      Walk-forward validation (return-space) ...')
        wf = walk_forward(feat, cfg, _fold_metrics)
        print(wf.round(4).to_string())
        report['walk_forward_return_space'] = wf.round(6).to_dict(orient='index')
    print(f'[6/6] Forecasting {args.forecast_days} days & plotting ...')
    future = forecast_future(model, prep, steps=args.forecast_days)
    future_idx = pd.bdate_range(prep.test_index[-1], periods=args.forecast_days + 1)[1:]
    plots.plot_history(history, out_dir)
    plots.plot_predictions(prep.test_index, y_true, y_pred, out_dir)
    plots.plot_forecast(prep.test_index, y_true, future_idx, future, out_dir)
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    pd.DataFrame({'date': future_idx, 'forecast': future}).to_csv(os.path.join(out_dir, 'forecast.csv'), index=False)
    print(f"\nDone. Artifacts written to '{out_dir}/':")
    for name in sorted(os.listdir(out_dir)):
        print('  -', name)
if __name__ == '__main__':
    main()
