from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from src.config import load_config, set_seed
from src.data import load_prices
from src.features import add_features
from src.classify_data import prepare_classification
from src.classify_train import train_classifier, predict_proba
from src import plots


def parse_args():
    p = argparse.ArgumentParser(description='Direction classifier (PyTorch, class-balanced)')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--ticker', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--no-walk-forward', action='store_true')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def overrides(cfg, a):
    if a.ticker:
        cfg['data']['ticker'] = a.ticker
    if a.epochs:
        cfg['train']['epochs'] = a.epochs
    if a.horizon:
        cfg['window']['horizon'] = a.horizon
    if a.no_walk_forward:
        cfg['walk_forward']['enabled'] = False
    if a.smoke:
        cfg['data']['start'] = '2021-01-01'
        cfg['data']['synthetic_fallback'] = True
        cfg['window']['lookback'] = 20
        cfg['model']['lstm_units'] = [16]
        cfg['model']['dense_units'] = []
        cfg['train']['epochs'] = 2
        cfg['train']['patience'] = 2
        cfg['walk_forward']['enabled'] = False
    return cfg


def cls_metrics(y_true, proba, thr=0.5):
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    acc = (tp + tn) / max(len(y), 1)
    up_rec = tp / (tp + fn) if tp + fn else float('nan')
    dn_rec = tn / (tn + fp) if tn + fp else float('nan')
    bal_acc = np.nanmean([up_rec, dn_rec])
    up_prec = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * up_prec * up_rec / (up_prec + up_rec) if up_prec and up_rec and (not np.isnan(up_prec)) and (not np.isnan(up_rec)) else float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    return {'accuracy': float(acc), 'balanced_acc': float(bal_acc), 'roc_auc': auc,
            'up_precision': float(up_prec), 'up_recall': float(up_rec),
            'down_recall': float(dn_rec), 'up_F1': float(f1),
            'base_up_rate': float(np.mean(y)),
            'confusion': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}}


def _walk_forward(feat, cfg, thr):
    n = len(feat)
    n_splits = cfg['walk_forward']['n_splits']
    fold_size = n // (n_splits + 1)
    rows = []
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n)
        sub = feat.iloc[:test_end].copy()
        lc = {**cfg, 'split': {**cfg['split']}}
        lc['split']['test_size'] = (test_end - train_end) / test_end
        lc['split']['val_size'] = 0.1
        prep = prepare_classification(sub, lc)
        model, _ = train_classifier(prep, lc, cfg['output']['dir'])
        m = cls_metrics(prep.y_test, predict_proba(model, prep.X_test), thr)
        rows.append({'fold': k, 'balanced_acc': m['balanced_acc'], 'roc_auc': m['roc_auc'],
                     'up_recall': m['up_recall'], 'down_recall': m['down_recall']})
    return pd.DataFrame(rows).set_index('fold')


def main():
    a = parse_args()
    cfg = overrides(load_config(a.config), a)
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg['output']['seed'])
    h = cfg['window']['horizon']
    print(f"[1/4] Loading {cfg['data']['ticker']} (horizon={h}) ...")
    raw, source = load_prices(cfg)
    feat = add_features(raw, cfg)
    print(f'      {feat.shape[1]} features, {len(feat)} rows')
    print('[2/4] Windowing (balanced labels) ...')
    prep = prepare_classification(feat, cfg)
    print(f'      train up-rate={prep.train_up_rate:.3f}  pos_weight={prep.pos_weight:.3f}  test up-rate={prep.test_up_rate:.3f}')
    print('[3/4] Training classifier ...')
    model, history = train_classifier(prep, cfg, out_dir)
    print('[4/4] Evaluating ...')
    proba = predict_proba(model, prep.X_test)
    m = cls_metrics(prep.y_test, proba, a.threshold)
    print('  DIRECTION classifier (up = positive):')
    for k, v in m.items():
        if k != 'confusion':
            print(f'     {k:14s}: {v:.4f}' if isinstance(v, float) else f'     {k}: {v}')
    print('     confusion:', m['confusion'])
    print('  (roc_auc>0.5 and balanced_acc>0.5 with BOTH recalls>0 = real directional skill.)')
    report = {'source': source, 'horizon': h, 'test': m}
    if cfg['walk_forward']['enabled']:
        print('  Walk-forward (balanced_acc / auc) ...')
        wf = _walk_forward(feat, cfg, a.threshold)
        print(wf.round(4).to_string())
        report['walk_forward'] = wf.round(6).to_dict(orient='index')
    plots.plot_history(history, out_dir)
    with open(os.path.join(out_dir, 'classify_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f'\nDone. Wrote {out_dir}/classify_report.json and best_classifier.pt')


if __name__ == '__main__':
    main()
