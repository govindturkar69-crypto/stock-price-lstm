from __future__ import annotations
import copy
import os
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import build_model, get_device

@dataclass
class History:
    history: dict[str, list[float]] = field(default_factory=lambda : {'loss': [], 'val_loss': []})

def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(np.asarray(X, dtype='float32')), torch.as_tensor(np.asarray(y, dtype='float32')))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def _eval_loss(model, loader, loss_fn, device):
    model.eval()
    (total, n) = (0.0, 0)
    with torch.no_grad():
        for (xb, yb) in loader:
            (xb, yb) = (xb.to(device), yb.to(device))
            total += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)

def train_model(prep, cfg: dict[str, Any], out_dir: str):
    t = cfg['train']
    m = cfg['model']
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    grad_clip = float(m.get('grad_clip', 0.0) or 0.0)
    model = build_model(prep.X_train.shape[1:], cfg).to(device)
    loss_fn = nn.HuberLoss() if m.get('loss') == 'huber' else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=m.get('learning_rate', 0.001), weight_decay=float(m.get('weight_decay', 0.0) or 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=t['reduce_lr_patience'], min_lr=1e-06)
    train_loader = _loader(prep.X_train, prep.y_train, t['batch_size'], t.get('shuffle', False))
    val_loader = _loader(prep.X_val, prep.y_val, t['batch_size'], False)
    history = History()
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_left = t['patience']
    for epoch in range(1, t['epochs'] + 1):
        model.train()
        (run, n) = (0.0, 0)
        for (xb, yb) in train_loader:
            (xb, yb) = (xb.to(device), yb.to(device))
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += loss.item() * len(xb)
            n += len(xb)
        train_loss = run / max(n, 1)
        val_loss = _eval_loss(model, val_loader, loss_fn, device)
        sched.step(val_loss)
        history.history['loss'].append(train_loss)
        history.history['val_loss'].append(val_loss)
        print(f'epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}')
        if val_loss < best_val - 1e-06:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = t['patience']
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'Early stopping at epoch {epoch} (best val {best_val:.5f}).')
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, 'best_model.pt'))
    return (model, history)
