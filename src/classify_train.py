"""Train the direction classifier with a class-balanced BCE loss."""
from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model, get_device
from .train import History


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.as_tensor(np.asarray(X, dtype="float32")),
        torch.as_tensor(np.asarray(y, dtype="float32")),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _val_loss(model, loader, loss_fn, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tot += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    return tot / max(n, 1)


def train_classifier(prep, cfg: dict[str, Any], out_dir: str):
    t = cfg["train"]; m = cfg["model"]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    grad_clip = float(m.get("grad_clip", 0.0) or 0.0)

    model = build_model(prep.X_train.shape[1:], cfg).to(device)  # 1 logit output
    pos_weight = torch.tensor([prep.pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        model.parameters(), lr=m.get("learning_rate", 1e-3),
        weight_decay=float(m.get("weight_decay", 0.0) or 0.0),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=t["reduce_lr_patience"], min_lr=1e-6
    )
    tr = _loader(prep.X_train, prep.y_train, t["batch_size"], t.get("shuffle", False))
    va = _loader(prep.X_val, prep.y_val, t["batch_size"], False)

    history = History()
    best, best_state, left = float("inf"), copy.deepcopy(model.state_dict()), t["patience"]
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += loss.item() * len(xb); n += len(xb)
        trl = run / max(n, 1)
        vl = _val_loss(model, va, loss_fn, device)
        sched.step(vl)
        history.history["loss"].append(trl)
        history.history["val_loss"].append(vl)
        print(f"epoch {epoch:3d}  train {trl:.5f}  val {vl:.5f}")
        if vl < best - 1e-6:
            best, best_state, left = vl, copy.deepcopy(model.state_dict()), t["patience"]
        else:
            left -= 1
            if left <= 0:
                print(f"Early stopping at epoch {epoch} (best val {best:.5f}).")
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_classifier.pt"))
    return model, history


@torch.no_grad()
def predict_proba(model, X, device=None, batch_size=256):
    device = device or next(model.parameters()).device
    model.eval()
    xb = torch.as_tensor(np.asarray(X, dtype="float32"))
    out = []
    for i in range(0, len(xb), batch_size):
        logits = model(xb[i:i + batch_size].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.empty(0)
