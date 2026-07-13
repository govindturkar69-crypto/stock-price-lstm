from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import nn

class LSTMRegressor(nn.Module):

    def __init__(self, n_features: int, cfg: dict[str, Any]):
        super().__init__()
        m = cfg['model']
        units = m['lstm_units']
        dropout = m.get('dropout', 0.2)
        dense_units = m.get('dense_units', [])
        bidir = m.get('bidirectional', False)
        dirs = 2 if bidir else 1
        self.lstms = nn.ModuleList()
        in_size = n_features
        for u in units:
            self.lstms.append(nn.LSTM(in_size, u, batch_first=True, bidirectional=bidir))
            in_size = u * dirs
        self.inter_dropout = nn.Dropout(dropout)
        head: list[nn.Module] = []
        h = in_size
        for du in dense_units:
            head += [nn.Linear(h, du), nn.ReLU(), nn.Dropout(dropout)]
            h = du
        head.append(nn.Linear(h, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        last_i = len(self.lstms) - 1
        for (i, lstm) in enumerate(self.lstms):
            (out, _) = lstm(out)
            if i < last_i:
                out = self.inter_dropout(out)
        return self.head(out[:, -1, :]).squeeze(-1)

def build_model(input_shape: tuple[int, int], cfg: dict[str, Any]) -> LSTMRegressor:
    (_, n_features) = input_shape
    return LSTMRegressor(n_features, cfg)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

@torch.no_grad()
def torch_predict(model: nn.Module, X: np.ndarray, device: torch.device | None=None, batch_size: int=256) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0, dtype='float32')
    device = device or next(model.parameters()).device
    model.eval()
    xb = torch.as_tensor(np.asarray(X, dtype='float32'))
    preds = []
    for i in range(0, len(xb), batch_size):
        chunk = xb[i:i + batch_size].to(device)
        preds.append(model(chunk).cpu().numpy().reshape(-1))
    return np.concatenate(preds)
