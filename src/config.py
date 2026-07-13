from __future__ import annotations
import os
import random
from typing import Any
import numpy as np
import yaml

def load_config(path: str='config.yaml') -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int=42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
