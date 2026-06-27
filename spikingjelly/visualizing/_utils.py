from __future__ import annotations

from typing import Union

import numpy as np
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x).__name__}")
