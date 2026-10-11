from functools import partial

import torch

from ..native_loader import _load_native
from .validation import _forward_fake

_build_info = _load_native(__package__)
torch.library.register_fake(
    "sj_activation_aware_if::native_forward", partial(_forward_fake, _strided=True)
)
