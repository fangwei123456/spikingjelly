import torch

from ..native_loader import _load_native
from .validation import _forward_fake

_build_info = _load_native(__package__)
torch.library.register_fake("sj_stbif::native_forward", _forward_fake)
