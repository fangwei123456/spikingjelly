from __future__ import annotations

from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from ..base import ActiveModuleTracker
from .counter import SpikeSimEventCounter

__all__ = []


class SpikeSimEventTraceMode(TorchDispatchMode):
    def __init__(
        self,
        tracker: ActiveModuleTracker,
        *,
        counter: SpikeSimEventCounter,
    ):
        super().__init__()
        self.tracker = tracker
        self.counter = counter

    def _leaf_scope(self) -> str:
        names = [name for name in self.tracker.parents if name != "Global"]
        if not names:
            return "Global"
        return max(names, key=lambda name: (name.count("."), len(name)))

    def __torch_dispatch__(self, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        if self.counter.has_rule(func):
            counter_kwargs = dict(kwargs)
            counter_kwargs["__spikesim_scope__"] = self._leaf_scope()
            counter_kwargs["__spikesim_op_name__"] = resolve_name(func)
            _ = self.counter.count(func, args, counter_kwargs, out)
        return out
