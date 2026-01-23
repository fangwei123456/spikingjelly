from collections import defaultdict
from typing import Any, Callable
import logging

import torch
import torch.nn as nn
from torch.autograd.graph import register_multi_grad_hook
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.module_tracker import ModuleTracker
from torch.utils._pytree import tree_flatten

logger = logging.getLogger(__name__)
_arrow = chr(0x2937)


class ActiveModuleTracker(ModuleTracker):

    def __init__(self):
        super().__init__()
        self.active_modules: set[nn.Module] = set() # align with self.parents: set[str]

    def _get_append_fn(self, mod, name, is_bw):
        def fn(*args) -> None:
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents:
                logger.info(
                    "The module hierarchy tracking seems to be broken as this Module was already entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )
            self.parents.add(name)
            self.active_modules.add(mod)

        return fn

    def _get_pop_fn(self, mod, name, is_bw):
        def fn(*args) -> None:
            if name in self.parents:
                self.parents.remove(name)
            else:
                logger.info(
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )

            if not self.active_modules:
                raise RuntimeError("active_modules stack underflow")

            if mod in self.active_modules:
                self.active_modules.remove(mod)
            else:
                logger.info(
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )

        return fn

    def _fw_pre_hook(self, mod, input) -> None:
        name = self._get_mod_name(mod)
        self._get_append_fn(mod, name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_pop_fn(mod, name, True))
            )

    def _fw_post_hook(self, mod, input, output) -> None:
        name = self._get_mod_name(mod)
        self._get_pop_fn(mod, name, False)()

        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_append_fn(mod, name, True))
            )


class BaseCounter:

    def __init__(self):
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {}
        self.ignore_modules = []

    def has_rule(self, func) -> bool:
        return func in self.rules

    def count(self, func, args, kwargs, out) -> int:
        return int(self.rules[func](args, kwargs, out))

    def record(self, parent, func, value):
        self.records[parent][func] += value

    def get_counts(self) -> dict[str, dict[Any, int]]:
        return {k: dict(v) for k, v in self.records.items()}

    def get_total(self) -> int:
        return sum(self.records["Global"].values())


class DispatchCounterMode(TorchDispatchMode):
    def __init__(self, counters: list[BaseCounter], strict: bool = False, verbose: bool = False):
        super().__init__()
        self.counters = counters
        self.strict = strict
        self.verbose = verbose
        self.module_tracker = ActiveModuleTracker()

    def __enter__(self):
        self.module_tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        ret = super().__exit__(*args)
        self.module_tracker.__exit__(*args)
        return ret

    def should_skip(self, counter, func) -> bool:
        parent_names = self.module_tracker.parents
        if not counter.has_rule(func): # stats rule not defined
            if self.strict:
                raise NotImplementedError(
                    f"DispatchCounterMode: {parent_names} - {resolve_name(func)}"
                    f" not defined by {counter.__class__.__name__}"
                )
            if self.verbose:
                print(
                    f"{_arrow} not defined by {counter.__class__.__name__}"
                )
            return True

        active_modules = self.module_tracker.active_modules
        for am in active_modules:
            if isinstance(am, tuple(counter.ignore_modules)): # inside a ignored module
                if self.verbose:
                    print(
                        f"{_arrow} ignored by {counter.__class__.__name__} as it is "
                        f"inside {am.__class__.__name__}"
                    )
                return True

        if self.verbose:
            print(
                f"{_arrow} counted by {counter.__class__.__name__}"
            )
        return False

    def __torch_dispatch__(self, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        parent_names = self.module_tracker.parents

        if self.verbose:
            print(f"DispatchCounterMode: {parent_names} - {resolve_name(func)}")

        for counter in self.counters:
            if self.should_skip(counter, func):
                continue
            value = counter.count(func, args, kwargs, out)
            for parent in set(parent_names):
                counter.record(parent, func, value) # add the count to every ancestor

        return out


class FunctionCounterMode(TorchFunctionMode):
    def __init__(self, counters: list[BaseCounter], strict: bool = False, verbose: bool = False):
        super().__init__()
        self.counters = counters
        self.strict = strict
        self.verbose = verbose
        self.module_tracker = ActiveModuleTracker()

    def __enter__(self):
        self.module_tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        ret = super().__exit__(*args)
        self.module_tracker.__exit__(*args)
        return ret

    def should_skip(self, counter, func) -> bool:
        parent_names = self.module_tracker.parents
        if not counter.has_rule(func): # stats rule not defined
            if self.strict:
                raise NotImplementedError(
                    f"FunctionCounterMode: {parent_names} - {resolve_name(func)} "
                    f"not defined by {counter.__class__.__name__}"
                )
            if self.verbose:
                print(
                    f"{_arrow} not defined by {counter.__class__.__name__}"
                )
            return True

        active_modules = self.module_tracker.active_modules
        for am in active_modules:
            if isinstance(am, tuple(counter.ignore_modules)): # inside a ignored module
                if self.verbose:
                    print(
                        f"{_arrow} ignored by {counter.__class__.__name__} as it is "
                        f"inside {am.__class__.__name__}"
                    )
                return True

        if self.verbose:
            print(
                f"{_arrow} counted by {counter.__class__.__name__}"
            )
        return False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        parent_names = self.module_tracker.parents

        if self.verbose:
            print(f"FunctionCounterMode: {parent_names} - {resolve_name(func)}")

        for counter in self.counters:
            if self.should_skip(counter, func):
                continue
            value = counter.count(func, args, kwargs, out)
            for parent in set(parent_names):
                counter.record(parent, func, value) # add the count to every ancestor

        return out