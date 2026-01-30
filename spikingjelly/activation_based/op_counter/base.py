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


__all__ = [
    "ActiveModuleTracker",
    "BaseCounter",
    "DispatchCounterMode",
    "FunctionCounterMode",
]


class ActiveModuleTracker(ModuleTracker):
    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <ActiveModuleTracker.__init__-cn>` | :ref:`English <ActiveModuleTracker.__init__-en>`

        ----

        .. _ActiveModuleTracker.__init__-cn:

        * **中文**

        模块追踪器，用于在 PyTorch 的前向和反向传播过程中追踪模块的调用层次结构。
        它通过在模块的前向和反向钩子上进行回调来记录当前活跃的模块 :attr:`active_modules` 。

        :attr:`active_modules` 和 :attr:`parents` 的区别在于：前者是 ``nn.Module`` 的集合，
        后者是 ``str`` （模块名）的集合。

        ----

        .. _ActiveModuleTracker.__init__-en:

        * **English**

        Module tracker that tracks the module call hierarchy during PyTorch forward and backward passes.
        It records the currently executing module instances to :attr:`active_modules`
        through callbacks on module forward and backward hooks.

        Attributes :attr:`active_modules` and :attr:`parents` are different: the former
        is a set of ``nn.Module`` instances, while the latter is a set of ``str`` (module names).
        """
        super().__init__()
        self.active_modules: set[nn.Module] = set()  # align with self.parents: set[str]

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
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.__init__-cn>` | :ref:`English <BaseCounter.__init__-en>`

        ----

        .. _BaseCounter.__init__-cn:

        * **中文**

        操作计数器的基类。所有具体的计数器实现都继承自此类。

        该基类提供了计数器的核心属性：

        - :attr:`records`: 存储计数记录，结构为 ``dict[scope][operation] = count``
        - :attr:`rules`: 定义如何计算各个操作的计数的函数
        - :attr:`ignore_modules`: 需要忽略的模块列表，这些模块中的操作不会被计数

        子类需要实现具体的规则 ``rules`` 来定义如何计算特定操作的计数。

        ----

        .. _BaseCounter.__init__-en:

        * **English**

        Base class for operation counters.
        All concrete counter implementations inherit from this class.

        This base class provides core attributes for counters:

        - :attr:`records`: stores count records, structured as ``dict[scope][operation] = count``
        - :attr:`rules`: functions that define how to calculate counts for each operation
        - :attr:`ignore_modules`: list of modules to ignore. Operations within these modules will not be counted

        Subclasses need to implement specific rule functions in :attr:`rules` to define
        how to calculate counts for particular operations.
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {}
        self.ignore_modules: list[nn.Module] = []

    def has_rule(self, func) -> bool:
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.has_rule-cn>` | :ref:`English <BaseCounter.has_rule-en>`

        ----

        .. _BaseCounter.has_rule-cn:

        * **中文**

        :param func: 待判断的函数。其类型应与 :attr:`rules` 的键类型一致
        :type func: Any

        :return: ``func`` 是否有对应的计数规则
        :rtype: bool

        ----

        .. _BaseCounter.has_rule-en:

        * **English**

        :param func: the function or operation to be checked. Its type should be the
            same as the keys in :attr:`rules`
        :type func: Any

        :return: whether ``func`` has a corresponding counting rule
        :rtype: bool
        """
        return func in self.rules

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.count-cn>` | :ref:`English <BaseCounter.count-en>`

        ----

        .. _BaseCounter.count-cn:

        * **中文**

        根据 :attr:`rules` ，计算一次函数或操作调用所产生的计数值。

        :param func: 待计算的函数或操作。其类型应与 :attr:`rules` 的键类型一致
        :type func: Any

        :param args: `func` 的位置参数
        :type args: tuple

        :param kwargs: `func` 的关键字参数
        :type kwargs: dict

        :param out: `func` 输出
        :type out: Any

        :return: 计算得到的计数值
        :rtype: int

        ----

        .. _BaseCounter.count-en:

        * **English**

        Calculate the count for a function or operation call according to :attr:`rules`.

        :param func: the function or operation to be calculated. Its type should be the
            same as the keys in :attr:`rules`
        :type func: Any

        :param args: positional arguments of `func`
        :type args: tuple

        :param kwargs: keyword arguments of `func`
        :type kwargs: dict

        :param out: output of `func`
        :type out: Any

        :return: the calculated count
        :rtype: int
        """
        return int(self.rules[func](args, kwargs, out))

    def record(self, scope, func, value):
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.record-cn>` | :ref:`English <BaseCounter.record-en>`

        ----

        .. _BaseCounter.record-cn:

        * **中文**

        向 :attr:`records` 中添加记录。

        :param scope: 模块作用域字符串，如 ``"SimpleNet.lif1"``
        :type scope: str

        :param func: 待记录的函数或操作。其类型应与 :attr:`rules` 的键类型一致
        :type func: Any

        :param value: 计数值
        :type value: int

        ----

        .. _BaseCounter.record-en:

        * **English**

        Record the calculated count to :attr:`records`.

        :param scope: the module scope, e.g., ``"SimpleNet.lif1"``
        :type scope: str

        :param func: the function or operation to be recorded. Its type should be the
            same as the keys in :attr:`rules`
        :type func: Any

        :param value: the calculated count
        :type value: int
        """
        self.records[scope][func] += value

    def get_counts(self) -> dict[str, dict[Any, int]]:
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.get_counts-cn>` | :ref:`English <BaseCounter.get_counts-en>`

        ----

        .. _BaseCounter.get_counts-cn:

        * **中文**

        :return: 所有计数记录 :attr:`records`
        :rtype: dict[str, dict[Any, int]]

        ----

        .. _BaseCounter.get_counts-en:

        * **English**

        :return: all count records in :attr:`records`
        :rtype: dict[str, dict[Any, int]]
        """
        return {k: dict(v) for k, v in self.records.items()}

    def get_total(self) -> int:
        r"""
        **API Language:**
        :ref:`中文 <BaseCounter.get_total-cn>` | :ref:`English <BaseCounter.get_total-en>`

        ----

        .. _BaseCounter.get_total-cn:

        * **中文**

        :return: 顶层作用域 ``"Global"`` 下所有计数的总和。
        :rtype: int

        ----

        .. _BaseCounter.get_total-en:

        * **English**

        :return: the total count of all records in the ``"Global"`` scope.
        :rtype: int
        """
        return sum(self.records["Global"].values())


class DispatchCounterMode(TorchDispatchMode):
    def __init__(
        self, counters: list[BaseCounter], strict: bool = False, verbose: bool = False
    ):
        r"""
        **API Language:**
        :ref:`中文 <DispatchCounterMode.__init__-cn>` | :ref:`English <DispatchCounterMode.__init__-en>`

        ----

        .. _DispatchCounterMode.__init__-cn:

        * **中文**

        基于 PyTorch 的 Dispatch 机制的 **上下文管理器** ，用于计算aten操作对应计数。

        该类通过重写 ``__torch_dispatch__`` 方法来捕捉所有 PyTorch aten 操作的调用，并使用注册的计数器
        来统计这些操作的某些计数。

        **机制：**

        1. 通过 :class:`ActiveModuleTracker` 追踪当前执行所在的模块层级
        2. 对于每个被拦截的操作，检查是否有对应的计数规则
        3. 如果存在规则且不在被忽略的模块中，则调用规则函数计算计数值
        4. 将计数值记录到每一个父模块作用域中。

        :param counters: 计数器列表
        :type counters: list[BaseCounter]

        :param strict: 如果为 ``True`` ，当遇到未定义规则的操作时会报错；否则，未定义的操作将被跳过。
            默认为 ``False``
        :type strict: bool

        :param verbose: 如果为 ``True`` ，会在控制台打印每个被计数的操作及其计数值
        :type verbose: bool

        :return: 上下文管理器对象
        :rtype: DispatchCounterMode

        ----

        .. _DispatchCounterMode.__init__-en:

        * **English**

        **Context manager** based on PyTorch's Dispatch mechanism for counting aten operations.
        It intercepts all PyTorch aten operations through overriding `__torch_dispatch__`
        and uses registered counters to track these operations.

        **Working Mechanism:**

        1. Tracks the current module hierarchy using :class:`ActiveModuleTracker`
        2. For each intercepted operation, checks if there's a corresponding counting rule
        3. If a rule exists and the operation is not in an ignored module, calls the rule function to calculate the count
        4. Records the count to the parent module scope

        :param counters: list of counters
        :type counters: list[BaseCounter]

        :param strict: if ``True``, raises ``NotImplementedError`` when encountering
            operations without defined rules; if ``False``, skip the operations without
            defined rules. Default to ``False``.
        :type strict: bool

        :param verbose: if ``True``, prints each counted operation and its count to the console
        :type verbose: bool

        :return: Context manager object
        :rtype: DispatchCounterMode

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                FlopCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn


            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(100, 50)

                def forward(self, x):
                    return self.linear(x)


            model = SimpleNet()
            x = torch.randn(32, 100)

            # Initialize counter
            flop_counter = FlopCounter()
            with DispatchCounterMode([flop_counter], verbose=True):
                output = model(x)

            # Get and print results
            print("FLOP counts:", flop_counter.get_total())
        """
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

    def _should_skip(self, counter, func) -> bool:
        parent_names = self.module_tracker.parents
        if not counter.has_rule(func):  # stats rule not defined
            if self.strict:
                raise NotImplementedError(
                    f"DispatchCounterMode: {parent_names} - {resolve_name(func)}"
                    f" not defined by {counter.__class__.__name__}"
                )
            if self.verbose:
                print(f"{_arrow} not defined by {counter.__class__.__name__}")
            return True

        active_modules = self.module_tracker.active_modules
        for am in active_modules:
            if isinstance(am, tuple(counter.ignore_modules)):  # inside a ignored module
                if self.verbose:
                    print(
                        f"{_arrow} ignored by {counter.__class__.__name__} as it is "
                        f"inside {am.__class__.__name__}"
                    )
                return True

        return False

    def __torch_dispatch__(self, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        parent_names = self.module_tracker.parents

        if self.verbose:
            print(f"DispatchCounterMode: {parent_names} - {resolve_name(func)}")

        for counter in self.counters:
            if self._should_skip(counter, func):
                continue
            value = counter.count(func, args, kwargs, out)
            if self.verbose:
                print(f"{_arrow} + {value} [{counter.__class__.__name__}]")
            for parent in set(parent_names):
                counter.record(parent, func, value)  # add the count to every ancestor

        return out


class FunctionCounterMode(TorchFunctionMode):
    def __init__(
        self, counters: list[BaseCounter], strict: bool = False, verbose: bool = False
    ):
        r"""
        **API Language:**
        :ref:`中文 <FunctionCounterMode.__init__-cn>` | :ref:`English <FunctionCounterMode.__init__-en>`

        ----

        .. _FunctionCounterMode.__init__-cn:

        * **中文**

        基于 PyTorch Function 机制的 **上下文管理器** ，用于计算函数的计数。

        该类通过重写 ``__torch_function__`` 方法来拦截所有 PyTorch 函数调用，并使用注册的计数器来统计这些
        操作的某些计数。

        工作原理与 :class:`DispatchCounterMode` 类似。

        :param counters: 计数器列表
        :type counters: list[BaseCounter]

        :param strict: 如果为 ``True``，当遇到未定义规则的操作时会报错；否则，未定义的操作将被跳过。
            默认为 ``False``
        :type strict: bool

        :param verbose: 如果为 ``True``，会在控制台打印每个被计数的操作及其计数值
        :type verbose: bool

        :return: 上下文管理器对象
        :rtype: FunctionCounterMode

        ----

        .. _FunctionCounterMode.__init__-en:

        * **English**

        **Context manager** based on PyTorch's Function mechanism for counting operations.
        It intercepts all PyTorch function calls through overriding ``__torch_function__`` and
        uses registered counters to track these operations.

        It has a similar working mechanism to :class:`DispatchCounterMode` .

        :param counters: list of counters
        :type counters: list[BaseCounter]

        :param strict: if ``True``, raises ``NotImplementedError`` when encountering operations without defined rules;
            if ``False``, skips operations without defined rules. Default to ``False``
        :type strict: bool

        :param verbose: if ``True``, prints each counted operation and its count to the console
        :type verbose: bool

        :return: Context manager object
        :rtype: FunctionCounterMode
        """
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

    def _should_skip(self, counter, func) -> bool:
        parent_names = self.module_tracker.parents
        if not counter.has_rule(func):  # stats rule not defined
            if self.strict:
                raise NotImplementedError(
                    f"FunctionCounterMode: {parent_names} - {resolve_name(func)} "
                    f"not defined by {counter.__class__.__name__}"
                )
            if self.verbose:
                print(f"{_arrow} not defined by {counter.__class__.__name__}")
            return True

        active_modules = self.module_tracker.active_modules
        for am in active_modules:
            if isinstance(am, tuple(counter.ignore_modules)):  # inside a ignored module
                if self.verbose:
                    print(
                        f"{_arrow} ignored by {counter.__class__.__name__} as it is "
                        f"inside {am.__class__.__name__}"
                    )
                return True

        if self.verbose:
            print(f"{_arrow} counted by {counter.__class__.__name__}")
        return False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        parent_names = self.module_tracker.parents

        if self.verbose:
            print(f"FunctionCounterMode: {parent_names} - {resolve_name(func)}")

        for counter in self.counters:
            if self._should_skip(counter, func):
                continue
            value = counter.count(func, args, kwargs, out)
            if self.verbose:
                print(f"{_arrow} + {value}")
            for parent in set(parent_names):
                counter.record(parent, func, value)  # add the count to every ancestor

        return out
