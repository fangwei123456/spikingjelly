from spikingjelly.logger import logger
from collections import defaultdict
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.autograd.graph import register_multi_grad_hook
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.module_tracker import ModuleTracker

_arrow = chr(0x2937)


__all__ = [
    "ActiveModuleTracker",
    "BaseCounter",
    "is_binary_tensor",
    "DispatchCounterMode",
    "FunctionCounterMode",
]


_OPTIMIZER_HINTS = (
    "add_.",
    "sub_.",
    "mul_.",
    "div_.",
    "addcmul",
    "addcdiv",
    "lerp_",
    "copy_",
)


def _tensor_bits(x: Any) -> int:
    if not torch.is_tensor(x):
        return 0
    return int(x.numel() * x.element_size() * 8)


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def call_model(model: nn.Module, inputs: Any) -> Any:
    r"""
    **API Language** - :ref:`中文 <call_model-cn>` | :ref:`English <call_model-en>`

    ----

    .. _call_model-cn:

    * **中文**

    根据 ``inputs`` 的容器类型，以位置参数、关键字参数或单参数调用模型。

    :param model: 要调用的模型。
    :type model: torch.nn.Module
    :param inputs: 单个输入，或要展开的 tuple、list、dict。
    :type inputs: Any
    :return: 模型输出。
    :rtype: Any

    ----

    .. _call_model-en:

    * **English**

    Call ``model`` with one argument, expanded positional arguments, or expanded
    keyword arguments according to the container type of ``inputs``.

    :param model: Model to call.
    :type model: torch.nn.Module
    :param inputs: One input, or a tuple, list, or dict to expand.
    :type inputs: Any
    :return: Model output.
    :rtype: Any
    """
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


def _infer_stage(func, args, kwargs, out) -> str:
    op_name = resolve_name(func)
    if "backward" in op_name:
        return "backward"
    if torch.is_grad_enabled():
        return "forward"

    tensors = _collect_tensors((args, kwargs, out))
    if any(t.requires_grad for t in tensors) and any(
        hint in op_name for hint in _OPTIMIZER_HINTS
    ):
        return "optimizer"
    return "forward"


def is_binary_tensor(x: torch.Tensor) -> bool:
    r"""
    **API Language** - :ref:`中文 <is_binary_tensor-cn>` | :ref:`English <is_binary_tensor-en>`

    ----

    .. _is_binary_tensor-cn:

    * **中文**

    判断输入张量 ``x`` 是否为二元张量（即所有元素都在 {0, 1} 中或 ``dtype`` 为 ``bool``）。

    :param x: 输入张量
    :type x: torch.Tensor

    :return: 如果 ``x`` 是二元张量或 bool 张量则返回 ``True``
    :rtype: bool

    ----

    .. _is_binary_tensor-en:

    * **English**

    Check if the input tensor ``x`` is a binary tensor (all elements are in {0, 1} or its ``dtype`` is ``bool``).

    :param x: input tensor
    :type x: torch.Tensor

    :return: ``True`` if ``x`` is a binary tensor or a bool tensor
    :rtype: bool
    """
    if x.dtype == torch.bool:
        return True
    value = bool((x.eq(0) | x.eq(1)).all().item())
    return value


class ActiveModuleTracker(ModuleTracker):
    def __init__(self):
        r"""
        **API Language** - :ref:`中文 <ActiveModuleTracker.__init__-cn>` | :ref:`English <ActiveModuleTracker.__init__-en>`

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
                    "The module hierarchy tracking seems to be broken as this Module was already entered. {} during {}",
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
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. {} during {}",
                    name,
                    "backward" if is_bw else "forward",
                )

            if not self.active_modules:
                raise RuntimeError("active_modules stack underflow")

            if mod in self.active_modules:
                self.active_modules.remove(mod)
            else:
                logger.info(
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. {} during {}",
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
        **API Language** - :ref:`中文 <BaseCounter.__init__-cn>` | :ref:`English <BaseCounter.__init__-en>`

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
        **API Language** - :ref:`中文 <BaseCounter.has_rule-cn>` | :ref:`English <BaseCounter.has_rule-en>`

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

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules: Optional[set[nn.Module]] = None,
        parent_names: Optional[set[str]] = None,
    ) -> int:
        r"""
        **API Language** - :ref:`中文 <BaseCounter.count-cn>` | :ref:`English <BaseCounter.count-en>`

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

        :param active_modules: 当前处于活跃状态的模块集合。大多数计数器可忽略该参数，
            但需要结合模块上下文做语义统计的计数器可以使用它
        :type active_modules: Optional[set[nn.Module]]

        :param parent_names: 当前活跃模块名称集合。大多数计数器可忽略该参数
        :type parent_names: Optional[set[str]]

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

        :param active_modules: currently active module instances. Most counters can
            ignore it, while context-aware counters may use it for semantic counting
        :type active_modules: Optional[set[nn.Module]]

        :param parent_names: names of the currently active parent modules. Most
            counters can ignore it
        :type parent_names: Optional[set[str]]

        :return: the calculated count
        :rtype: int
        """
        return int(self.rules[func](args, kwargs, out))

    def record(self, scope, func, value):
        r"""
        **API Language** - :ref:`中文 <BaseCounter.record-cn>` | :ref:`English <BaseCounter.record-en>`

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
        **API Language** - :ref:`中文 <BaseCounter.get_counts-cn>` | :ref:`English <BaseCounter.get_counts-en>`

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
        **API Language** - :ref:`中文 <BaseCounter.get_total-cn>` | :ref:`English <BaseCounter.get_total-en>`

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

    def reset(self):
        r"""
        **API Language** - :ref:`中文 <BaseCounter.reset-cn>` | :ref:`English <BaseCounter.reset-en>`

        ----

        .. _BaseCounter.reset-cn:

        * **中文**

        重置计数器，清空所有已记录的计数。

        此方法会将 :attr:`records` 重新初始化为空的嵌套字典，移除之前累积的全部计数结果。
        适用于开始新的计数会话之前显式清零计数器状态。


        ----

        .. _BaseCounter.reset-en:

        * **English**

        Reset the counter and clear all recorded counts.

        This method reinitializes :attr:`records` to an empty nested dictionary,
        removing all previously accumulated count results. Call it before starting
        a new counting session when a counter instance is reused.
        """
        self.records = defaultdict(lambda: defaultdict(int))


class _CounterMode:
    def __init__(self, counters: list[BaseCounter], strict: bool = False):
        super().__init__()
        self.counters = counters
        self.strict = strict
        self.module_tracker = ActiveModuleTracker()

    def __enter__(self):
        self.module_tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        ret = super().__exit__(*args)
        self.module_tracker.__exit__(*args)
        return ret

    def _should_skip(self, counter, func) -> bool:
        if any(
            isinstance(module, tuple(counter.ignore_modules))
            for module in self.module_tracker.active_modules
        ):
            logger.debug("{} ignored by {}", _arrow, counter.__class__.__name__)
            return True
        if counter.has_rule(func):
            return False
        if self.strict:
            raise NotImplementedError(
                f"{type(self).__name__}: {self.module_tracker.parents} - "
                f"{resolve_name(func)} not defined by {counter.__class__.__name__}. "
                "To disable this error, set strict=False."
            )
        logger.debug("{} not defined by {}", _arrow, counter.__class__.__name__)
        return True

    def _record_counts(self, func, args, kwargs, out):
        active_modules = set(self.module_tracker.active_modules)
        parent_names = set(self.module_tracker.parents)
        for counter in self.counters:
            if self._should_skip(counter, func):
                continue
            value = counter.count(
                func,
                args,
                kwargs,
                out,
                active_modules=active_modules,
                parent_names=parent_names,
            )
            logger.debug("{} + {} [{}]", _arrow, value, counter.__class__.__name__)
            for parent in parent_names:
                counter.record(parent, func, value)
            if hasattr(counter, "finalize_record"):
                counter.finalize_record()


class DispatchCounterMode(_CounterMode, TorchDispatchMode):
    def __init__(self, counters: list[BaseCounter], strict: bool = False):
        r"""
        **API Language** - :ref:`中文 <DispatchCounterMode.__init__-cn>` | :ref:`English <DispatchCounterMode.__init__-en>`

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
            with DispatchCounterMode([flop_counter]):
                output = model(x)

            # Get and print results
            print("FLOP counts:", flop_counter.get_total())
        """
        super().__init__(counters, strict)

    def __torch_dispatch__(self, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        self._record_counts(func, args, kwargs, out)
        return out


class FunctionCounterMode(_CounterMode, TorchFunctionMode):
    def __init__(self, counters: list[BaseCounter], strict: bool = False):
        r"""
        **API Language** - :ref:`中文 <FunctionCounterMode.__init__-cn>` | :ref:`English <FunctionCounterMode.__init__-en>`

        ----

        .. _FunctionCounterMode.__init__-cn:

        * **中文**

        基于 PyTorch ``__torch_function__`` 机制的计数上下文。计数器必须为
        当前被拦截的 PyTorch 函数注册对应规则。

        :param counters: 计数器列表
        :type counters: list[BaseCounter]
        :param strict: 遇到未注册规则时是否抛出异常
        :type strict: bool

        ----

        .. _FunctionCounterMode.__init__-en:

        * **English**

        Counting context based on PyTorch's ``__torch_function__`` mechanism.
        Counters must register rules for the intercepted PyTorch functions.

        :param counters: Counter list
        :type counters: list[BaseCounter]
        :param strict: Whether to raise for an unregistered rule
        :type strict: bool
        """
        super().__init__(counters, strict)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        self._record_counts(func, args, kwargs, out)
        return out
