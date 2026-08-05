from spikingjelly.logger import logger
import copy
from typing import Tuple, Generator, Optional, Callable, Any

import torch
import torch.nn as nn

try:
    import cupy
except (ImportError, OSError) as e:
    logger.info("Optional CuPy backend unavailable: %s", e)
    _CUPY_IMPORT_ERROR = e
    cupy = None
else:
    _CUPY_IMPORT_ERROR = None

try:
    import triton
except (ImportError, OSError) as e:
    logger.info("Optional Triton backend unavailable: %s", e)
    _TRITON_IMPORT_ERROR = e
    triton = None
else:
    _TRITON_IMPORT_ERROR = None

try:
    import lava.lib.dl.slayer as slayer
except (ImportError, OSError) as e:
    _LAVA_IMPORT_ERROR = e
    slayer = None
else:
    _LAVA_IMPORT_ERROR = None


def check_backend_library(backend: str):
    r"""
    **API Language** - :ref:`中文 <check_backend_library-cn>` | :ref:`English <check_backend_library-en>`

    ----

    .. _check_backend_library-cn:

    * **中文**

    检查某个后端的python库是否已经安装。若未安装则此函数会报 ``ImportError`` 。

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'`` 或 ``'lava'``
    :type backend: str

    :raises ImportError: 若所请求后端依赖的 Python 库未安装，则抛出 ``ImportError``

    ----

    .. _check_backend_library-en:

    * **English**

    Check whether the python lib for backend is installed.
    If not, this function will raise an ``ImportError`` .

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'`` or ``'lava'``
    :type backend: str

    :raises ImportError: Raised when the Python package required by ``backend`` is not installed
    """
    if backend == "torch":
        return
    elif backend == "cupy":
        if cupy is None:
            error = ImportError(
                "CuPy is not installed! "
                'You can install it from "https://github.com/cupy/cupy".'
            )
            if _CUPY_IMPORT_ERROR is not None:
                raise error from _CUPY_IMPORT_ERROR
            raise error
    elif backend == "triton":
        if triton is None:
            error = ImportError(
                "Triton is not installed! "
                'You can install it from "https://github.com/openai/triton".'
            )
            if _TRITON_IMPORT_ERROR is not None:
                raise error from _TRITON_IMPORT_ERROR
            raise error
    elif backend == "lava":
        if slayer is None:
            error = ImportError(
                "Lava-DL is not installed! You can install it from "
                '"https://github.com/lava-nc/lava-dl". '
            )
            if _LAVA_IMPORT_ERROR is not None:
                raise error from _LAVA_IMPORT_ERROR
            raise error
    else:
        pass


class StepModule:
    r"""
    **API Language** - :ref:`中文 <StepModule-cn>` | :ref:`English <StepModule-en>`

    ----

    .. _StepModule-cn:

    * **中文**

    步进模式接口基类。

    实现该接口的模块通过 ``step_mode`` 区分单步模式 ``"s"`` 与多步模式 ``"m"``。

    ----

    .. _StepModule-en:

    * **English**

    Base interface for step-mode aware modules.

    Modules implementing this interface distinguish single-step mode ``"s"``
    from multi-step mode ``"m"`` through ``step_mode``.
    """

    def supported_step_mode(self) -> Tuple[str]:
        r"""
        **API Language** - :ref:`中文 <StepModule.supported_step_mode-cn>` | :ref:`English <StepModule.supported_step_mode-en>`

        ----

        .. _StepModule.supported_step_mode-cn:

        * **中文**

        :return: 包含支持的步进模式的tuple。``"s"`` 代表单步模式， ``"m"`` 代表多步模式。
        :rtype: Tuple[str]

        ----

        .. _StepModule.supported_step_mode-en:

        * **English**

        :return: a tuple that contains the supported step mode(s).  ``"s"`` is for
            single-step mode, and ``"m"`` is for multi-step mode.
        :rtype: Tuple[str]
        """
        return ("s", "m")

    @property
    def step_mode(self) -> str:
        r"""
        **API Language** - :ref:`中文 <StepModule.step_mode-cn>` | :ref:`English <StepModule.step_mode-en>`

        ----

        .. _StepModule.step_mode-cn:

        * **中文**

        :return: 模块当前使用的步进模式
        :rtype: str

        ----

        .. _StepModule.step_mode-en:

        * **English**

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        r"""
        **API Language** - :ref:`中文 <StepModule.step_mode-setter-cn>` | :ref:`English <StepModule.step_mode-setter-en>`

        ----

        .. _StepModule.step_mode-setter-cn:

        * **中文**

        将本模块的步进模式设置为 ``value`` 。

        :param value: 步进模式
        :type value: str

        :raises ValueError: 当 ``value`` 不在 ``self.supported_step_mode()`` 中时抛出

        ----

        .. _StepModule.step_mode-setter-en:

        * **English**

        Set the step mode of this module to be ``value`` .

        :param value: the step mode
        :type value: str

        :raises ValueError: Raised when ``value`` is not included in ``self.supported_step_mode()``
        """
        if value not in self.supported_step_mode():
            raise ValueError(
                f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!'
            )
        self._step_mode = value


class SingleStepModule(StepModule):
    r"""
    **API Language** - :ref:`中文 <SingleStepModule-cn>` | :ref:`English <SingleStepModule-en>`

    ----

    .. _SingleStepModule-cn:

    * **中文**

    单步模式模块的接口基类。

    实现该接口的模块仅支持单步模式 ``"s"``。

    ----

    .. _SingleStepModule-en:

    * **English**

    Base interface for single-step mode modules.

    Modules implementing this interface only support single-step mode ``"s"``.
    """

    def supported_step_mode(self):
        r"""
        **API Language** - :ref:`中文 <SingleStepModule.supported_step_mode-cn>` | :ref:`English <SingleStepModule.supported_step_mode-en>`

        ----

        .. _SingleStepModule.supported_step_mode-cn:

        * **中文**

        :return: 仅包含 ``"s"`` 的 tuple
        :rtype: Tuple[str]

        ----

        .. _SingleStepModule.supported_step_mode-en:

        * **English**

        :return: A tuple containing only ``"s"``
        :rtype: Tuple[str]
        """
        return ("s",)


class MultiStepModule(StepModule):
    r"""
    **API Language** - :ref:`中文 <MultiStepModule-cn>` | :ref:`English <MultiStepModule-en>`

    ----

    .. _MultiStepModule-cn:

    * **中文**

    多步模式模块的接口基类。

    实现该接口的模块仅支持多步模式 ``"m"``。

    ----

    .. _MultiStepModule-en:

    * **English**

    Base interface for multi-step mode modules.

    Modules implementing this interface only support multi-step mode ``"m"``.
    """

    def supported_step_mode(self):
        r"""
        **API Language** - :ref:`中文 <MultiStepModule.supported_step_mode-cn>` | :ref:`English <MultiStepModule.supported_step_mode-en>`

        ----

        .. _MultiStepModule.supported_step_mode-cn:

        * **中文**

        :return: 仅包含 ``"m"`` 的 tuple
        :rtype: Tuple[str]

        ----

        .. _MultiStepModule.supported_step_mode-en:

        * **English**

        :return: A tuple containing only ``"m"``
        :rtype: Tuple[str]
        """
        return ("m",)


class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.__init__-cn>` | :ref:`English <MemoryModule.__init__-en>`

        ----

        .. _MemoryModule.__init__-cn:

        * **中文**

        SpikingJelly 中所有有状态模块的基类。

        ``MemoryModule`` 通过 ``register_memory`` 注册内部状态变量，并提供
        ``reset``、``detach``、显式 memory 提取与恢复等通用能力。

        ----

        .. _MemoryModule.__init__-en:

        * **English**

        Base class of all stateful modules in SpikingJelly.

        ``MemoryModule`` registers internal state variables via
        ``register_memory`` and provides common utilities such as ``reset``,
        ``detach``, and explicit memory extraction / restoration.
        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = "torch"
        self._step_mode = "s"

    @property
    def supported_backends(self) -> Tuple[str]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.supported_backends-cn>` | :ref:`English <MemoryModule.supported_backends-en>`

        ----

        .. _MemoryModule.supported_backends-cn:

        * **中文**

        :return: 支持的后端
        :rtype: Tuple[str]

        ----

        .. _MemoryModule.supported_backends-en:

        * **English**

        :return: supported backends
        :rtype: Tuple[str]
        """
        return ("torch",)

    @property
    def backend(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.backend-cn>` | :ref:`English <MemoryModule.backend-en>`

        ----

        .. _MemoryModule.backend-cn:

        * **中文**

        :return: 当前后端名称
        :rtype: str

        ----

        .. _MemoryModule.backend-en:

        * **English**

        :return: the name of the current backend
        :rtype: str
        """
        return self._backend

    @backend.setter
    def backend(self, value: str):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.backend-setter-cn>` | :ref:`English <MemoryModule.backend-setter-en>`

        ----

        .. _MemoryModule.backend-setter-cn:

        * **中文**

        设置当前模块的后端。

        只有当 ``value`` 属于 :meth:`supported_backends` 且对应依赖库已安装时，
        赋值才会成功。

        :param value: 目标后端名称
        :type value: str
        :raises NotImplementedError: 当 ``value`` 不在 ``supported_backends`` 中时抛出
        :raises ImportError: 当 ``value`` 对应的后端库未安装时抛出

        ----

        .. _MemoryModule.backend-setter-en:

        * **English**

        Set the backend of the current module.

        The assignment succeeds only when ``value`` is listed in
        :meth:`supported_backends` and the corresponding backend library is
        installed.

        :param value: Target backend name
        :type value: str
        :raises NotImplementedError: Raised when ``value`` is not listed in ``supported_backends``
        :raises ImportError: Raised when the backend library required by ``value`` is not installed
        """
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        check_backend_library(value)
        self._backend = value

    def materialize_states(
        self,
        inputs: tuple[Any, ...],
        states: tuple[Any, ...],
        step_mode: str,
    ) -> tuple[Any, ...]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.materialize_states-cn>` | :ref:`English <MemoryModule.materialize_states-en>`

        ----

        .. _MemoryModule.materialize_states-cn:

        * **中文**

        默认实现原样返回 ``states``。需要将标量或空状态转换为输入相关张量的
        模块可以重写本方法。``inputs`` 是当前前向传播的完整输入；实现根据
        ``step_mode`` 选择合适的形状和设备参照。实现不得修改模块 memory 或
        传入的 ``states``。

        :param inputs: 当前前向传播的完整输入元组
        :type inputs: tuple[Any, ...]
        :param states: 待物化的显式状态
        :type states: tuple[Any, ...]
        :param step_mode: 当前前向传播的步进模式，取 ``"s"`` 或 ``"m"``
        :type step_mode: str
        :return: 可传入 functional forward 的状态
        :rtype: tuple[Any, ...]

        ----

        .. _MemoryModule.materialize_states-en:

        * **English**

        The default implementation returns ``states`` unchanged. Modules that
        need to turn scalar or empty states into input-dependent tensors may
        override this method. ``inputs`` contains the complete inputs of the
        current forward pass; implementations select an appropriate shape and
        device reference according to ``step_mode``. Implementations must not
        mutate module memories or the supplied ``states``.

        :param inputs: Complete input tuple of the current forward pass
        :type inputs: tuple[Any, ...]
        :param states: Explicit states to materialize
        :type states: tuple[Any, ...]
        :param step_mode: Step mode of the current forward pass, ``"s"`` or ``"m"``
        :type step_mode: str
        :return: States ready for functional forward
        :rtype: tuple[Any, ...]
        """
        return states

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[Any, ...],
        **kwargs: Any,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.single_step_functional_forward-cn>` | :ref:`English <MemoryModule.single_step_functional_forward-en>`

        ----

        .. _MemoryModule.single_step_functional_forward-cn:

        * **中文**

        使用显式状态执行单步前向传播。实现不得修改本模块注册的 memory
        或传入的 ``states``。``states`` 的顺序与当前模块的
        ``self._memories`` 一致，不包含子模块状态。复合模块的完整状态由
        :func:`to_functional_forward` 负责展开。

        :param inputs: 单步输入张量元组
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 当前模块的显式状态元组
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]
        :raises NotImplementedError: 当子类未实现单步 functional forward 时抛出

        ----

        .. _MemoryModule.single_step_functional_forward-en:

        * **English**

        Run a single-step forward pass with explicit state. The implementation
        must not mutate this module's registered memories or the supplied
        ``states``. State order follows ``self._memories`` and excludes states
        owned by child modules. Use :func:`to_functional_forward` to flatten the
        complete state of a composite module.

        :param inputs: Tuple of single-step input tensors
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Tuple of explicit states owned by this module
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]
        :raises NotImplementedError: If the subclass does not implement single-step functional forward
        """
        raise NotImplementedError(
            f"{self._get_name()} does not implement single_step_functional_forward."
        )

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[Any, ...],
        **kwargs: Any,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.multi_step_functional_forward-cn>` | :ref:`English <MemoryModule.multi_step_functional_forward-en>`

        ----

        .. _MemoryModule.multi_step_functional_forward-cn:

        * **中文**

        使用显式状态执行多步前向传播。输入张量的第 0 维为时间维。
        默认实现逐时间步调用 :meth:`single_step_functional_forward`；具有
        专用多步实现或 kernel 的子类可以重写本方法。实现不得修改本模块
        注册的 memory 或传入的 ``states``。

        :param inputs: 多步输入张量元组
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 当前模块的显式状态元组
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]

        ----

        .. _MemoryModule.multi_step_functional_forward-en:

        * **English**

        Run a multi-step forward pass with explicit state. Dimension 0 of each
        input tensor is the time dimension. The default implementation calls
        :meth:`single_step_functional_forward` at each time step. Subclasses with
        a specialized sequence implementation or kernel may override this method.
        The implementation must not mutate this module's registered memories or
        the supplied ``states``.

        :param inputs: Tuple of multi-step input tensors
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Tuple of explicit states owned by this module
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]
        """
        output_steps = []
        for t in range(inputs[0].shape[0]):
            outputs, states = self.single_step_functional_forward(
                tuple(x[t] for x in inputs), states, **kwargs
            )
            output_steps.append(outputs)

        return (
            tuple(
                torch.stack(output_seq)
                for output_seq in zip(*output_steps, strict=True)
            ),
            states,
        )

    def functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[Any, ...],
        **kwargs: Any,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.functional_forward-cn>` | :ref:`English <MemoryModule.functional_forward-en>`

        ----

        .. _MemoryModule.functional_forward-cn:

        * **中文**

        先调用 :meth:`materialize_states`，再按当前 ``step_mode`` 调用单步或
        多步 functional forward。

        :param inputs: 输入张量元组
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 当前模块的显式状态元组
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]
        :raises ValueError: 当 ``step_mode`` 不是 ``"s"`` 或 ``"m"`` 时抛出

        ----

        .. _MemoryModule.functional_forward-en:

        * **English**

        Call :meth:`materialize_states`, then dispatch to single-step or
        multi-step functional forward according to the current ``step_mode``.

        :param inputs: Tuple of input tensors
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Tuple of explicit states owned by this module
        :type states: tuple[Any, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]
        :raises ValueError: If ``step_mode`` is neither ``"s"`` nor ``"m"``
        """
        if self.step_mode == "s":
            states = self.materialize_states(inputs, states, "s")
            return self.single_step_functional_forward(inputs, states, **kwargs)
        if self.step_mode == "m":
            states = self.materialize_states(inputs, states, "m")
            return self.multi_step_functional_forward(inputs, states, **kwargs)
        raise ValueError(self.step_mode)

    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.single_step_forward-cn>` | :ref:`English <MemoryModule.single_step_forward-en>`

        ----

        .. _MemoryModule.single_step_forward-cn:

        * **中文**

        基于 :meth:`single_step_functional_forward` 执行单步前向传播：读取
        ``self._memories``，执行显式状态转移，再写回更新后的状态。

        :param x: 输入张量，约定 ``shape = [N, *]``，其中 ``N`` 通常为 batch 维
        :type x: torch.Tensor

        :return: 单步前向传播的输出
        :rtype: Any
        :raises ValueError: 更新状态数量与已注册 memory 数量不一致时抛出

        ----

        .. _MemoryModule.single_step_forward-en:

        * **English**

        Run a single-step forward pass through
        :meth:`single_step_functional_forward`: read ``self._memories``, execute
        the explicit state transition, and write the updated states back.

        :param x: Input tensor, conventionally with ``shape = [N, *]`` where ``N`` is usually the batch dimension
        :type x: torch.Tensor

        :return: Output of the single-step forward pass
        :rtype: Any
        :raises ValueError: If the number of updated states does not match the registered memories
        """
        inputs = (x, *args)
        states = self.materialize_states(inputs, tuple(self._memories.values()), "s")
        for name, value in zip(self._memories, states, strict=True):
            self._memories[name] = value
        outputs, updated_states = self.single_step_functional_forward(
            inputs, states, **kwargs
        )
        for name, value in zip(self._memories.keys(), updated_states, strict=True):
            self._memories[name] = value
        return outputs[0] if len(outputs) == 1 else outputs

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.multi_step_forward-cn>` | :ref:`English <MemoryModule.multi_step_forward-en>`

        ----

        .. _MemoryModule.multi_step_forward-cn:

        * **中文**

        基于 :meth:`multi_step_functional_forward` 执行多步前向传播：读取
        ``self._memories``，执行显式状态转移，再写回更新后的状态。

        :param x_seq: 输入序列张量，约定 ``shape = [T, N, *]``，其中第 0 维为时间维
        :type x_seq: torch.Tensor

        :return: 按时间堆叠的输出序列
        :rtype: torch.Tensor

        :raises ValueError: 更新状态数量与已注册 memory 数量不一致时抛出

        ----

        .. _MemoryModule.multi_step_forward-en:

        * **English**

        Run a multi-step forward pass through
        :meth:`multi_step_functional_forward`: read ``self._memories``, execute
        the explicit state transition, and write the updated states back.

        :param x_seq: Input sequence tensor, conventionally with ``shape = [T, N, *]`` and the time axis at dimension 0
        :type x_seq: torch.Tensor

        :return: Output sequence stacked along the time dimension
        :rtype: torch.Tensor

        :raises ValueError: If the number of updated states does not match the registered memories
        """
        inputs = (x_seq, *args)
        states = self.materialize_states(inputs, tuple(self._memories.values()), "m")
        for name, value in zip(self._memories, states, strict=True):
            self._memories[name] = value
        outputs, updated_states = self.multi_step_functional_forward(
            inputs, states, **kwargs
        )
        for name, value in zip(self._memories, updated_states, strict=True):
            self._memories[name] = value
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.forward-cn>` | :ref:`English <MemoryModule.forward-en>`

        ----

        .. _MemoryModule.forward-cn:

        * **中文**

        按 ``step_mode`` 调用 :meth:`single_step_forward` 或
        :meth:`multi_step_forward`。functional 状态读写由这两个方法负责。

        :return: 与当前 ``step_mode`` 对应的前向传播结果
        :rtype: Any

        :raises ValueError: 当 ``self.step_mode`` 既不是 ``"s"`` 也不是 ``"m"`` 时抛出

        ----

        .. _MemoryModule.forward-en:

        * **English**

        Dispatch to :meth:`single_step_forward` or :meth:`multi_step_forward`
        according to ``step_mode``. Those methods own functional state loading
        and committing.

        :return: Forward result selected according to the current ``step_mode``
        :rtype: Any

        :raises ValueError: Raised when ``self.step_mode`` is neither ``"s"`` nor ``"m"``
        """
        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        if self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        raise ValueError(self.step_mode)

    def extra_repr(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.extra_repr-cn>` | :ref:`English <MemoryModule.extra_repr-en>`

        ----

        .. _MemoryModule.extra_repr-cn:

        * **中文**

        返回附加到 ``nn.Module.__repr__`` 输出中的摘要字符串。

        :return: 附加到模块字符串表示中的摘要，包含 ``step_mode`` 与 ``backend``
        :rtype: str

        ----

        .. _MemoryModule.extra_repr-en:

        * **English**

        Return the summary string appended to ``nn.Module.__repr__``.

        :return: Summary appended to the module string representation, including ``step_mode`` and ``backend``
        :rtype: str
        """
        return f"step_mode={self.step_mode}, backend={self.backend}"

    def register_memory(self, name: str, value):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.register_memory-cn>` | :ref:`English <MemoryModule.register_memory-en>`

        ----

        .. _MemoryModule.register_memory-cn:

        * **中文**

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量将被初始化为 ``value`` 。
        每次调用 ``self.reset()`` 函数后， ``self.name`` 都会被重置为 ``value`` 。

        .. warning::

            若状态变量是个 ``torch.Tensor`` ，则 **不应对其做原地修改操作** 。

        :param name: 状态变量的名字
        :type name: str
        :param value: 状态变量的初始与重制值
        :type value: Any

        :raises AssertionError: 当 ``name`` 已经是模块现有成员属性时抛出

        ----

        .. _MemoryModule.register_memory-en:

        * **English**

        Register the state variable to memory dict, which saves stateful variables (e.g.,
        the membrane potential of a spiking neuron). The variable will be initialized as
        ``value`` . ``self.name`` will be set to ``value`` after calling ``self.reset()`` .

        .. warning::

            **Do not modify the state variable in-place** if it's a ``torch.Tensor`` .

        :param name: state variable's name
        :type name: str
        :param value: state variable's initial and reset value
        :type value: Any

        :raises AssertionError: Raised when ``name`` already exists as an attribute of the module
        """
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.reset-cn>` | :ref:`English <MemoryModule.reset-en>`

        ----

        .. _MemoryModule.reset-cn:

        * **中文**

        重置所有有状态变量为重制值。

        若当前状态与重制值均为同形状、同 dtype、同 device 的张量，则优先原地恢复；
        否则使用复制或重新赋值恢复。


        ----

        .. _MemoryModule.reset-en:

        * **English**

        Reset all stateful variables to their reset values.

        If both the current state and the reset value are tensors with the same
        shape, dtype, and device, the state is restored in-place whenever
        possible; otherwise it falls back to copy or reassignment.
        """
        for key in self._memories.keys():
            cur = self._memories[key]
            rv = self._memories_rv[key]
            if (
                isinstance(cur, torch.Tensor)
                and isinstance(rv, torch.Tensor)
                and cur.shape == rv.shape
                and cur.dtype == rv.dtype
                and cur.device == rv.device
            ):
                # detach_() breaks stale autograd graphs before in-place copy.
                # Falls back to deepcopy when cur is a view tensor
                # (detach_() raises RuntimeError on views).
                try:
                    cur.detach_()
                except RuntimeError:
                    self._memories[key] = rv.detach().clone()
                else:
                    cur.copy_(rv)
            elif isinstance(cur, torch.Tensor) and isinstance(rv, (int, float)):
                # Preserve Python-scalar sentinel semantics so the next forward
                # can materialize a fresh tensor with the new runtime shape.
                self._memories[key] = rv
            else:
                self._memories[key] = copy.deepcopy(rv)

    def set_reset_value(self, name: str, value):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.set_reset_value-cn>` | :ref:`English <MemoryModule.set_reset_value-en>`

        ----

        .. _MemoryModule.set_reset_value-cn:

        * **中文**

        设置状态变量 ``self.name`` 的重制值。

        :param name: 状态变量名称
        :type name: str
        :param value: 新的重制值
        :type value: Any

        ----

        .. _MemoryModule.set_reset_value-en:

        * **English**

        Set the reset value of state variable ``self.name`` .

        :param name: Name of the state variable
        :type name: str
        :param value: New reset value
        :type value: Any
        """
        self._memories_rv[name] = copy.deepcopy(value)

    def get_reset_value(self, name: str) -> Any:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.get_reset_value-cn>` | :ref:`English <MemoryModule.get_reset_value-en>`

        ----

        .. _MemoryModule.get_reset_value-cn:

        * **中文**

        获取状态变量 ``self.name`` 的重置值。

        :param name: 状态变量名称
        :type name: str
        :return: 状态变量的重置值
        :rtype: Any
        :raises KeyError: 当 ``name`` 不是已注册的状态变量，或没有设置重置值时抛出

        ----

        .. _MemoryModule.get_reset_value-en:

        * **English**

        Get the reset value of state variable ``self.name``.

        :param name: Name of the state variable
        :type name: str
        :return: Reset value of the state variable
        :rtype: Any
        :raises KeyError: Raised when ``name`` is not a registered state variable, or has no reset value
        """
        if name not in self._memories:
            raise KeyError(f"{name} is not a registered memory.")
        if name not in self._memories_rv:
            raise KeyError(f"{name} has no reset value. Call set_reset_value first.")
        return self._memories_rv[name]

    def __getattr__(self, name: str):
        #! use self.__dict__ instead of direct access to avoid infinite recursion
        if "_memories" in self.__dict__:
            memories = self.__dict__["_memories"]
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        #! use self.__dict__ instead of direct access to avoid infinite recursion
        _memories = self.__dict__.get("_memories")
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self) -> Generator:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.memories-cn>` | :ref:`English <MemoryModule.memories-en>`

        ----

        .. _MemoryModule.memories-cn:

        * **中文**

        :return: 返回一个所有状态变量的生成器
        :rtype: Generator

        ----

        .. _MemoryModule.memories-en:

        * **English**

        :return: a generator over all stateful variables
        :rtype: Generator
        """
        for _, value in self._memories.items():
            yield value

    def named_memories(self) -> Generator:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.named_memories-cn>` | :ref:`English <MemoryModule.named_memories-en>`

        ----

        .. _MemoryModule.named_memories-cn:

        * **中文**

        :return: 返回一个所有状态变量名称及其值的生成器
        :rtype: Generator

        ----

        .. _MemoryModule.named_memories-en:

        * **English**

        :return: a generator over all stateful variables' names and values
        :rtype: Generator
        """
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.detach-cn>` | :ref:`English <MemoryModule.detach-en>`

        ----

        .. _MemoryModule.detach-cn:

        * **中文**

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT (Truncated Back Propagation Through Time)。


        ----

        .. _MemoryModule.detach-en:

        * **English**

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT (Truncated Back Propagation Through Time).
        """
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


def named_memories(module: nn.Module, prefix: str = "") -> Generator:
    r"""
    **API Language** - :ref:`中文 <named_memories-cn>` | :ref:`English <named_memories-en>`

    ----

    .. _named_memories-cn:

    * **中文**

    递归地生成模块树中的所有状态变量。类似于 ``named_parameters()`` 方法。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param prefix: 名称前缀
    :type prefix: str

    :return: 状态变量名称和值的生成器
    :rtype: Generator

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _named_memories-en:

    * **English**

    Recursively yield all memory variables in a module tree. Similar to ``named_parameters()`` .

    :param module: the target module
    :type module: torch.nn.Module

    :param prefix: name prefix
    :type prefix: str

    :return: a generator of memory variable names and values
    :rtype: Generator

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    if isinstance(module, MemoryModule):
        for name, value in module.named_memories():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, value

    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        yield from named_memories(child, prefix=child_prefix)


def memories(module: nn.Module) -> Generator:
    r"""
    **API Language** - :ref:`中文 <memories-cn>` | :ref:`English <memories-en>`

    ----

    .. _memories-cn:

    * **中文**

    递归地生成模块树中的所有状态变量值。类似于 ``parameters()`` 方法。

    :param module: 目标模块
    :type module: nn.Module

    :return: 状态变量值的生成器
    :rtype: Generator

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _memories-en:

    * **English**

    Recursively yield all memory variables in a module tree. Similar to ``parameters()`` .

    :param module: the target module
    :type module: nn.Module

    :return: a generator of memory variable values
    :rtype: Generator

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    for _, value in named_memories(module):
        yield value


def extract_memories(module: nn.Module) -> list:
    r"""
    **API Language** - :ref:`中文 <extract_memories-cn>` | :ref:`English <extract_memories-en>`

    ----

    .. _extract_memories-cn:

    * **中文**

    提取模块中所有的状态变量值并返回列表。

    :param module: 目标模块
    :type module: torch.nn.Module

    :return: 状态变量值的列表
    :rtype: list

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _extract_memories-en:

    * **English**

    Extract all memory variable values from the module and return as a list.

    :param module: the target module
    :type module: torch.nn.Module

    :return: a list of memory variable values
    :rtype: list

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    return [m for m in memories(module)]


def load_memories(module: nn.Module, memory_list: list):
    r"""
    **API Language** - :ref:`中文 <load_memories-cn>` | :ref:`English <load_memories-en>`

    ----

    .. _load_memories-cn:

    * **中文**

    将状态变量列表加载到模块中。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param memory_list: 状态变量值列表
    :type memory_list: list

    :raises ValueError: 当 ``memory_list`` 的长度与 ``module`` 当前状态变量数量不一致时抛出

    ----

    .. _load_memories-en:

    * **English**

    Load memory variables from a list into the module.

    :param module: the target module
    :type module: torch.nn.Module

    :param memory_list: list of memory variable values
    :type memory_list: list

    :raises ValueError: Raised when the length of ``memory_list`` does not match the number of current memory variables in ``module``
    """

    def _assign_memory_by_name(module: nn.Module, name: str, value):
        parts = name.split(".")
        obj = module
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    named = list(named_memories(module))

    if len(named) != len(memory_list):
        raise ValueError(
            f"Memory length mismatch: model has {len(named)} memories "
            f"but list contains {len(memory_list)}"
        )

    for (name, _), value in zip(named, memory_list):
        _assign_memory_by_name(module, name, value)


def to_functional_forward(
    module: nn.Module, fn: Optional[Callable] = None
) -> Callable[..., tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]]:
    r"""
    **API Language** - :ref:`中文 <to_functional_forward-cn>` | :ref:`English <to_functional_forward-en>`

    ----

    .. _to_functional_forward-cn:

    * **中文**

    给定一个可能包含隐式状态变量（记忆，memory）的模块，获取其显式状态的前向传播函数。

    返回的函数签名为
    ``(inputs, states, **kwargs) -> (outputs, updated_states)`` ，
    其中：

    - ``inputs`` 为原始 ``forward`` 所需的常规输入参数；
    - ``states`` 为当前模块中所有状态变量的值，其顺序与 ``extract_memories(module)`` 一致；
    - ``outputs`` 为原始 ``forward`` 的输出结果；
    - ``new_states`` 为执行前向传播后更新得到的状态变量。

    单个输入和输出也使用 tuple 包装。状态顺序与
    ``extract_memories(module)`` 一致。

    .. note::

        转换优先直接使用 MemoryModule 的 functional 实现，其次递归组合
        ``nn.Sequential``，其他模块才使用临时 ``_memories`` 字典替换。

    .. warning::

        如果某个状态变量为 ``torch.Tensor`` ，则其不应在 ``module.forward`` 中被原地修改。否则，
        会导致输入给前向传播函数的状态变量被修改，导致意想不到的错误。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param fn: 含隐式状态的前向传播函数。若为 ``None`` ，则默认使用 ``module.forward`` 。
           该参数可用于指定特殊的前向传播函数（如， ``module`` 的父类的 ``forward`` ）。默认值
           为 ``None`` 。
    :type fn: Optional[Callable]

    :return: 带有显式输入输出状态的前向传播函数
    :rtype: Callable

    :raises ValueError: 调用时的状态数量或 memory 布局与转换时不一致

    ----

    .. _to_functional_forward-en:

    * **English**

    Given a module that may contain implicit state variables, get the forward function
    with explicit state variables.

    The returned function has the signature
    ``(inputs, states, **kwargs) -> (outputs, updated_states)``
    where:

    - ``inputs`` are the regular input arguments required by the original ``forward``;
    - ``states`` are the current memory variable values, in the same order as
      returned by ``extract_memories(module)``;
    - ``outputs`` are the outputs of the original ``forward`` method;
    - ``new_states`` are the updated memory variables after the forward pass.

    Single inputs and outputs are still wrapped in tuples. State order matches
    ``extract_memories(module)``.

    .. note::

        Conversion first uses a MemoryModule functional implementation directly,
        then recursively composes ``nn.Sequential`` modules. Other modules use a
        temporary ``_memories`` dictionary swap as the fallback.

    .. warning::

        If a state variable is a ``torch.Tensor``, it should not be modified in-place
        in ``module.forward``. Otherwise, the provided states will be modified,
        which may lead to unexpected errors.

    :param module: the target module
    :type module: torch.nn.Module

    :param fn: the forward function to be used. If ``None``, ``module.forward`` is used
           by default. This argument can be used to explicitly specify another forward
           function (e.g., the ``forward`` method of ``module``'s parent class).
           Defaults to ``None``.
    :type fn: Optional[Callable]

    :return: a functional-style forward function with grouped inputs and states
    :rtype: Callable
    :raises ValueError: If the state count or memory layout differs from conversion time

    ----

    * **代码示例 | Example**

    .. code:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based import base

        class StatefulModule(base.MemoryModule):
            def __init__(self):
                super().__init__()
                self.register_memory("counter", torch.tensor(0.0))
                self.linear = nn.Linear(10, 5)

            def single_step_forward(self, x):
                self.counter = self.counter + 1.0
                return self.linear(x)

        module = StatefulModule()
        f_forward = base.to_functional_forward(module)
        x = torch.randn(3, 10)
        initial_state = torch.tensor(0.0)
        outputs, new_states = f_forward((x,), (initial_state,))

        assert torch.equal(outputs[0], module.linear(x))
        assert torch.equal(new_states[0], initial_state + 1.0)
    """
    memory_modules = [
        child for child in module.modules() if isinstance(child, MemoryModule)
    ]
    layout = [(child, tuple(child._memories)) for child in memory_modules]
    num_states = sum(len(names) for _, names in layout)

    if fn is None and isinstance(module, nn.Sequential):
        occurrences = [
            id(child) for _, child in module.named_modules(remove_duplicate=False)
        ]
        if len(occurrences) == len(set(occurrences)) and not (
            isinstance(module, MemoryModule) and module._memories
        ):
            children = list(module._modules.values())
            child_forwards = [to_functional_forward(child) for child in children]
            child_state_counts = [
                sum(
                    len(memory_module._memories)
                    for memory_module in child.modules()
                    if isinstance(memory_module, MemoryModule)
                )
                for child in children
            ]

            def sequential_forward(
                inputs: tuple[torch.Tensor, ...],
                states: tuple[Any, ...],
                **kwargs: Any,
            ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
                if len(states) != num_states:
                    raise ValueError(
                        f"{module.__class__.__name__} expected {num_states} states, "
                        f"but got {len(states)}."
                    )
                outputs = inputs
                updated_states = []
                offset = 0
                for index, (child_forward, state_count) in enumerate(
                    zip(child_forwards, child_state_counts)
                ):
                    child_states = states[offset : offset + state_count]
                    outputs, child_updated_states = child_forward(
                        outputs,
                        child_states,
                        **(kwargs if index == 0 else {}),
                    )
                    updated_states.extend(child_updated_states)
                    offset += state_count
                return outputs, tuple(updated_states)

            return sequential_forward

    if fn is None and isinstance(module, MemoryModule):
        method_names = ["functional_forward"]
        if module.step_mode == "s":
            method_names.append("single_step_functional_forward")
        elif module.step_mode == "m":
            method_names.extend(
                ("single_step_functional_forward", "multi_step_functional_forward")
            )
        has_functional_forward = any(
            getattr(type(module), name) is not getattr(MemoryModule, name)
            for name in method_names
        )
        if has_functional_forward and len(memory_modules) == 1:

            def direct_forward(
                inputs: tuple[torch.Tensor, ...],
                states: tuple[Any, ...],
                **kwargs: Any,
            ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
                if len(states) != num_states:
                    raise ValueError(
                        f"{module._get_name()} expected {num_states} states, "
                        f"but got {len(states)}."
                    )
                outputs, updated_states = module.functional_forward(
                    inputs, states, **kwargs
                )
                return outputs, updated_states

            return direct_forward

    if fn is None and isinstance(module, nn.Sequential):

        def forward_fn(*inputs, **kwargs):
            outputs = inputs
            for index, child in enumerate(module._modules.values()):
                result = child(*outputs, **(kwargs if index == 0 else {}))
                outputs = (
                    tuple(result) if isinstance(result, (tuple, list)) else (result,)
                )
            return outputs[0] if len(outputs) == 1 else outputs

    else:
        forward_fn = module.forward if fn is None else fn

    def fallback_forward(
        inputs: tuple[torch.Tensor, ...],
        states: tuple[Any, ...],
        **kwargs: Any,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        if len(states) != num_states:
            raise ValueError(
                f"{module.__class__.__name__} expected {num_states} states, "
                f"but got {len(states)}."
            )
        original_memories = [memory_module._memories for memory_module, _ in layout]
        offset = 0
        for memory_module, names in layout:
            values = states[offset : offset + len(names)]
            temporary = dict(zip(names, values))
            memory_module.__dict__["_memories"] = temporary
            offset += len(names)

        try:
            outputs = forward_fn(*inputs, **kwargs)
            if isinstance(outputs, (tuple, list)):
                outputs = tuple(outputs)
            else:
                outputs = (outputs,)
            updated_states = tuple(
                memory_module._memories[name]
                for memory_module, names in layout
                for name in names
            )
        finally:
            for (memory_module, _), original in zip(layout, original_memories):
                memory_module.__dict__["_memories"] = original
        return outputs, updated_states

    return fallback_forward
