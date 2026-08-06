from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .. import base, functional


__all__ = [
    "MultiStepContainer",
    "SeqToANNContainer",
    "TLastMultiStepContainer",
    "TLastSeqToANNContainer",
    "StepModeContainer",
    "ElementWiseRecurrentContainer",
    "LinearRecurrentContainer",
]


def _check_step_mode(block: nn.Sequential):
    for m in block:
        if getattr(m, "step_mode", "s") != "s":
            raise ValueError(
                f"{type(m).__name__}.step_mode must be 's' inside this container."
            )


class MultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        r"""
        **API Language** - :ref:`中文 <MultiStepContainer.__init__-cn>` | :ref:`English <MultiStepContainer.__init__-en>`

        ----

        .. _MultiStepContainer.__init__-cn:

        * **中文**

        :func:`spikingjelly.activation_based.functional.multi_step_forward`
        的容器。构造方式与 `torch.nn.Sequential` 一致。

        ----

        .. _MultiStepContainer.__init__-en:

        * **English**

        Container of :func:`spikingjelly.activation_based.functional.multi_step_forward`.
        Its constructor signature is the same as `torch.nn.Sequential`.
        """
        super().__init__(*args)
        _check_step_mode(self)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: with shape ``[T, batch_size, ...]``
        :type x_seq: torch.Tensor

        :return: y_seq with shape ``[T, batch_size, ...]``
        :rtype: torch.Tensor
        """
        return functional.multi_step_forward(x_seq, (super().forward,))


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        **API Language** - :ref:`中文 <SeqToANNContainer-cn>` | :ref:`English <SeqToANNContainer-en>`

        ----

        .. _SeqToANNContainer-cn:

        * **中文**

        :func:`spikingjelly.activation_based.functional.seq_to_ann_forward`
        的容器。构造方式与 `torch.nn.Sequential` 一致。

        ----

        .. _SeqToANNContainer-en:

        * **English**

        Container of :func:`spikingjelly.activation_based.functional.seq_to_ann_forward`.
        Its constructor signature is the same as `torch.nn.Sequential`.
        """
        super().__init__(*args)
        _check_step_mode(self)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: with shape ``[T, batch_size, ...]``
        :type x_seq: torch.Tensor

        :return: y_seq with shape ``[T, batch_size, ...]``
        :rtype: torch.Tensor
        """
        return functional.seq_to_ann_forward(x_seq, (super().forward,))


class TLastSeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        See :func:`spikingjelly.activation_based.functional.forward.t_last_seq_to_ann_forward` .
        """
        super().__init__(*args)
        _check_step_mode(self)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape ``[batch_size, ..., T]``
        :type x_seq: Tensor

        :return: y_seq with shape ``[batch_size, ..., T]``
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, (super().forward,))


class TLastMultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        See :func:`spikingjelly.activation_based.functional.forward.t_last_multi_step_forward` .
        """
        super().__init__(*args)
        _check_step_mode(self)

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape ``[batch_size, ..., T]``
        :type x_seq: Tensor

        :return: y_seq with shape ``[batch_size, ..., T]``
        :rtype: Tensor
        """
        return functional.t_last_multi_step_forward(x_seq, (super().forward,))


class StepModeContainer(nn.Sequential, base.StepModule):
    def __init__(self, stateful: bool, step_mode: str = "s", *args):
        """
        Call single-step forward, multi-step forward or seq-to-ANN forward according to
        ``stateful`` and ``step_mode``.

        :param stateful: 是否是有状态的容器
        :type stateful: bool
        :param step_mode: 步进模式，``\"s\"`` 或 ``\"m\"``
        :type step_mode: str
        :param args: 与 ``torch.nn.Sequential`` 相同的构造参数

        :param stateful: Whether the container is stateful
        :type stateful: bool
        :param step_mode: Step mode, ``\"s\"`` or ``\"m\"``
        :type step_mode: str
        :param args: Same constructor arguments as ``torch.nn.Sequential``
        :type args: tuple
        """
        super().__init__(*args)
        self.stateful = stateful
        _check_step_mode(self)
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if self.stateful:
            return functional.multi_step_forward(x, (super().forward,))
        return functional.seq_to_ann_forward(x, (super().forward,))


class ElementWiseRecurrentContainer(base.MemoryModule):
    def __init__(
        self, sub_module: nn.Module, element_wise_function: Callable, step_mode="s"
    ):
        r"""
        **API Language** - :ref:`中文 <ElementWiseRecurrentContainer.__init__-cn>` | :ref:`English <ElementWiseRecurrentContainer.__init__-en>`

        ----

        .. _ElementWiseRecurrentContainer.__init__-cn:

        * **中文**

        使用逐元素运算的自连接包装器。记 ``sub_module`` 的输入输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入为 :math:`x[t]`，则

        .. math::

            i[t] = f(y[t-1], x[t])

        其中 :math:`f` 是用户自定义的逐元素函数。我们默认 :math:`y[-1] = 0`。


        .. Note::

            ``sub_module`` 输入和输出的尺寸需要相同。

        :param sub_module: 被包含的模块
        :type sub_module: torch.nn.Module

        :param element_wise_function: 用户自定义的逐元素函数，应该形如 ``z=f(y, x)``
        :type element_wise_function: Callable

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _ElementWiseRecurrentContainer.__init__-en:

        * **English**

        A container that use a element-wise recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = f(y[t-1], x[t])

        where :math:`f` is the user-defined element-wise function. We set :math:`y[-1] = 0`.

        .. admonition:: Note
            :class: note

            The shape of inputs and outputs of ``sub_module`` must be the same.

        :param sub_module: the contained module
        :type sub_module: torch.nn.Module

        :param element_wise_function: the user-defined element-wise function, which should have the format ``z=f(y, x)``
        :type element_wise_function: Callable

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            T = 8
            net = ElementWiseRecurrentContainer(
                neuron.IFNode(v_reset=None), element_wise_function=lambda x, y: x + y
            )
            print(net)
            x = torch.zeros([T])
            x[0] = 1.5
            for t in range(T):
                print(t, f"x[t]={x[t]}, s[t]={net(x[t])}")

            functional.reset_net(net)
        """
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, "step_mode") or sub_module.step_mode == "s"
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory("y", None)

    def materialize_states(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0] if step_mode == "s" else inputs[0][0]
        return (torch.zeros_like(x) if states[0] is None else states[0],)

    def single_step_functional_forward(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        y = states[0]
        y = self.sub_module(self.element_wise_function(y, x))
        return (y,), (y,)

    def extra_repr(self) -> str:
        return f"element-wise function={self.element_wise_function}, step_mode={self.step_mode}"


class LinearRecurrentContainer(base.MemoryModule):
    def __init__(
        self,
        sub_module: nn.Module,
        in_features: int,
        out_features: int,
        bias: bool = True,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <LinearRecurrentContainer.__init__-cn>` | :ref:`English <LinearRecurrentContainer.__init__-en>`

        ----

        .. _LinearRecurrentContainer.__init__-cn:

        * **中文**

        使用线性层的自连接包装器。记 ``sub_module`` 的输入和输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入记作 :math:`x[t]` ，则

        .. math::

            i[t] = \begin{pmatrix} x[t] \\ y[t-1]\end{pmatrix} W^{T} + b

        其中 :math:`W, b` 是线性层的权重和偏置项。默认 :math:`y[-1] = 0`。

        :math:`x[t]` 应该 ``shape = [N, *, in_features]``，:math:`y[t]` 则应该 ``shape = [N, *, out_features]``。

        .. Note::

            自连接是由 ``torch.nn.Linear(in_features + out_features, in_features, bias)`` 实现的。

        :param sub_module: 被包含的模块
        :type sub_module: torch.nn.Module

        :param in_features: 输入的特征数量
        :type in_features: int

        :param out_features: 输出的特征数量
        :type out_features: int

        :param bias: 若为 ``False``，则线性自连接不会带有可学习的偏执项
        :type bias: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _LinearRecurrentContainer.__init__-en:

        * **English**

        A container that use a linear recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = \begin{pmatrix} x[t] \\ y[t-1]\end{pmatrix} W^{T} + b

        where :math:`W, b` are the weight and bias of the linear connection. We set :math:`y[-1] = 0`.

        :math:`x[t]` should have the shape ``[N, *, in_features]``, and :math:`y[t]` has the shape ``[N, *, out_features]``.

        .. admonition:: Note
            :class: note

            The recurrent connection is implement by ``torch.nn.Linear(in_features + out_features, in_features, bias)``.

        :param sub_module: the contained module
        :type sub_module: torch.nn.Module

        :param in_features: size of each input sample
        :type in_features: int

        :param out_features: size of each output sample
        :type out_features: int

        :param bias: If set to ``False``, the linear recurrent layer will not learn an additive bias
        :type bias: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            in_features = 4
            out_features = 2
            T = 8
            N = 2
            net = LinearRecurrentContainer(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    neuron.LIFNode(),
                ),
                in_features,
                out_features,
            )
            print(net)
            x = torch.rand([T, N, in_features])
            for t in range(T):
                print(t, net(x[t]))

            functional.reset_net(net)
        """
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, "step_mode") or sub_module.step_mode == "s"
        self.sub_module_out_features = out_features
        self.rc = nn.Linear(in_features + out_features, in_features, bias)
        self.sub_module = sub_module
        self.register_memory("y", None)

    def materialize_states(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0] if step_mode == "s" else inputs[0][0]
        y = states[0]
        if y is None:
            y = x.new_zeros(*x.shape[:-1], self.sub_module_out_features)
        return (y,)

    def single_step_functional_forward(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        y = states[0]
        y = self.sub_module(self.rc(torch.cat((x, y), dim=-1)))
        return (y,), (y,)

    def extra_repr(self) -> str:
        return f", step_mode={self.step_mode}"
