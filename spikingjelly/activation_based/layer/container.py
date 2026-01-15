"""For more information about container, refer to :doc:`../tutorials/en/container` ."""

import logging
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


class MultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        **API Language:**
        :ref:`中文 <MultiStepContainer-cn>` | :ref:`English <MultiStepContainer-en>`

        ----

        .. _MultiStepContainer-cn:

        * **中文**

        :func:`spikingjelly.activation_based.functional.multi_step_forward`
        的容器。构造方式与 `torch.nn.Sequential` 一致。

        ----

        .. _MultiStepContainer-en:

        * **English**

        Container of :func:`spikingjelly.activation_based.functional.multi_step_forward`.
        Its constructor signature is the same as `torch.nn.Sequential`.
        """
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, "step_mode") or m.step_mode == "s"
            if isinstance(m, base.StepModule):
                if "m" in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports step_mode == 'm', which should not be contained by MultiStepContainer!"
                    )

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[T, batch_size, ...]``
        :type x_seq: torch.Tensor

        :return: y_seq with ``shape=[T, batch_size, ...]``
        :rtype: torch.Tensor
        """
        return functional.multi_step_forward(x_seq, super().forward)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        **API Language:**
        :ref:`中文 <SeqToANNContainer-cn>` | :ref:`English <SeqToANNContainer-en>`

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
        for m in self:
            assert not hasattr(m, "step_mode") or m.step_mode == "s"
            if isinstance(m, base.StepModule):
                if "m" in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports step_mode == 'm', which should not be contained by SeqToANNContainer!"
                    )

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: torch.Tensor

        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: torch.Tensor
        """
        return functional.seq_to_ann_forward(x_seq, super().forward)


class TLastMultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, "step_mode") or m.step_mode == "s"
            if isinstance(m, base.StepModule):
                if "m" in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!"
                    )

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[batch_size, ..., T]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[batch_size, ..., T]``
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward)


class TLastSeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        Please refer to :class:`spikingjelly.activation_based.functional.t_last_seq_to_ann_forward` .
        """
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, "step_mode") or m.step_mode == "s"
            if isinstance(m, base.StepModule):
                if "m" in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!"
                    )

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[batch_size, ..., T]
        :type x_seq: Tensor
        :return: y_seq, shape=[batch_size, ..., T]
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward)


class StepModeContainer(nn.Sequential, base.StepModule):
    def __init__(self, stateful: bool, *args):
        super().__init__(*args)
        self.stateful = stateful
        for m in self:
            assert not hasattr(m, "step_mode") or m.step_mode == "s"
            if isinstance(m, base.StepModule):
                if "m" in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by StepModeContainer!"
                    )
        self.step_mode = "s"

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        elif self.step_mode == "m":
            if self.stateful:
                return functional.multi_step_forward(x, super().forward)
            else:
                return functional.seq_to_ann_forward(x, super().forward)


class ElementWiseRecurrentContainer(base.MemoryModule):
    def __init__(
        self, sub_module: nn.Module, element_wise_function: Callable, step_mode="s"
    ):
        """
        * :ref:`API in English <ElementWiseRecurrentContainer-en>`

        .. _ElementWiseRecurrentContainer-cn:

        :param sub_module: 被包含的模块
        :type sub_module: torch.nn.Module
        :param element_wise_function: 用户自定义的逐元素函数，应该形如 ``z=f(x, y)``
        :type element_wise_function: Callable
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        使用逐元素运算的自连接包装器。记 ``sub_module`` 的输入输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入为 :math:`x[t]`，则

        .. math::

            i[t] = f(x[t], y[t-1])

        其中 :math:`f` 是用户自定义的逐元素函数。我们默认 :math:`y[-1] = 0`。


        .. Note::

            ``sub_module`` 输入和输出的尺寸需要相同。

        示例代码：

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


        * :ref:`中文 API <ElementWiseRecurrentContainer-cn>`

        .. _ElementWiseRecurrentContainer-en:

        :param sub_module: the contained module
        :type sub_module: torch.nn.Module
        :param element_wise_function: the user-defined element-wise function, which should have the format ``z=f(x, y)``
        :type element_wise_function: Callable
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        A container that use a element-wise recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = f(x[t], y[t-1])

        where :math:`f` is the user-defined element-wise function. We set :math:`y[-1] = 0`.

        .. admonition:: Note
            :class: note

            The shape of inputs and outputs of ``sub_module`` must be the same.

        Codes example:

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

    def single_step_forward(self, x: Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.y, x))
        return self.y

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
        """
        * :ref:`API in English <LinearRecurrentContainer-en>`

        .. _LinearRecurrentContainer-cn:

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

        使用线性层的自连接包装器。记 ``sub_module`` 的输入和输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入记作 :math:`x[t]` ，则

        .. math::

            i[t] = \\begin{pmatrix} x[t] \\\\ y[t-1]\\end{pmatrix} W^{T} + b

        其中 :math:`W, b` 是线性层的权重和偏置项。默认 :math:`y[-1] = 0`。

        :math:`x[t]` 应该 ``shape = [N, *, in_features]``，:math:`y[t]` 则应该 ``shape = [N, *, out_features]``。

        .. Note::

            自连接是由 ``torch.nn.Linear(in_features + out_features, in_features, bias)`` 实现的。

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

        * :ref:`中文 API <LinearRecurrentContainer-cn>`

        .. _LinearRecurrentContainer-en:

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

        A container that use a linear recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = \\begin{pmatrix} x[t] \\\\ y[t-1]\\end{pmatrix} W^{T} + b

        where :math:`W, b` are the weight and bias of the linear connection. We set :math:`y[-1] = 0`.

        :math:`x[t]` should have the shape ``[N, *, in_features]``, and :math:`y[t]` has the shape ``[N, *, out_features]``.

        .. admonition:: Note
            :class: note

            The recurrent connection is implement by ``torch.nn.Linear(in_features + out_features, in_features, bias)``.

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

    def single_step_forward(self, x: Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.sub_module_out_features]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.zeros(out_shape).to(x)
        x = torch.cat((x, self.y), dim=-1)
        self.y = self.sub_module(self.rc(x))
        return self.y

    def extra_repr(self) -> str:
        return f", step_mode={self.step_mode}"
