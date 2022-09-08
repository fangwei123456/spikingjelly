import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from . import neuron, monitor, base


def stdp_linear_single_step(fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
                            trace_pre: float or torch.Tensor or None, trace_post: float or torch.Tensor or None,
                            tau_pre: float, tau_post: float, f_pre: Callable = lambda x: x,
                            f_post: Callable = lambda x: x):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    # shape = [N, C_in]
    trace_post = trace_post - trace_post / tau_post + out_spike
    # shape = [N, C_out]

    delta_w_pre = - f_pre(weight) * (trace_pre * in_spike).sum(0)
    delta_w_post = f_post(weight) * (trace_post * out_spike).sum(0).unsqueeze(1)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def stdp_linear_multi_step(fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
                           trace_pre: float or torch.Tensor or None, trace_post: float or torch.Tensor or None,
                           tau_pre: float, tau_post: float, f_pre: Callable = lambda x: x,
                           f_post: Callable = lambda x: x):
    """
    T = 4
    N = 2
    C_in = 4
    C_out = 8
    in_spike = (torch.rand([T, N, C_in]) > 0.9).float()
    out_spike = (torch.rand([T, N, C_out]) > 0.8).float()
    fc = nn.Linear(C_in, C_out)
    delta_w = stdp_linear_multi_step(fc, in_spike=in_spike, out_spike=out_spike, tau_pre=10., tau_post=20.)
    print(delta_w)
    """
    weight = fc.weight.data
    # weight.shape = [C_out, C_in]
    # in_spike.shape = [T, N, C_in]
    # out_spike.shape = [T, N, C_out]
    delta_w = torch.zeros_like(weight)

    for t in range(in_spike.shape[0]):
        trace_pre, trace_post, dw = stdp_linear_single_step(fc, in_spike[t], out_spike[t], trace_pre, trace_post,
                                                            tau_pre, tau_post, f_pre, f_post)
        delta_w += dw
    return trace_pre, trace_post, delta_w


def stdp_conv2d_single_step(conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
                            trace_pre: torch.Tensor or None, trace_post: torch.Tensor or None, tau_pre: float,
                            tau_post: float, f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x):
    if conv.dilation != (1, 1):
        raise NotImplementedError('STDP with dilation != 1 for Conv2d has not been implemented!')
    if conv.groups != 1:
        raise NotImplementedError('STDP with groups != 1 for Conv2d has not been implemented!')

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(in_spike, conv._reversed_padding_repeated_twice,
                             mode=conv.padding_mode)
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros(
            [conv.weight.shape[2], conv.weight.shape[3], in_spike.shape[0], in_spike.shape[1], out_spike.shape[2],
             out_spike.shape[3]], device=in_spike.device,
            dtype=in_spike.dtype)

    if trace_post is None:
        trace_post = torch.zeros([conv.weight.shape[2], conv.weight.shape[3], *out_spike.shape], device=in_spike.device,
                                 dtype=in_spike.dtype)

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]

            # shape = [N, C_in, ?, ?]
            post_spike = out_spike
            # shape = [N, C_out, ?, ?]
            weight = conv.weight.data[:, :, h, w]
            # shape = [C_out, C_in]
            tr_pre = trace_pre[h][w]
            tr_post = trace_post[h][w]

            tr_pre = tr_pre - tr_pre / tau_pre + pre_spike
            tr_post = tr_post - tr_post / tau_post + post_spike
            trace_pre[h][w] = tr_pre
            trace_post[h][w] = tr_post

            delta_w_pre = - f_pre(weight) * (tr_pre * pre_spike).transpose(0, 1).flatten(1).sum(1)
            delta_w_post = f_post(weight) * (tr_post * post_spike).transpose(0, 1).flatten(1).sum(1, keepdims=True)
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


def stdp_conv2d_multi_step(conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
                           trace_pre: torch.Tensor or None, trace_post: torch.Tensor or None, tau_pre: float,
                           tau_post: float, f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x):
    """
    T = 4
    N = 2
    C_in = 4
    C_out = 8
    in_spike = (torch.rand([T, N, C_in, 16, 16]) > 0.9).float()
    out_spike = (torch.rand([T, N, C_out, 16, 16]) > 0.8).float()
    conv = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False)

    delta_w = stdp_conv2d_multi_step(conv, in_spike=in_spike, out_spike=out_spike, tau_pre=10., tau_post=20.)
    print(delta_w)

    """

    delta_w = torch.zeros_like(conv.weight.data)

    for t in range(in_spike.shape[0]):
        trace_pre, trace_post, dw = stdp_conv2d_single_step(conv, in_spike[t], out_spike[t], trace_pre, trace_post,
                                                            tau_pre, tau_post, f_pre, f_post)
        delta_w += dw
    return trace_pre, trace_post, delta_w


class STDPLearner(base.MemoryModule):
    def __init__(self, step_mode: str, synapse: nn.Conv2d or nn.Linear, sn: neuron.BaseNode, tau_pre: float,
                 tau_post: float, f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x):
        """
        * :ref:`API in English <STDPLearner.__init__-en>`
        .. _STDPLearner.__init__-cn:

        :param step_mode: 步进模式，需要和 ``synapse`` 和 ``sn`` 在前向传播时使用的步进模式保持一致
        :type step_mode: str
        :param synapse: 突触层
        :type synapse: nn.Conv2d or nn.Linear
        :param sn: 脉冲神经元层
        :type sn: neuron.BaseNode
        :param tau_pre: pre神经元的迹时间常数
        :type tau_pre: float
        :param tau_post: post神经元的迹时间常数
        :type tau_post: float
        :param f_pre: pre神经元的权重函数
        :type f_pre: Callable
        :param f_post: post神经元的权重函数
        :type f_post: Callable

        STDP学习器。将 ``synapse`` 的输入脉冲视作 ``pre_spike``，``sn`` 的输出视作 ``post_spike``，根据 ``pre_spike`` 和 ``post_spike`` 生成 \
        ``trace_pre`` 和 ``trace_post``。迹 :math:`tr[t]` 的更新按照如下方式：

        .. math::

            tr_{pre}[t] = tr_{pre}[t] - \\frac{tr_{pre}[t-1]}{\\tau_{pre}} + s_{pre}[t]

            tr_{post}[t] = tr_{post}[t] -\\frac{tr_{post}[t-1]}{\\tau_{post}} + s_{post}[t]


        其中 :math:`tr_{pre}, tr_{post}` 是迹时间常数，即为参数 ``tau_pre`` 和 ``tau_post`` ；:math:`s_{pre}[t], s_{post}[t]` 即 ``pre_spike`` 和 ``post_spike``。

        对于pre神经元 ``i`` 和post神经元 ``j`` ，其连接权重 ``w[i][j]`` 的更新按照STDP学习规则

        .. math::

            \\Delta W[i][j][t] = F_{post}(w[i][j][t]) \\cdot tr_{j}[t] \\cdot s[j][t] - F_{pre}(w[i][j][t]) \\cdot tr_{i}[t] \\cdot s[i][t]

        其中 :math:`F_{pre}, F_{post}` 即为参数 ``f_pre`` 和 ``f_post``。


        ``STDPLearner`` 会使用2个监视器，记录 ``synapse`` 的输入当作 ``pre_spike``，和记录 ``sn`` 的输出当作 ``post_spike``。使用 \
        ``.enable()`` 和 ``.disable()`` 函数可以启用和暂停 ``STDPLearner`` 内部的数据记录。

        使用 ``step(on_grad, scale)`` 函数将根据STDP学习规则，计算出权重的更新量 ``delta_w``，而实际的梯度更新量为 ``delta_w * scale``，默认 ``scale = 1.``。

        特别的，设置 ``on_grad=False`` 则 ``step()`` 函数返回 ``delta_w * scale``；
        若设置 ``on_grad=True``，则 ``- delta_w * scale`` 会被加到 ``weight.grad``，这意味着我们可以通过 :class:`torch.optim.SGD` 之类的优化器来更新权重。注意这里有一个负号 ``-``，因为我们希望 ``weight.data += delta_w * scale``，但优化器的操作则是 ``weight.data -= lr * weight.grad``。
        默认 ``on_grad=True``。

        需要注意，``STDPLearner`` 也是有状态的，因其内部的 ``trace_pre`` 和 ``trace_post`` 是有状态的。因此在给与网络新输入前，也需要调用 ``.reset()`` 进行重置。

        代码示例：

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import neuron, layer, functional, learning
            step_mode = 's'
            lr = 0.001
            tau_pre = 100.
            tau_post = 50.
            T = 8
            N = 4
            C_in = 16
            C_out = 4
            H = 28
            W = 28
            x_seq = torch.rand([T, N, C_in, H, W])
            def f_weight(x):
                return torch.clamp(x, -1, 1.)

            net = nn.Sequential(
                neuron.IFNode(),
                layer.Conv2d(C_in, C_out, kernel_size=5, stride=2, padding=1, bias=False),
                neuron.IFNode()
            )
            functional.set_step_mode(net, step_mode)
            optimizer = torch.optim.SGD(net[1].parameters(), lr=lr, momentum=0.)
            stdp_learner = learning.STDPLearner(step_mode=step_mode, synapse=net[1], sn=net[2], tau_pre=tau_pre, tau_post=tau_post, f_pre=f_weight, f_post=f_weight)

            with torch.no_grad():
                for t in range(T):
                    weight = net[1].weight.data.clone()
                    net(x_seq[t])
                    stdp_learner.step(on_grad=True)
                    optimizer.step()
                    delta_w = net[1].weight.data - weight
                    print(f'delta_w=\\n{delta_w}')

            functional.reset_net(net)
            stdp_learner.reset()

        * :ref:`中文API <STDPLearner.__init__-cn>`
        .. _STDPLearner.__init__-en:

        :param step_mode: the step mode, which should be same with that of ``synapse`` and ``sn``
        :type step_mode: str
        :param synapse: the synapse
        :type synapse: nn.Conv2d or nn.Linear
        :param sn: the spiking neurons layer
        :type sn: neuron.BaseNode
        :param tau_pre: the time constant of trace of pre neurons
        :type tau_pre: float
        :param tau_post: the time constant of trace of post neurons
        :type tau_post: float
        :param f_pre: the weight function for pre neurons
        :type f_pre: Callable
        :param f_post: the weight function for post neurons
        :type f_post: Callable

        The STDP learner. It will regard inputs of ``synapse`` as ``pre_spike`` and outputs of ``sn`` as ``post_spike``, which will be used to generate ``trace_pre`` and ``trace_post``.

        The update of ``trace_pre`` and ``trace_post`` defined as:

        .. math::

            tr_{pre}[t] = tr_{pre}[t] - \\frac{tr_{pre}[t-1]}{\\tau_{pre}} + s_{pre}[t]

            tr_{post}[t] = tr_{post}[t] -\\frac{tr_{post}[t-1]}{\\tau_{post}} + s_{post}[t]

        where :math:`tr_{pre}, tr_{post}` are time constants，which are ``tau_pre`` and ``tau_post``. :math:`s_{pre}[t], s_{post}[t]` are ``pre_spike`` and ``post_spike``.

        For the pre neuron ``i`` and post neuron ``j``, the synapse ``weight[i][j]`` is updated by the STDP learning rule:

         .. math::

            \\Delta W[i][j][t] = F_{post}(w[i][j][t]) \\cdot tr_{j}[t] \\cdot s[j][t] - F_{pre}(w[i][j][t]) \\cdot tr_{i}[t] \\cdot s[i][t]

        where :math:`F_{pre}, F_{post}` are ``f_pre`` and ``f_post``.


        ``STDPLearner`` will use two monitors to record inputs of ``synapse`` as ``pre_spike`` and outputs of ``sn`` as ``post_spike``. We can use ```.enable()``` or ``.disable()`` to start or pause the monitors.

        We can use ``step(on_grad, scale)`` to apply the STDP learning rule and get the update variation ``delta_w``, while the actual update variation is ``delta_w * scale``. We set ``scale = 1.`` as the default value.

        Note that when we set ``on_grad=False``, then ``.step()`` will return ``delta_w * scale``.
        If we set ``on_grad=True``, then ``- delta_w * scale`` will be added in ``weight.grad``, indicating that we can use optimizers like :class:`torch.optim.SGD` to update weights. Note that there is a negative sign ``-`` because we want the operation ``weight.data += delta_w * scale``, but the optimizer will apply ``weight.data -= lr * weight.grad``.
        We set ``on_grad=True`` as the default value.

        Note that ``STDPLearner`` is also stateful because its ``trace_pre`` and ``trace_post`` are stateful. Do not forget to call ``.reset()`` before giving a new sample to the network.

        Codes example:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import neuron, layer, functional, learning
            step_mode = 's'
            lr = 0.001
            tau_pre = 100.
            tau_post = 50.
            T = 8
            N = 4
            C_in = 16
            C_out = 4
            H = 28
            W = 28
            x_seq = torch.rand([T, N, C_in, H, W])
            def f_weight(x):
                return torch.clamp(x, -1, 1.)

            net = nn.Sequential(
                neuron.IFNode(),
                layer.Conv2d(C_in, C_out, kernel_size=5, stride=2, padding=1, bias=False),
                neuron.IFNode()
            )
            functional.set_step_mode(net, step_mode)
            optimizer = torch.optim.SGD(net[1].parameters(), lr=lr, momentum=0.)
            stdp_learner = learning.STDPLearner(step_mode=step_mode, synapse=net[1], sn=net[2], tau_pre=tau_pre, tau_post=tau_post, f_pre=f_weight, f_post=f_weight)

            with torch.no_grad():
                for t in range(T):
                    weight = net[1].weight.data.clone()
                    net(x_seq[t])
                    stdp_learner.step(on_grad=True)
                    optimizer.step()
                    delta_w = net[1].weight.data - weight
                    print(f'delta_w=\\n{delta_w}')

            functional.reset_net(net)
            stdp_learner.reset()
        """


        super().__init__()
        self.step_mode = step_mode

        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

    def reset(self):
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        """
        * :ref:`API in English <STDPLearner.disable-en>`
        .. _STDPLearner.disable-cn:

        暂停对 ``synapse`` 的输入和 ``sn`` 的输出的记录。

        * :ref:`中文API <STDPLearner.disable-cn>`
        .. _STDPLearner.disable-en:

        Pause the monitoring of inputs of ``synapse`` and outputs of ``sn``.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        """
        * :ref:`API in English <STDPLearner.disable-en>`
        .. _STDPLearner.disable-cn:

        恢复对 ``synapse`` 的输入和 ``sn`` 的输出的记录。

        * :ref:`中文API <STDPLearner.disable-cn>`
        .. _STDPLearner.disable-en:

        Enable the monitoring of inputs of ``synapse`` and outputs of ``sn``.
        """
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad=True, scale: float = 1.):
        """
        * :ref:`API in English <STDPLearner.step-en>`
        .. _STDPLearner.step-cn:

        :param on_grad: 是否将更新量叠加到参数梯度。若为 ``True`` 则会将 ``- delta_w * scale`` 加到 ``weight.grad``；若为 ``False`` 则本函数会返回 ``delta_w * scale``
        :type on_grad: bool
        :param scale: 更新量的系数，作用类似于学习率
        :type scale: float
        :return: None or ``delta_w * scale``
        :rtype: None or torch.Tensor

        * :ref:`中文API <STDPLearner.step-cn>`
        .. _STDPLearner.step-en:

        :param on_grad: whether add the update variation on ``weight.grad``. If ``True``, then ``- delta_w * scale`` will be added on ``weight.grad``. If `False`, then this function will return ``delta_w * scale``
        :type on_grad: bool
        :param scale: the scale of ``delta_w``, which acts like the learning rate
        :type scale: float
        :return: None or ``delta_w * scale``
        :rtype: None or torch.Tensor
        """
        length = self.in_spike_monitor.records.__len__()
        for i in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)

            if self.step_mode == 's':
                if isinstance(self.synapse, nn.Conv2d):
                    stdp_f = stdp_conv2d_single_step
                elif isinstance(self.synapse, nn.Linear):
                    stdp_f = stdp_linear_single_step
                else:
                    raise NotImplementedError(self.synapse)

            elif self.step_mode == 'm':
                if isinstance(self.synapse, nn.Conv2d):
                    stdp_f = stdp_conv2d_multi_step
                elif isinstance(self.synapse, nn.Linear):
                    stdp_f = stdp_linear_multi_step
                else:
                    raise NotImplementedError(self.synapse)
            else:
                raise ValueError(self.step_mode)

            self.trace_pre, self.trace_post, delta_w = stdp_f(self.synapse, in_spike, out_spike, self.trace_pre,
                                                              self.trace_post, self.tau_pre, self.tau_post, self.f_pre,
                                                              self.f_post)
            if scale != 1.:
                delta_w *= scale

            if on_grad:
                if self.synapse.weight.grad is None:
                    self.synapse.weight.grad = - delta_w
                else:
                    self.synapse.weight.grad = self.synapse.weight.grad - delta_w
            else:
                return delta_w
