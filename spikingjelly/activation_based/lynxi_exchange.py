import copy
import logging
import os
from typing import Dict, Union

import torch
import torch.nn as nn

from . import layer, neuron

"""
TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  for t in range(x_seq.shape[0]):

不支持inplace操作，因此形如x[t] = y之类的操作都无效，x[t]并不会被设置为y，且不会报错

不支持5D的tensor参与模型编译，在任何位置都不能出现超过4D的tensor

"""


class BaseNode(nn.Module):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        step_mode="s",
        T: int = None,
        return_v: bool = False,
    ):
        r"""
        **API Language:**
        :ref:`中文 <BaseNode.__init__-cn>` | :ref:`English <BaseNode.__init__-en>`

        ----

        .. _BaseNode.__init__-cn:

        * **中文**

        适配灵汐（Lynxi）芯片的脉冲神经元基类。
        与标准的 :class:`spikingjelly.activation_based.neuron.BaseNode` 不同，
        该类需要显式指定时间步数 ``T``，以满足灵汐编译器对静态计算图的要求。
        不支持inplace操作，且模型中任何位置均不得出现超过4D的tensor。

        神经元动力学分为三步：

        1. 充电（由子类实现的 ``neuronal_charge``）；
        2. 放电：:math:`\text{spike} = \Theta(v - v_{\text{threshold}})`，其中
           :math:`\Theta` 为 Heaviside 阶跃函数；
        3. 重置：

           - 硬重置（``v_reset`` 非 ``None``）：
             :math:`v = (1 - \text{spike}) \cdot v + \text{spike} \cdot v_{\text{reset}}`
           - 软重置（``v_reset`` 为 ``None``）：
             :math:`v = v - \text{spike} \cdot v_{\text{threshold}}`

        :param v_threshold: 神经元膜电位的发放阈值
        :type v_threshold: float

        :param v_reset: 神经元的硬重置电位。若为 ``None``，则采用软重置
        :type v_reset: float

        :param step_mode: 步进模式，``'s'`` 为单步模式，``'m'`` 为多步模式
        :type step_mode: str

        :param T: 仿真时间步数。多步模式下必须指定
        :type T: int

        :param return_v: 若为 ``True``，``forward`` 同时返回膜电位 ``v``；否则只返回脉冲
        :type return_v: bool

        ----

        .. _BaseNode.__init__-en:

        * **English**

        Base class for Lynxi-chip-compatible spiking neurons. Unlike the standard
        :class:`spikingjelly.activation_based.neuron.BaseNode`, this class requires an
        explicit number of time steps ``T`` to satisfy the static computation-graph
        requirements of the Lynxi compiler. Inplace operations are not supported, and
        tensors with more than 4 dimensions must not appear anywhere in the model.

        Neuron dynamics follow three steps:

        1. Charge (``neuronal_charge``, implemented by subclasses);
        2. Fire: :math:`\text{spike} = \Theta(v - v_{\text{threshold}})`, where
           :math:`\Theta` is the Heaviside step function;
        3. Reset:

           - Hard reset (``v_reset`` is not ``None``):
             :math:`v = (1 - \text{spike}) \cdot v + \text{spike} \cdot v_{\text{reset}}`
           - Soft reset (``v_reset`` is ``None``):
             :math:`v = v - \text{spike} \cdot v_{\text{threshold}}`

        :param v_threshold: Firing threshold of the membrane potential
        :type v_threshold: float

        :param v_reset: Hard-reset potential. If ``None``, soft reset is used instead
        :type v_reset: float

        :param step_mode: Step mode. ``'s'`` for single-step, ``'m'`` for multi-step
        :type step_mode: str

        :param T: Number of simulation time steps. Required for multi-step mode
        :type T: int

        :param return_v: If ``True``, ``forward`` returns ``(spike, v)``; otherwise
            only ``spike``
        :type return_v: bool
        """
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.step_mode = step_mode
        self.T = T
        self.return_v = return_v

    def neuronal_charge(self, x: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError

    def single_step_forward(self, x: torch.Tensor, v: torch.Tensor = None):
        if v is None:
            v = torch.zeros_like(x)
        v = self.neuronal_charge(x, v)

        spike = (v >= self.v_threshold).to(x)
        if self.v_reset is None:
            v = v - spike * self.v_threshold
        else:
            v = (1.0 - spike) * v + spike * self.v_reset

        return spike, v

    def multi_step_forward(self, x_seq: torch.Tensor, v_init: torch.Tensor = None):
        if v_init is None:
            v = torch.zeros_like(x_seq[0])
        else:
            v = v_init
        spike_seq = []
        for t in range(self.T):
            spike, v = self.single_step_forward(x_seq[t], v)
            spike_seq.append(spike.unsqueeze(0))

        spike_seq = torch.cat(spike_seq)
        return spike_seq, v

    def forward(self, x: torch.Tensor, v: torch.Tensor = None):
        if self.step_mode == "s":
            spike, v = self.single_step_forward(x, v)
            if self.return_v:
                return spike, v
            else:
                return spike
        elif self.step_mode == "m":
            x_shape = x.shape

            # 起始 编译通过-------------------
            x = x.reshape(self.T, x.shape[0] // self.T, -1)
            # 终结 编译通过-------------------

            # 起始 编译报错-------------------
            # x = x.flatten(1)
            # x = unfold_seq(self.T, x)
            # 终结 编译报错-------------------

            if v is not None:
                v = v.flatten()
            spike_seq, v = self.multi_step_forward(x, v)

            spike_seq = spike_seq.flatten(0, 1).reshape(x_shape)

            if self.return_v:
                v = v.reshape([x_shape[0] // self.T] + list(x_shape[1:]))
                return spike_seq, v
            else:
                return spike_seq


class IFNode(BaseNode):
    r"""
    **API Language:**
    :ref:`中文 <IFNode-cn>` | :ref:`English <IFNode-en>`

    ----

    .. _IFNode-cn:

    * **中文**

    适配灵汐芯片的积分放电（Integrate-and-Fire，IF）神经元。
    继承自 :class:`BaseNode` ，充电方程为：

    .. math::

        v[t] = v[t-1] + x[t]

    构造参数请参见 :class:`BaseNode` 。

    ----

    .. _IFNode-en:

    * **English**

    Lynxi-compatible Integrate-and-Fire (IF) neuron. Inherits from
    :class:`BaseNode` . The charge equation is:

    .. math::

        v[t] = v[t-1] + x[t]

    For constructor parameters, see :class:`BaseNode` .
    """

    def neuronal_charge(self, x: torch.Tensor, v: torch.Tensor):
        return x + v


class LIFNode(BaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        step_mode="s",
        T: int = None,
        return_v: bool = False,
    ):
        r"""
        **API Language:**
        :ref:`中文 <LIFNode.__init__-cn>` | :ref:`English <LIFNode.__init__-en>`

        ----

        .. _LIFNode.__init__-cn:

        * **中文**

        适配灵汐芯片的泄漏积分放电（Leaky Integrate-and-Fire，LIF）神经元。继承自
        :class:`BaseNode` ，充电方程取决于重置类型和 ``decay_input`` ：

        硬重置（ ``v_reset`` 非 ``None`` ）：

        .. math::

            v[t] = \left(1 - \frac{1}{\tau}\right)(v[t-1] - v_{\text{reset}}) +
            \begin{cases}
                \dfrac{x[t]}{\tau} & \text{if decay\_input} \\
                x[t] & \text{otherwise}
            \end{cases}

        软重置（ ``v_reset`` 为 ``None`` ）：

        .. math::

            v[t] = \left(1 - \frac{1}{\tau}\right) v[t-1] +
            \begin{cases}
                \dfrac{x[t]}{\tau} & \text{if decay\_input} \\
                x[t] & \text{otherwise}
            \end{cases}

        :param tau: 膜电位时间常数，控制泄漏速率，:math:`\tau > 1`
        :type tau: float

        :param decay_input: 若为 ``True``，输入电流也会按 ``1/tau`` 衰减
        :type decay_input: bool

        :param v_threshold: 神经元膜电位的发放阈值
        :type v_threshold: float

        :param v_reset: 神经元的硬重置电位。若为 ``None``，则采用软重置
        :type v_reset: float

        :param step_mode: 步进模式，``'s'`` 为单步模式，``'m'`` 为多步模式
        :type step_mode: str

        :param T: 仿真时间步数。多步模式下必须指定
        :type T: int

        :param return_v: 若为 ``True``，``forward`` 同时返回膜电位 ``v``；否则只返回脉冲
        :type return_v: bool

        ----

        .. _LIFNode.__init__-en:

        * **English**

        Lynxi-compatible Leaky Integrate-and-Fire (LIF) neuron. Inherits from
        :class:`BaseNode`. The charge equation depends on the reset type and
        ``decay_input``:

        Hard reset (``v_reset`` is not ``None``):

        .. math::

            v[t] = \left(1 - \frac{1}{\tau}\right)(v[t-1] - v_{\text{reset}}) +
            \begin{cases}
                \dfrac{x[t]}{\tau} & \text{if decay\_input} \\
                x[t] & \text{otherwise}
            \end{cases}

        Soft reset (``v_reset`` is ``None``):

        .. math::

            v[t] = \left(1 - \frac{1}{\tau}\right) v[t-1] +
            \begin{cases}
                \dfrac{x[t]}{\tau} & \text{if decay\_input} \\
                x[t] & \text{otherwise}
            \end{cases}

        :param tau: Membrane time constant controlling the leak rate, :math:`\tau > 1`
        :type tau: float

        :param decay_input: If ``True``, the input current is also scaled by ``1/tau``
        :type decay_input: bool

        :param v_threshold: Firing threshold of the membrane potential
        :type v_threshold: float

        :param v_reset: Hard-reset potential. If ``None``, soft reset is used instead
        :type v_reset: float

        :param step_mode: Step mode. ``'s'`` for single-step, ``'m'`` for multi-step
        :type step_mode: str

        :param T: Number of simulation time steps. Required for multi-step mode
        :type T: int

        :param return_v: If ``True``, ``forward`` returns ``(spike, v)``; otherwise
            only ``spike``
        :type return_v: bool
        """
        super().__init__(v_threshold, v_reset, step_mode, T, return_v)
        self.decay = 1.0 / tau
        self.decay_input = decay_input

    def neuronal_charge(self, x: torch.Tensor, v: torch.Tensor):
        if self.v_reset is None:
            v = (1.0 - self.decay) * v
        else:
            v = (1.0 - self.decay) * (v - self.v_reset)

        if self.decay_input:
            x = x * self.decay

        return v + x


def to_lynxi_supported_module(m_in: nn.Module, T: int):
    r"""
    **API Language:**
    :ref:`中文 <to_lynxi_supported_module-cn>` | :ref:`English <to_lynxi_supported_module-en>`

    ----

    .. _to_lynxi_supported_module-cn:

    * **中文**

    将单个 SpikingJelly 步进模块转换为灵汐芯片兼容的标准 PyTorch 模块。

    转换规则如下：

    - :class:`spikingjelly.activation_based.layer.Conv2d` →
      :class:`torch.nn.Conv2d`（权重通过 ``load_state_dict`` 复制）
    - :class:`spikingjelly.activation_based.layer.BatchNorm2d` →
      :class:`torch.nn.BatchNorm2d`
    - :class:`spikingjelly.activation_based.layer.MaxPool2d` →
      :class:`torch.nn.MaxPool2d`
    - :class:`spikingjelly.activation_based.layer.AvgPool2d` →
      :class:`torch.nn.AvgPool2d`
    - :class:`spikingjelly.activation_based.layer.AdaptiveAvgPool2d` →
      :class:`torch.nn.AdaptiveAvgPool2d`
    - :class:`spikingjelly.activation_based.layer.Flatten` →
      :class:`torch.nn.Flatten`
    - :class:`spikingjelly.activation_based.neuron.IFNode` →
      :class:`IFNode`（本模块内）
    - :class:`spikingjelly.activation_based.neuron.LIFNode` →
      :class:`LIFNode`（本模块内）
    - 其他类型：记录 ``critical`` 日志，返回深拷贝的原模块（移至 CPU）

    :param m_in: 待转换的输入模块
    :type m_in: torch.nn.Module

    :param T: 仿真时间步数，传递给神经元节点
    :type T: int

    :return: 转换后的灵汐兼容模块
    :rtype: torch.nn.Module

    ----

    .. _to_lynxi_supported_module-en:

    * **English**

    Convert a single SpikingJelly step module to a Lynxi-chip-compatible standard
    PyTorch module.

    Conversion rules:

    - :class:`spikingjelly.activation_based.layer.Conv2d` →
      :class:`torch.nn.Conv2d` (weights copied via ``load_state_dict``)
    - :class:`spikingjelly.activation_based.layer.BatchNorm2d` →
      :class:`torch.nn.BatchNorm2d`
    - :class:`spikingjelly.activation_based.layer.MaxPool2d` →
      :class:`torch.nn.MaxPool2d`
    - :class:`spikingjelly.activation_based.layer.AvgPool2d` →
      :class:`torch.nn.AvgPool2d`
    - :class:`spikingjelly.activation_based.layer.AdaptiveAvgPool2d` →
      :class:`torch.nn.AdaptiveAvgPool2d`
    - :class:`spikingjelly.activation_based.layer.Flatten` →
      :class:`torch.nn.Flatten`
    - :class:`spikingjelly.activation_based.neuron.IFNode` →
      :class:`IFNode` (this module)
    - :class:`spikingjelly.activation_based.neuron.LIFNode` →
      :class:`LIFNode` (this module)
    - Other types: log a ``critical`` message and return a deep-copied CPU version
      of the original module

    :param m_in: Input module to convert
    :type m_in: torch.nn.Module

    :param T: Number of simulation time steps, forwarded to neuron nodes
    :type T: int

    :return: Lynxi-compatible module
    :rtype: torch.nn.Module
    """
    if isinstance(m_in, layer.Conv2d):
        m_out = nn.Conv2d(
            in_channels=m_in.in_channels,
            out_channels=m_in.out_channels,
            kernel_size=m_in.kernel_size,
            stride=m_in.stride,
            padding=m_in.padding,
            dilation=m_in.dilation,
            groups=m_in.groups,
            bias=m_in.bias is not None,
            padding_mode=m_in.padding_mode,
        )
        m_out.load_state_dict(m_in.state_dict())

    elif isinstance(m_in, layer.BatchNorm2d):
        m_out = nn.BatchNorm2d(
            num_features=m_in.num_features,
            eps=m_in.eps,
            momentum=m_in.momentum,
            affine=m_in.affine,
            track_running_stats=m_in.affine,
        )
        m_out.load_state_dict(m_in.state_dict())

    elif isinstance(m_in, layer.MaxPool2d):
        m_out = nn.MaxPool2d(
            kernel_size=m_in.kernel_size,
            stride=m_in.stride,
            padding=m_in.padding,
            dilation=m_in.dilation,
            return_indices=m_in.return_indices,
            ceil_mode=m_in.ceil_mode,
        )

    elif isinstance(m_in, layer.AvgPool2d):
        m_out = nn.AvgPool2d(
            kernel_size=m_in.kernel_size,
            stride=m_in.stride,
            padding=m_in.padding,
            ceil_mode=m_in.ceil_mode,
            count_include_pad=m_in.count_include_pad,
            divisor_override=m_in.divisor_override,
        )

    elif isinstance(m_in, layer.AdaptiveAvgPool2d):
        m_out = nn.AdaptiveAvgPool2d(output_size=m_in.output_size)

    elif isinstance(m_in, layer.Flatten):
        m_out = nn.Flatten(start_dim=m_in.start_dim, end_dim=m_in.end_dim)

    elif isinstance(m_in, neuron.IFNode):
        m_out = IFNode(
            v_threshold=m_in.v_threshold,
            v_reset=m_in.v_reset,
            step_mode=m_in.step_mode,
            T=T,
            return_v=False,
        )

    elif isinstance(m_in, neuron.LIFNode):
        m_out = LIFNode(
            tau=m_in.tau,
            v_threshold=m_in.v_threshold,
            v_reset=m_in.v_reset,
            decay_input=m_in.decay_input,
            step_mode=m_in.step_mode,
            T=T,
            return_v=False,
        )

    else:
        logging.critical(
            f"{type(m_in)} is not processed and the origin module is used for lynxi compiling."
        )
        m_out = copy.deepcopy(m_in).cpu()

    return m_out


def to_lynxi_supported_modules(net: Union[list, tuple, nn.Sequential], T: int):
    r"""
    **API Language:**
    :ref:`中文 <to_lynxi_supported_modules-cn>` | :ref:`English <to_lynxi_supported_modules-en>`

    ----

    .. _to_lynxi_supported_modules-cn:

    * **中文**

    将 SpikingJelly 模块序列中的每个模块依次通过
    :func:`to_lynxi_supported_module` 转换为灵汐兼容模块，返回转换后模块的列表。

    :param net: 待转换的模块序列，可为 ``list``、``tuple`` 或
        ``torch.nn.Sequential``
    :type net: list or tuple or torch.nn.Sequential

    :param T: 仿真时间步数，传递给各神经元节点
    :type T: int

    :return: 转换后的灵汐兼容模块列表
    :rtype: list

    ----

    .. _to_lynxi_supported_modules-en:

    * **English**

    Convert every module in a SpikingJelly module sequence to a Lynxi-compatible
    module by calling :func:`to_lynxi_supported_module` on each element, and
    return the results as a list.

    :param net: Module sequence to convert. Can be a ``list``, ``tuple``, or
        ``torch.nn.Sequential``
    :type net: list or tuple or torch.nn.Sequential

    :param T: Number of simulation time steps, forwarded to each neuron node
    :type T: int

    :return: List of Lynxi-compatible modules
    :rtype: list
    """
    output_net = []
    for i in range(net.__len__()):
        m_in = net[i]
        m_out = to_lynxi_supported_module(m_in, T)
        output_net.append(m_out)

    return output_net


try:
    """
    适配灵汐科技的芯片

    """
    import lyngor
    import lynpy

    logging.info(f"lynpy.version={lynpy.version}")
    logging.info(f"lyngor.version={lyngor.version}")

    def torch_tensor_to_lynxi(x: torch.Tensor, device_id: int = 0, to_apu: bool = True):
        r"""
        **API Language:**
        :ref:`中文 <torch_tensor_to_lynxi-cn>` | :ref:`English <torch_tensor_to_lynxi-en>`

        ----

        .. _torch_tensor_to_lynxi-cn:

        * **中文**

        将 PyTorch tensor 转换为灵汐（Lynxi） ``lynpy.Tensor``，并可选地将其搬运至
        APU 设备显存。

        转换流程：

        1. 计算 tensor 占用的字节数；
        2. 将 tensor 搬至 CPU 并转为 NumPy 数组；
        3. 构造 ``lynpy.Tensor`` 并从 NumPy 数组初始化；
        4. 若 ``to_apu=True``，调用 ``.apu()`` 将数据搬运至 APU 设备。

        :param x: 待转换的 PyTorch tensor
        :type x: torch.Tensor

        :param device_id: 目标灵汐设备 ID
        :type device_id: int

        :param to_apu: 若为 ``True``，在构造后立即将 tensor 搬运至 APU 设备
        :type to_apu: bool

        :return: 灵汐设备上的 tensor
        :rtype: lynpy.Tensor

        ----

        .. _torch_tensor_to_lynxi-en:

        * **English**

        Convert a PyTorch tensor to a Lynxi ``lynpy.Tensor``, optionally moving it
        to APU device memory.

        Conversion steps:

        1. Compute the byte size of the tensor;
        2. Move the tensor to CPU and convert to a NumPy array;
        3. Construct a ``lynpy.Tensor`` and initialise it from the NumPy array;
        4. If ``to_apu=True``, call ``.apu()`` to transfer data to the APU device.

        :param x: PyTorch tensor to convert
        :type x: torch.Tensor

        :param device_id: Target Lynxi device ID
        :type device_id: int

        :param to_apu: If ``True``, move the tensor to the APU device after construction
        :type to_apu: bool

        :return: Tensor on the Lynxi device
        :rtype: lynpy.Tensor
        """
        x_size_in_byte = x.element_size() * x.numel()
        x = x.cpu().detach().numpy()
        x = lynpy.Tensor(dev_id=device_id, size=x_size_in_byte).from_numpy(x)
        if to_apu:
            x = x.apu()
        return x

    def lynxi_tensor_to_torch(
        x: lynpy.Tensor, shape: Union[tuple, list] = None, dtype: str = None
    ):
        r"""
        **API Language:**
        :ref:`中文 <lynxi_tensor_to_torch-cn>` | :ref:`English <lynxi_tensor_to_torch-en>`

        ----

        .. _lynxi_tensor_to_torch-cn:

        * **中文**

        将灵汐（Lynxi） ``lynpy.Tensor`` 转换回 PyTorch tensor。

        转换流程：

        1. 若同时提供 ``shape`` 和 ``dtype``，先调用 ``view_as`` 重新解释内存布局；
        2. 若 tensor 仍在 APU 设备（``devptr`` 非 ``None``），调用 ``.cpu()`` 将数据
           搬回主机内存；
        3. 调用 ``.numpy()`` 取得 NumPy 数组，再通过 ``torch.from_numpy`` 转为
           PyTorch tensor。

        :param x: 待转换的灵汐 tensor
        :type x: lynpy.Tensor

        :param shape: 目标形状。需与 ``dtype`` 同时提供
        :type shape: tuple or list, optional

        :param dtype: 目标数据类型字符串（如 ``'float32'``）。需与 ``shape`` 同时提供
        :type dtype: str, optional

        :return: 转换后的 PyTorch tensor（位于 CPU）
        :rtype: torch.Tensor

        ----

        .. _lynxi_tensor_to_torch-en:

        * **English**

        Convert a Lynxi ``lynpy.Tensor`` back to a PyTorch tensor.

        Conversion steps:

        1. If both ``shape`` and ``dtype`` are provided, call ``view_as`` to
           reinterpret the memory layout;
        2. If the tensor is still on the APU device (``devptr`` is not ``None``),
           call ``.cpu()`` to transfer data back to host memory;
        3. Call ``.numpy()`` to obtain a NumPy array, then convert to a PyTorch
           tensor via ``torch.from_numpy``.

        :param x: Lynxi tensor to convert
        :type x: lynpy.Tensor

        :param shape: Target shape. Must be provided together with ``dtype``
        :type shape: tuple or list, optional

        :param dtype: Target dtype string (e.g. ``'float32'``). Must be provided
            together with ``shape``
        :type dtype: str, optional

        :return: PyTorch tensor on CPU
        :rtype: torch.Tensor
        """
        if shape is not None and dtype is not None:
            x = x.view_as(shape, dtype)
        if x.devptr is not None:
            x = x.cpu()
        x = torch.from_numpy(x.numpy())
        return x

    def compile_lynxi_model(
        output_dir: str,
        net: nn.Module,
        in_data_type: str = "float32",
        out_data_type: str = "float32",
        input_shape_dict: Dict = {},
    ):
        r"""
        **API Language:**
        :ref:`中文 <compile_lynxi_model-cn>` | :ref:`English <compile_lynxi_model-en>`

        ----

        .. _compile_lynxi_model-cn:

        * **中文**

        使用 ``lyngor`` 将 PyTorch 模型编译为灵汐 APU 离线模型，并将编译产物保存至
        指定目录。

        编译流程：

        1. 创建 ``lyngor.DLModel`` 并以 Pytorch 格式加载模型；
        2. 创建 ``lyngor.Builder`` 并以 APU 为目标执行离线编译；
        3. 打印输出目录内容并返回主网络路径。

        .. note::
            传入的 ``net`` 应已通过 :func:`to_lynxi_supported_modules` 转换为
            灵汐兼容模块，且其中不得出现超过4D的tensor或inplace操作。

        :param output_dir: 编译产物的输出目录路径
        :type output_dir: str

        :param net: 待编译的 PyTorch 模型
        :type net: torch.nn.Module

        :param in_data_type: 模型输入数据类型，默认 ``'float32'``
        :type in_data_type: str

        :param out_data_type: 模型输出数据类型，默认 ``'float32'``
        :type out_data_type: str

        :param input_shape_dict: 输入张量形状字典，键为输入名，值为形状
        :type input_shape_dict: dict

        :return: 编译后主网络的路径（``output_dir/Net_0``）
        :rtype: str

        ----

        .. _compile_lynxi_model-en:

        * **English**

        Compile a PyTorch model into a Lynxi APU offline model using ``lyngor``,
        and save the compilation artifacts to the specified directory.

        Compilation steps:

        1. Create a ``lyngor.DLModel`` and load the model in Pytorch format;
        2. Create a ``lyngor.Builder`` targeting the APU and run offline compilation;
        3. Print the output directory contents and return the main network path.

        .. note::
            The ``net`` passed in should have been converted to Lynxi-compatible
            modules via :func:`to_lynxi_supported_modules`. Tensors with more than
            4 dimensions and inplace operations are not allowed.

        :param output_dir: Output directory path for compilation artifacts
        :type output_dir: str

        :param net: PyTorch model to compile
        :type net: torch.nn.Module

        :param in_data_type: Input data type for the model, default ``'float32'``
        :type in_data_type: str

        :param out_data_type: Output data type for the model, default ``'float32'``
        :type out_data_type: str

        :param input_shape_dict: Dict mapping input names to their shapes
        :type input_shape_dict: dict

        :return: Path to the compiled main network (``output_dir/Net_0``)
        :rtype: str
        """
        model = lyngor.DLModel()
        model.load(
            net,
            model_type="Pytorch",
            in_type=in_data_type,
            out_type=out_data_type,
            inputs_dict=input_shape_dict,
        )
        offline_builder = lyngor.Builder(target="apu", is_map=True)
        out_path = offline_builder.build(
            model.graph, model.params, out_path=output_dir, apu_only=True
        )
        print(os.listdir(out_path))
        return os.path.join(out_path, "Net_0")

    def load_lynxi_model(device_id: int, model_path: str):
        r"""
        **API Language:**
        :ref:`中文 <load_lynxi_model-cn>` | :ref:`English <load_lynxi_model-en>`

        ----

        .. _load_lynxi_model-cn:

        * **中文**

        从磁盘加载已编译的灵汐离线模型，返回可用于推理的 ``lynpy.Model`` 实例。

        模型路径通常由 :func:`compile_lynxi_model` 返回的路径（``Net_0`` 目录）给出。

        :param device_id: 目标灵汐设备 ID
        :type device_id: int

        :param model_path: 已编译模型所在目录路径（通常为 ``Net_0``）
        :type model_path: str

        :return: 可用于推理的灵汐模型实例
        :rtype: lynpy.Model

        ----

        .. _load_lynxi_model-en:

        * **English**

        Load a compiled Lynxi offline model from disk and return a ``lynpy.Model``
        instance ready for inference.

        The model path is typically the ``Net_0`` directory returned by
        :func:`compile_lynxi_model`.

        :param device_id: Target Lynxi device ID
        :type device_id: int

        :param model_path: Path to the compiled model directory (usually ``Net_0``)
        :type model_path: str

        :return: Lynxi model instance ready for inference
        :rtype: lynpy.Model
        """
        return lynpy.Model(dev_id=device_id, path=model_path)


except BaseException as e:
    logging.info(f"spikingjelly.activation_based.lynxi_exchange: {e}")
