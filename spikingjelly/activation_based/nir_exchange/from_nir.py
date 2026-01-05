from typing import Union

import numpy as np
import torch
from torch import fx
import nir
import nirtorch

from .. import layer, functional, neuron


def _from_numpy(x):
    if isinstance(x, (int, np.int64)):
        return int(x)
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    else:
        return tuple(x)


class _NodeMapper:

    def __init__(self, dt: float = 1e-4):
        self.dt = dt

    @property
    def map_dict(self) -> dict:
        return {
            nir.Affine: self.map_affine,
            nir.Linear: self.map_linear,
            nir.Conv2d: self.map_conv2d,
            nir.AvgPool2d: self.map_avgpool2d,
            nir.Flatten: self.map_flatten,
            nir.IF: self.map_if,
            nir.LIF: self.map_lif,
        } # all the functions return single-step modules

    def map_affine(self, node: nir.Affine) -> layer.Linear:
        module = layer.Linear(
            node.weight.shape[-1], node.weight.shape[-2], bias=True
        )
        module.weight.data = torch.from_numpy(node.weight)
        module.bias.data = torch.from_numpy(node.bias)
        return module

    def map_linear(self, node: nir.Linear) -> layer.Linear:
        module = layer.Linear(
            node.weight.shape[-1], node.weight.shape[-2], bias=False
        )
        module.weight.data = torch.from_numpy(node.weight)
        return module

    def map_conv2d(self, node: nir.Conv2d) -> layer.Conv2d:
        weight = node.weight
        bias = node.bias
        stride = _from_numpy(node.stride)
        padding = _from_numpy(node.padding)
        dilation = _from_numpy(node.dilation)
        groups = _from_numpy(node.groups)

        module = layer.Conv2d(
            in_channels=weight.shape[-4],
            out_channels=weight.shape[-3],
            kernel_size=weight.shape[-2:],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        module.weight.data = torch.from_numpy(weight)
        module.bias.data = torch.from_numpy(bias)
        return module

    def map_avgpool2d(self, node: nir.AvgPool2d) -> layer.AvgPool2d:
        kernel_size = _from_numpy(node.kernel_size)
        stride = _from_numpy(node.stride)
        padding = _from_numpy(node.padding)

        return layer.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def map_flatten(self, node: nir.Flatten) -> layer.Flatten:
        start_dim, end_dim = node.start_dim, node.end_dim
        start_dim = start_dim + 1 if start_dim >= 0 else start_dim
        end_dim = end_dim + 1 if end_dim >= 0 else end_dim
        return layer.Flatten(start_dim, end_dim)

    def map_if(self, node: nir.IF) -> neuron.IFNode:
        v_threshold = np.unique(node.v_threshold)
        if v_threshold.size != 1:
            raise AssertionError(
                "`v_threshold` must be the same for all neurons!"
            )
        v_threshold = _from_numpy(v_threshold[0])

        v_reset = np.unique(node.v_reset)
        if v_reset.size != 1:
            raise AssertionError(
                "`v_reset` must be the same for all neurons!"
            )
        v_reset = _from_numpy(v_reset[0])

        # r can be reconstructed directly from self.dt
        r_value = 1.0 / self.dt
        if not np.allclose(np.unique(node.r), r_value):
            raise AssertionError(
                "`nir.IF.r` mismatch 1.0/dt !"
            )

        return neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

    def map_lif(self, node: nir.LIF) -> neuron.LIFNode:
        tau_ = np.unique(node.tau)
        if tau_.size != 1:
            raise AssertionError(
                "`tau` must be the same for all neurons!"
            )
        tau = _from_numpy(tau_[0] / self.dt)  # reverse tau_ = tau * dt

        r = np.unique(node.r)
        if r.size != 1:
            raise AssertionError(
                "`r` must be the same for all neurons!"
            )
        r = _from_numpy(r[0])
        # note that: r = 1. if decay_input else tau
        decay_input = False if r == tau else (True if r == 1. else False)

        v_reset = np.unique(node.v_reset)
        if v_reset.size != 1:
            raise AssertionError(
                "`v_reset` must be the same for all neurons!"
            )
        v_reset = _from_numpy(v_reset[0])

        # Recover v_leak (usually equals v_reset in torch->NIR conversion)
        v_leak = np.unique(node.v_leak)
        if v_leak.size != 1:
            raise AssertionError(
                "`v_leak` must be the same for all neurons!"
            )
        v_leak = _from_numpy(v_leak[0])
        if v_leak != v_reset:
            raise AssertionError(
                "`v_leak` should be equal to `v_reset`!"
            )

        v_threshold = np.unique(node.v_threshold)
        if v_threshold.size != 1:
            raise AssertionError(
                "`v_threshold` must be the same for all neurons!"
            )
        v_threshold = _from_numpy(v_threshold[0])

        return neuron.LIFNode(
            tau=tau,
            decay_input=decay_input,
            v_reset=v_reset,
            v_threshold=v_threshold,
        )


def import_from_nir(
    graph: Union[nir.NIRGraph, str],
    dt: float = 1e-4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    step_mode: str = "s",
) -> fx.GraphModule:
    """
    **API Language:**
    :ref:`中文 <import_from_nir-cn>` | :ref:`English <import_from_nir-en>`

    ----

    .. _import_from_nir-cn:

    * **中文**

    将 `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_ 图
    转换为 SpikingJelly 神经网络模型。函数会根据 NIR 节点类型自动映射为对应的 
    SpikingJelly 模块（如 Linear、Conv2d、IF/LIF 神经元等），并返回可直接运行的
    ``fx.GraphModule`` 对象。

    :param graph: NIR 图，或存储 NIR 图的 HDF5 文件路径
    :type graph: Union[nir.NIRGraph, str]

    :param dt: 网络时间步长，单位为秒，用于重构 IF/LIF 节点的时间常量等超参数。默认值为 ``1e-4``，与大多数兼容 NIR 的框架一致
    :type dt: float

    :param device: 模型运行设备，如 ``'cpu'`` 或 ``'cuda'``
    :type device: str

    :param dtype: 模型张量数据类型，通常为 ``torch.float32`` 或 ``torch.float64``
    :type dtype: torch.dtype

    :param step_mode: 步进模式，可选 ``'s'`` (单步) 或 ``'m'`` (多步)。NIR 图将首先转换到单步模式的 SpikingJelly 模型，
        随后统一改变模型中所有子模块的步进模式
    :type step_mode: str

    :return: 转换得到的 ``fx.GraphModule`` 对象
    :rtype: torch.fx.GraphModule

    ----

    .. _import_from_nir-en:

    * **English**

    Convert a `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_
    graph to a SpikingJelly model. The function automatically maps NIR nodes to
    corresponding SpikingJelly modules (e.g., Linear, Conv2d, IF/LIF neurons)
    and returns an runnable :class:`fx.GraphModule <https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule>` object.

    :param graph: NIR graph, or the path to the HDF5 file storing the NIR graph
    :type graph: Union[nir.NIRGraph, str]

    :param dt: simulation time step in seconds, used to reconstruct time constant
        and other neuronal hyperparameters. Default is ``1e-4``, which is consistent
        with most frameworks that support NIR
    :type dt: float

    :param device: device on which the model will run, e.g., ``'cpu'`` or ``'cuda'``
    :type device: str

    :param dtype: data type of model tensors, usually ``torch.float32`` or ``torch.float64``
    :type dtype: torch.dtype

    :param step_mode: step mode, either ``'s'`` (single-step) or ``'m'`` (multi-step).
        NIR graph will first be converted to a single-step SpikingJelly model.
        Then, all the submodules will be set to the specified step mode.
    :type step_mode: str

    :return: the converted SpikingJelly ``fx.GraphModule`` object
    :rtype: torch.fx.GraphModule
    """
    mapper = _NodeMapper(dt=dt)

    gm = nirtorch.nir_to_torch(
        graph, mapper.map_dict, device=device, dtype=dtype
    )
    functional.set_step_mode(gm, step_mode)
    return gm