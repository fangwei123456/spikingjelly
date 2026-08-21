from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Optional, Union

import nir
import nirtorch
import numpy as np
import torch
import torch.nn as nn
from torch import fx

from .. import base, functional, layer, neuron
from ...logger import logger


__all__ = ["import_from_nir"]


def _to_python_value(x):
    value = np.asarray(x)
    return value.item() if value.ndim == 0 else tuple(value.tolist())


def _uniform_scalar(value: np.ndarray, name: str):
    unique = np.unique(value)
    if unique.size != 1:
        raise ValueError(f"{name} must be uniform across all neurons.")
    return unique.item()


def _matches_discretization(value: float, expected: float) -> bool:
    return bool(
        np.isfinite(value)
        and np.isfinite(expected)
        and np.isclose(value, expected, rtol=1e-6, atol=1e-12)
    )


def _has_cycle(graph: nir.NIRGraph) -> bool:
    predecessors = {name: set() for name in graph.nodes}
    for source, target in graph.edges:
        predecessors[target].add(source)
    try:
        TopologicalSorter(predecessors).prepare()
    except CycleError:
        return True
    return False


class _NIRStatefulModule(nn.Module):
    def __init__(self, module: base.MemoryModule):
        super().__init__()
        self.module = module

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[object, ...]] = None,
    ) -> tuple[torch.Tensor, tuple[object, ...]]:
        if state is None:
            state = tuple(base.extract_memories(self.module))
        outputs, updated_state = self.module.functional_forward((x,), tuple(state))
        return outputs[0], updated_state


class _NodeMapper:
    def __init__(self, dt: float = 1e-4):
        self.dt = dt

    @property
    def map_dict(self) -> dict:
        return {
            nir.Affine: self.map_affine,
            nir.Linear: self.map_linear,
            nir.Conv1d: self.map_conv1d,
            nir.Conv2d: self.map_conv2d,
            nir.AvgPool2d: self.map_avgpool2d,
            nir.Flatten: self.map_flatten,
            nir.IF: self.map_if,
            nir.LIF: self.map_lif,
            nir.CubaLIF: self.map_cuba_lif,
        }

    def map_affine(self, node: nir.Affine) -> layer.Linear:
        module = layer.Linear(node.weight.shape[-1], node.weight.shape[-2], bias=True)
        module.weight.data = torch.from_numpy(node.weight)
        module.bias.data = torch.from_numpy(node.bias)
        return module

    def map_linear(self, node: nir.Linear) -> layer.Linear:
        module = layer.Linear(node.weight.shape[-1], node.weight.shape[-2], bias=False)
        module.weight.data = torch.from_numpy(node.weight)
        return module

    def map_conv1d(self, node: nir.Conv1d) -> layer.Conv1d:
        if _to_python_value(node.groups) != 1:
            raise NotImplementedError(
                "NIR type inference does not support grouped convolutions."
            )
        weight = node.weight
        module = layer.Conv1d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=weight.shape[-1],
            stride=_to_python_value(node.stride),
            padding=_to_python_value(node.padding),
            dilation=_to_python_value(node.dilation),
            groups=1,
            bias=True,
        )
        module.weight.data = torch.from_numpy(weight)
        module.bias.data = torch.from_numpy(node.bias)
        return module

    def map_conv2d(self, node: nir.Conv2d) -> layer.Conv2d:
        weight = node.weight
        bias = node.bias
        stride = _to_python_value(node.stride)
        padding = _to_python_value(node.padding)
        dilation = _to_python_value(node.dilation)
        groups = _to_python_value(node.groups)
        if groups != 1:
            raise NotImplementedError(
                "NIR type inference does not support grouped convolutions."
            )

        module = layer.Conv2d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
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
        kernel_size = _to_python_value(node.kernel_size)
        stride = _to_python_value(node.stride)
        padding = _to_python_value(node.padding)

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

    def map_if(self, node: nir.IF) -> nn.Module:
        r = _uniform_scalar(node.r, "nir.IF.r")
        if not _matches_discretization(r, 1.0 / self.dt):
            raise ValueError("nir.IF.r must equal 1 / dt.")
        return _NIRStatefulModule(
            neuron.IFNode(
                v_threshold=_uniform_scalar(node.v_threshold, "nir.IF.v_threshold"),
                v_reset=_uniform_scalar(node.v_reset, "nir.IF.v_reset"),
            )
        )

    def map_lif(self, node: nir.LIF) -> nn.Module:
        tau = _uniform_scalar(node.tau, "nir.LIF.tau") / self.dt
        if not np.isfinite(tau) or tau <= 1.0:
            raise ValueError("nir.LIF.tau / dt must be finite and greater than 1.")
        r = _uniform_scalar(node.r, "nir.LIF.r")
        if _matches_discretization(r, 1.0):
            decay_input = True
        elif _matches_discretization(r, tau):
            decay_input = False
        else:
            raise ValueError("nir.LIF.r must equal 1 or tau / dt.")

        v_reset = _uniform_scalar(node.v_reset, "nir.LIF.v_reset")
        v_leak = _uniform_scalar(node.v_leak, "nir.LIF.v_leak")
        if not _matches_discretization(v_leak, v_reset):
            raise ValueError("nir.LIF.v_leak must equal v_reset.")

        return _NIRStatefulModule(
            neuron.LIFNode(
                tau=tau,
                decay_input=decay_input,
                v_reset=v_reset,
                v_threshold=_uniform_scalar(node.v_threshold, "nir.LIF.v_threshold"),
            )
        )

    def map_cuba_lif(self, node: nir.CubaLIF) -> nn.Module:
        tau_syn = _uniform_scalar(node.tau_syn, "nir.CubaLIF.tau_syn")
        tau_mem = _uniform_scalar(node.tau_mem, "nir.CubaLIF.tau_mem")
        if not np.isfinite(tau_syn) or not np.isfinite(tau_mem):
            raise ValueError("nir.CubaLIF time constants must be finite.")
        if tau_syn < self.dt or tau_mem < self.dt:
            raise ValueError("nir.CubaLIF time constants must be at least dt.")
        r = _uniform_scalar(node.r, "nir.CubaLIF.r")
        if not _matches_discretization(r, tau_mem / self.dt):
            raise ValueError("nir.CubaLIF.r must equal tau_mem / dt.")
        w_in = _uniform_scalar(node.w_in, "nir.CubaLIF.w_in")
        if not _matches_discretization(w_in, tau_syn / self.dt):
            raise ValueError("nir.CubaLIF.w_in must equal tau_syn / dt.")
        v_leak = _uniform_scalar(node.v_leak, "nir.CubaLIF.v_leak")
        if not _matches_discretization(v_leak, 0.0):
            raise ValueError("nir.CubaLIF.v_leak must equal 0.")

        return _NIRStatefulModule(
            neuron.CUBALIFNode(
                c_decay=1.0 - self.dt / tau_syn,
                v_decay=1.0 - self.dt / tau_mem,
                v_threshold=_uniform_scalar(
                    node.v_threshold, "nir.CubaLIF.v_threshold"
                ),
                v_reset=_uniform_scalar(node.v_reset, "nir.CubaLIF.v_reset"),
            )
        )


def import_from_nir(
    graph: Union[nir.NIRGraph, str, Path],
    dt: float = 1e-4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    step_mode: str = "s",
) -> fx.GraphModule:
    """
    **API Language** - :ref:`中文 <import_from_nir-cn>` | :ref:`English <import_from_nir-en>`

    ----

    .. _import_from_nir-cn:

    * **中文**

    将 `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_ 图
    转换为 SpikingJelly 神经网络模型。函数会根据 NIR 节点类型自动映射为对应的
    SpikingJelly 模块（如 Linear、Conv、IF/LIF/CubaLIF 神经元等），并返回可直接运行的
    ``fx.GraphModule`` 对象。模型前向返回 ``(output, state)``；传入 ``state=None``
    会从初始状态运行，连续运行时应将上一次返回的 ``state`` 传回模型。

    :param graph: NIR 图，或存储 NIR 图的 HDF5 文件路径
    :type graph: Union[nir.NIRGraph, str, Path]

    :param dt: 网络时间步长，单位为秒，用于重构 IF/LIF 节点的时间常量等超参数。默认值为 ``1e-4``，与大多数兼容 NIR 的框架一致
    :type dt: float

    :param device: 模型运行设备，如 ``'cpu'`` 或 ``'cuda'``
    :type device: str

    :param dtype: 模型张量数据类型，通常为 ``torch.float32`` 或 ``torch.float64``
    :type dtype: torch.dtype

    :param step_mode: 步进模式，可选 ``'s'`` (单步) 或 ``'m'`` (多步)。NIR 图将首先转换到单步模式的 SpikingJelly 模型，
        随后统一改变模型中所有子模块的步进模式。循环图仅支持 ``'s'``
    :type step_mode: str

    :return: 转换得到的 ``fx.GraphModule`` 对象
    :rtype: torch.fx.GraphModule
    :raises ValueError: ``dt`` 非正、``step_mode`` 非法，或 NIR 神经元参数非均匀、不符合 SpikingJelly 离散化约束
    :raises NotImplementedError: NIR 图包含 grouped convolution，或循环图使用多步模式

    ----

    .. _import_from_nir-en:

    * **English**

    Convert a `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_
    graph to a SpikingJelly model. The function automatically maps NIR nodes to
    corresponding SpikingJelly modules (e.g., Linear, convolution, and
    IF/LIF/CubaLIF neurons) and returns a runnable
    :class:`fx.GraphModule <https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule>`.
    Its forward pass returns ``(output, state)``. Passing ``state=None`` starts
    from the initial state; pass the previously returned ``state`` to continue.

    :param graph: NIR graph, or the path to the HDF5 file storing the NIR graph
    :type graph: Union[nir.NIRGraph, str, Path]

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
        Then, all the submodules will be set to the specified step mode. Recurrent
        graphs support only ``'s'``.
    :type step_mode: str

    :return: the converted SpikingJelly ``fx.GraphModule`` object
    :rtype: torch.fx.GraphModule
    :raises ValueError: If ``dt`` is not positive, ``step_mode`` is invalid, or
        NIR neuron parameters are non-uniform or incompatible with SpikingJelly
        discretization
    :raises NotImplementedError: If the NIR graph contains grouped convolution,
        or a recurrent graph uses multi-step mode
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if step_mode not in ("s", "m"):
        raise ValueError("step_mode must be 's' or 'm'.")

    source_is_path = isinstance(graph, (str, Path))
    if source_is_path:
        graph = nir.read(graph)
    if step_mode == "m" and _has_cycle(graph):
        raise NotImplementedError("Recurrent NIR graphs require step_mode='s'.")
    mapper = _NodeMapper(dt=dt)

    gm = nirtorch.nir_to_torch(graph, mapper.map_dict, device=device, dtype=dtype)
    functional.set_step_mode(gm, step_mode)
    logger.info(
        "Import completed: source={} device={} dtype={} step_mode={} modules={}",
        "path" if source_is_path else type(graph).__name__,
        device,
        dtype,
        step_mode,
        sum(1 for _ in gm.modules()),
    )
    return gm
