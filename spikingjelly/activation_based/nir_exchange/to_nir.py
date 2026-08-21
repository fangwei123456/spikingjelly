import copy
from pathlib import Path
from typing import Optional, Union

import nir
import nirtorch
import numpy as np
import torch
import torch.nn as nn
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._pytree import tree_map

from .. import base, layer, neuron
from ...logger import logger


__all__ = ["export_to_nir"]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class _ModuleMapper:
    def __init__(
        self,
        net: nn.Module,
        example_input: torch.Tensor,
        dt: float = 1e-4,
    ):
        self.dt = dt
        self.net = net
        self.module_io_shape = {}
        self.set_module_io_shape(example_input)

    def set_module_io_shape(self, example_input: torch.Tensor):
        memories = tree_map(
            lambda value: (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            ),
            base.extract_memories(self.net),
        )
        try:
            tracer = nirtorch.torch_tracer.NIRTorchTracer(self.map_dict.keys())
            graph = tracer.trace(self.net)
            gm = fx.GraphModule(tracer.root, graph)
            with torch.no_grad():
                ShapeProp(gm).propagate(example_input)
        finally:
            base.load_memories(self.net, memories)

        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue
            if "tensor_meta" not in node.meta:
                continue

            module = gm.get_submodule(node.target)
            input_shapes = []
            for in_node in node.all_input_nodes:
                if "tensor_meta" in in_node.meta:
                    input_shapes.append(in_node.meta["tensor_meta"].shape)
            input_shape = input_shapes[0]  # most modules has only one input
            self.module_io_shape[module] = input_shape

    @property
    def map_dict(self) -> dict:
        return {
            nn.Linear: self.map_linear,
            layer.Linear: self.map_linear,
            nn.Conv1d: self.map_conv1d,
            layer.Conv1d: self.map_conv1d,
            nn.Conv2d: self.map_conv2d,
            layer.Conv2d: self.map_conv2d,
            nn.AvgPool2d: self.map_avgpool2d,
            layer.AvgPool2d: self.map_avgpool2d,
            nn.Flatten: self.map_flatten,
            layer.Flatten: self.map_flatten,
            neuron.IFNode: self.map_if,
            neuron.LIFNode: self.map_lif,
            neuron.ParametricLIFNode: self.map_plif,
            neuron.CUBALIFNode: self.map_cuba_lif,
        }

    def map_linear(self, module: nn.Linear) -> nir.NIRNode:
        if module.bias is None:
            return nir.Linear(_to_numpy(module.weight))
        return nir.Affine(_to_numpy(module.weight), _to_numpy(module.bias))

    @staticmethod
    def _conv_bias(module: nn.Module) -> np.ndarray:
        if module.bias is None:
            return np.zeros(
                module.weight.shape[0], dtype=_to_numpy(module.weight).dtype
            )
        return _to_numpy(module.bias)

    @staticmethod
    def _require_ungrouped(module: nn.Module) -> None:
        if module.groups != 1:
            raise NotImplementedError(
                "NIR type inference does not support grouped convolutions."
            )
        if module.padding_mode != "zeros":
            raise NotImplementedError("NIR supports only zero-padded convolutions.")

    def map_conv1d(self, module: nn.Conv1d) -> nir.Conv1d:
        self._require_ungrouped(module)
        padding = (
            module.padding if isinstance(module.padding, str) else module.padding[0]
        )
        return nir.Conv1d(
            input_shape=self.module_io_shape[module][-1],
            weight=_to_numpy(module.weight),
            stride=module.stride[0],
            padding=padding,
            dilation=module.dilation[0],
            groups=module.groups,
            bias=self._conv_bias(module),
        )

    def map_conv2d(self, module: nn.Conv2d) -> nir.Conv2d:
        self._require_ungrouped(module)
        height, width = self.module_io_shape[module][-2:]
        return nir.Conv2d(
            input_shape=(height, width),
            weight=_to_numpy(module.weight),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=self._conv_bias(module),
        )

    def map_avgpool2d(self, module: nn.AvgPool2d) -> nir.NIRNode:
        if (
            module.ceil_mode
            or (np.any(module.padding) and not module.count_include_pad)
            or module.divisor_override is not None
        ):
            raise NotImplementedError(
                "NIR does not represent this AvgPool2d configuration."
            )
        return nir.AvgPool2d(
            kernel_size=module.kernel_size,
            stride=module.kernel_size if module.stride is None else module.stride,
            padding=module.padding,
        )

    def map_flatten(self, module: nn.Flatten) -> nir.Flatten:
        start_dim, end_dim = module.start_dim, module.end_dim
        start_dim = start_dim - 1 if start_dim > 0 else start_dim
        end_dim = end_dim - 1 if end_dim > 0 else end_dim

        input_shape = self.module_io_shape[module]
        input_type_start = 1
        if hasattr(module, "step_mode") and module.step_mode == "m":
            input_type_start = 2

        return nir.Flatten(
            input_type=input_shape[input_type_start:],  # remove the T and B dims
            start_dim=start_dim,
            end_dim=end_dim,
        )

    def _neuron_shape(self, module: nn.Module) -> torch.Size:
        type_start = 1 if module.step_mode == "s" else 2
        return self.module_io_shape[module][type_start:]

    @staticmethod
    def _hard_reset(module: neuron.BaseNode) -> float:
        if module.v_reset is None:
            raise NotImplementedError("NIR does not distinguish soft reset.")
        return module.v_reset

    def map_if(self, module: neuron.IFNode) -> nir.IF:
        shape = self._neuron_shape(module)
        v_reset = self._hard_reset(module)

        return nir.IF(
            r=np.full(shape, 1.0 / self.dt),
            v_threshold=np.full(shape, module.v_threshold),
            v_reset=np.full(shape, v_reset),
        )

    def map_lif(self, module: neuron.LIFNode) -> nir.LIF:
        tau = module.tau
        v_reset = self._hard_reset(module)
        shape = self._neuron_shape(module)

        return nir.LIF(
            tau=np.full(shape, tau * self.dt),
            r=np.full(shape, 1.0 if module.decay_input else tau),
            v_leak=np.full(shape, v_reset),
            v_threshold=np.full(shape, module.v_threshold),
            v_reset=np.full(shape, v_reset),
        )

    def map_plif(self, module: neuron.ParametricLIFNode) -> nir.LIF:
        with torch.no_grad():
            tau = float((1.0 / module.w.sigmoid()).detach().cpu())
        v_reset = self._hard_reset(module)
        shape = self._neuron_shape(module)

        return nir.LIF(
            tau=np.full(shape, tau * self.dt),
            r=np.full(shape, 1.0 if module.decay_input else tau),
            v_leak=np.full(shape, v_reset),
            v_threshold=np.full(shape, module.v_threshold),
            v_reset=np.full(shape, v_reset),
        )

    def map_cuba_lif(self, module: neuron.CUBALIFNode) -> nir.CubaLIF:
        if not 0.0 <= module.c_decay < 1.0:
            raise ValueError("CUBALIFNode.c_decay must be in [0, 1).")
        if not 0.0 <= module.v_decay < 1.0:
            raise ValueError("CUBALIFNode.v_decay must be in [0, 1).")

        shape = self._neuron_shape(module)
        v_reset = self._hard_reset(module)
        tau_syn = self.dt / (1.0 - module.c_decay)
        tau_mem = self.dt / (1.0 - module.v_decay)
        return nir.CubaLIF(
            tau_syn=np.full(shape, tau_syn),
            tau_mem=np.full(shape, tau_mem),
            r=np.full(shape, tau_mem / self.dt),
            v_leak=np.zeros(shape),
            v_threshold=np.full(shape, module.v_threshold),
            v_reset=np.full(shape, v_reset),
            w_in=np.full(shape, tau_syn / self.dt),
        )


def export_to_nir(
    net: nn.Module,
    example_input: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    dt: float = 1e-4,
) -> nir.NIRGraph:
    """
    **API Language** - :ref:`中文 <export_to_nir-cn>` | :ref:`English <export_to_nir-en>`

    ----

    .. _export_to_nir-cn:

    * **中文**

    将 SpikingJelly 的模型转换为 `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_ 图，
    以供后续转换到其它框架或部署到神经形态芯片上。本函数会自动通过示例输入 ``example_input``
    推导每个模块的输入输出形状，将 SpikingJelly 或 PyTorch 模块转换为对应的 NIR 节点。

    :param net: 需要转换的 SpikingJelly / PyTorch 模型
    :type net: torch.nn.Module

    :param example_input: 用于推导 ``net`` 中各个子模块输入输出形状的示例输入张量。
        其 dtype、device、批量维和可选时间维应与 ``net`` 的实际输入一致
    :type example_input: torch.Tensor

    :param save_path: 转换后的 NIR 图保存路径。如果不为 ``None``，函数会将 NIR 图写入指定的
        HDF5 文件。默认为 ``None`` ，即不保存 NIR 图
    :type save_path: Optional[Union[str, Path]]

    :param dt: 网络时间步长，单位为秒，用于计算 NIR 神经元节点的时间常量等超参数。默认值为 ``1e-4``，
        与大多数兼容 NIR 的框架一致
    :type dt: float

    :return: 转换得到的 NIRGraph 对象
    :rtype: nir.NIRGraph
    :raises ValueError: ``dt`` 非正，或 CUBALIF 衰减参数超出可转换范围
    :raises NotImplementedError: 模型使用 soft reset、grouped convolution 或 NIR 无法表示的层配置

    ----

    .. _export_to_nir-en:

    * **English**

    Convert a SpikingJelly model to a `NIR (Neuromorphic Intermediate Representation) <https://neuroir.org/docs/index.html>`_ graph
    for conversion to other frameworks or deployment on neuromorphic hardware.
    This function automatically infers the input and output shapes of each submodule
    using ``example_input``, and converts SpikingJelly or PyTorch modules to the
    corresponding NIR nodes.

    :param net: the SpikingJelly / PyTorch model to convert
    :type net: torch.nn.Module

    :param example_input: an example input tensor used to infer the input and
        output shapes of each submodule in ``net``. Its dtype, device, batch
        dimension, and optional time dimension must match the actual model input
    :type example_input: torch.Tensor

    :param save_path: the path to save the converted NIR graph. If not ``None``,
        the NIR graph will be written to the specified HDF5 file. Defaults to
        `None`, which means the NIR graph will not be saved
    :type save_path: Optional[Union[str, Path]]

    :param dt: simulation time step in seconds, used to compute time constants
        and other hyperparameters for NIR neuron nodes. The default value is ``1e-4``,
        consistent with other frameworks that support NIR
    :type dt: float

    :return: the converted NIRGraph object
    :rtype: nir.NIRGraph
    :raises ValueError: If ``dt`` is not positive or CUBALIF decay parameters
        cannot be represented
    :raises NotImplementedError: If the model uses soft reset, grouped convolution,
        or another layer configuration that NIR cannot represent
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    mapper = _ModuleMapper(net, example_input, dt=dt)

    graph = nirtorch.torch_to_nir(net, mapper.map_dict, type_check=True)

    if save_path is not None:
        nir.write(save_path, graph)
    logger.info(
        "Export completed: model={} device={} dtype={} dt={} save_path={}",
        type(net).__name__,
        example_input.device,
        example_input.dtype,
        dt,
        save_path,
    )
    return graph
