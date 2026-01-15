from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp
import nir
import nirtorch

from .. import layer, neuron


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
        tracer = nirtorch.torch_tracer.NIRTorchTracer(self.map_dict.keys())
        graph = tracer.trace(self.net)
        gm = fx.GraphModule(tracer.root, graph)
        ShapeProp(gm).propagate(example_input)

        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue
            if "tensor_meta" not in node.meta:
                continue

            module = gm.get_submodule(node.target)
            output_shape = node.meta["tensor_meta"].shape

            input_shapes = []
            for in_node in node.all_input_nodes:
                if "tensor_meta" in in_node.meta:
                    input_shapes.append(in_node.meta["tensor_meta"].shape)
            input_shape = input_shapes[0]  # most modules has only one input

            self.module_io_shape[module] = {
                "input_shape": input_shape,
                "output_shape": output_shape,
            }

    @property
    def map_dict(self) -> dict:
        return {
            nn.Linear: self.map_linear,
            layer.Linear: self.map_linear,
            nn.Conv2d: self.map_conv2d,
            layer.Conv2d: self.map_conv2d,
            nn.AvgPool2d: self.map_avgpool2d,
            layer.AvgPool2d: self.map_avgpool2d,
            nn.Flatten: self.map_flatten,
            layer.Flatten: self.map_flatten,
            neuron.IFNode: self.map_if,
            neuron.LIFNode: self.map_lif,
            neuron.ParametricLIFNode: self.map_plif,
        }

    def map(self, module: nn.Module) -> nir.NIRNode:
        return self.map_dict[module.__class__](module)

    def map_linear(self, module: nn.Linear) -> nir.NIRNode:
        if module.bias is None:
            return nir.Linear(_to_numpy(module.weight))
        else:
            return nir.Affine(_to_numpy(module.weight), _to_numpy(module.bias))

    def map_conv2d(self, module: nn.Conv2d) -> nir.Conv2d:
        if module.bias is None:
            bias = np.zeros((module.weight.shape[0]))
        else:
            bias = _to_numpy(module.bias)

        H, W = self.module_io_shape[module]["input_shape"][-2:]

        return nir.Conv2d(
            input_shape=(H, W),
            weight=_to_numpy(module.weight),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=bias,
        )

    def map_avgpool2d(self, module: nn.AvgPool2d) -> nir.NIRNode:
        return nir.AvgPool2d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
        )

    def map_flatten(self, module: nn.Flatten) -> nir.Flatten:
        start_dim, end_dim = module.start_dim, module.end_dim
        start_dim = start_dim - 1 if start_dim > 0 else start_dim
        end_dim = end_dim - 1 if end_dim > 0 else end_dim

        input_shape = self.module_io_shape[module]["input_shape"]
        input_type_start = 1
        if hasattr(module, "step_mode") and module.step_mode == "m":
            input_type_start = 2

        return nir.Flatten(
            input_type=input_shape[input_type_start:],  # remove the T and B dims
            start_dim=start_dim,
            end_dim=end_dim,
        )

    def map_if(self, module: neuron.IFNode) -> nir.IF:
        """
        .. warning::

            `nir.IF` does not distinguish soft reset from hard reset. If
            `module.v_reset=None` (i.e. soft reset), it will be converted to
            `module.v_reset=0` (i.e. hard reset with 0 reset potential).
        """
        v_reset = module.v_reset
        v_threshold = module.v_threshold

        r = 1 / self.dt
        v_reset_ = 0.0 if v_reset is None else v_reset

        input_shape = self.module_io_shape[module]["input_shape"]
        output_shape = self.module_io_shape[module]["output_shape"]
        type_start = 1 if module.step_mode == "s" else 2
        input_type = input_shape[type_start:]
        output_type = output_shape[type_start:]  # remove the T and B dims

        return nir.IF(
            r=np.full(input_type, r),
            v_threshold=np.full(input_type, v_threshold),
            v_reset=np.full(input_type, v_reset_),
            input_type=input_type,
            output_type=output_type,
        )

    def map_lif(self, module: neuron.LIFNode) -> nir.LIF:
        """
        .. warning::

            `nir.LIF` does not distinguish soft reset from hard reset. If
            `module.v_reset=None` (i.e. soft reset), it will be converted to
            `module.v_reset=0` (i.e. hard reset with 0 reset potential).
        """
        tau = module.tau
        v_reset = module.v_reset
        v_threshold = module.v_threshold
        decay_input = module.decay_input

        tau_ = tau * self.dt
        r = 1.0 if decay_input else tau
        v_leak = 0.0 if v_reset is None else v_reset
        v_reset_ = 0.0 if v_reset is None else v_reset

        input_shape = self.module_io_shape[module]["input_shape"]
        output_shape = self.module_io_shape[module]["output_shape"]
        type_start = 1 if module.step_mode == "s" else 2
        input_type = input_shape[type_start:]
        output_type = output_shape[type_start:]  # remove the T and B dims

        return nir.LIF(
            tau=np.full(input_type, tau_),
            r=np.full(input_type, r),
            v_leak=np.full(input_type, v_leak),
            v_threshold=np.full(input_type, v_threshold),
            v_reset=np.full(input_type, v_reset_),
            input_type=input_type,
            output_type=output_type,
        )

    def map_plif(self, module: neuron.ParametricLIFNode) -> nir.LIF:
        """
        .. warning::

            `nir.LIF` does not distinguish soft reset from hard reset. If
            `module.v_reset=None` (i.e. soft reset), it will be converted to
            `module.v_reset=0` (i.e. hard reset with 0 reset potential).
        """
        with torch.no_grad():
            tau = 1.0 / module.w.sigmoid()
        v_reset = module.v_reset
        v_threshold = module.v_threshold
        decay_input = module.decay_input

        tau_ = tau * self.dt
        r = 1.0 if decay_input else tau
        v_leak = 0.0 if v_reset is None else v_reset
        v_reset_ = 0.0 if v_reset is None else v_reset

        input_shape = self.module_io_shape[module]["input_shape"]
        output_shape = self.module_io_shape[module]["output_shape"]
        type_start = 1 if module.step_mode == "s" else 2
        input_type = input_shape[type_start:]
        output_type = output_shape[type_start:]  # remove the T and B dims

        return nir.LIF(
            tau=np.full(input_type, tau_),
            r=np.full(input_type, r),
            v_leak=np.full(input_type, v_leak),
            v_threshold=np.full(input_type, v_threshold),
            v_reset=np.full(input_type, v_reset_),
            input_type=input_type,
            output_type=output_type,
        )


def export_to_nir(
    net: nn.Module,
    example_input: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    dt: float = 1e-4,
):
    """
    **API Language:**
    :ref:`中文 <export_to_nir-cn>` | :ref:`English <export_to_nir-en>`

    ----

    .. _export_to_nir-cn:

    * **中文**

    将 SpikingJelly 的模型转换为 `NIR（Neuromorphic Intermediate Representation） <https://neuroir.org/docs/index.html>`_ 图，
    以供后续转换到其它框架或部署到神经形态芯片上。本函数会自动通过示例输入 ``example_input``
    推导每个模块的输入输出形状，将 SpikingJelly 或 PyTorch 模块转换为对应的 NIR 节点。

    :param net: 需要转换的 SpikingJelly / PyTorch 模型
    :type net: torch.nn.Module

    :param example_input: 用于推导 ``net`` 中各个子模块输入输出形状的示例输入张量
    :type example_input: torch.Tensor

    :param save_path: 转换后的 NIR 图保存路径。如果不为 ``None``，函数会将 NIR 图写入指定的
        HDF5 文件。默认为 ``None`` ，即不保存 NIR 图
    :type save_path: Optional[Union[str, Path]]

    :param dt: 网络时间步长，单位为秒，用于计算 NIR 神经元节点的时间常量等超参数。默认值为 ``1e-4``，
        与大多数兼容 NIR 的框架一致
    :type dt: float

    :return: 转换得到的 NIRGraph 对象
    :rtype: nir.NIRGraph

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
        output shapes of each submodule in ``net``
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
    """
    mapper = _ModuleMapper(net, example_input, dt=dt)

    graph = nirtorch.torch_to_nir(net, mapper.map_dict, type_check=True)

    if save_path is not None:
        nir.write(save_path, graph)
    return graph
