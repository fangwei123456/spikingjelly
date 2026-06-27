from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import fx

from spikingjelly.activation_based.ann2snn.modules import (
    ChannelVoltageScaler,
    VoltageScaler,
)
from spikingjelly.activation_based.neuron.base_node import BaseNode


Scaler = Union[VoltageScaler, ChannelVoltageScaler]
_MIN_READOUT_STEPS = 4


def _reset_snn(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "reset"):
            module.reset()


def _extract_batch_input(batch):
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError(
        "estimate_delay_start supports dataloaders that yield a tensor or "
        "a non-empty tuple/list whose first element is the model input."
    )


def _as_runtime_tensor(value, x: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=x.device, dtype=x.dtype)
    return torch.as_tensor(value, device=x.device, dtype=x.dtype)


def _scale_view(scaler: Scaler, x: torch.Tensor) -> torch.Tensor:
    if isinstance(scaler, ChannelVoltageScaler):
        return scaler._view_scale(x).to(device=x.device, dtype=x.dtype)
    return scaler.scale.to(device=x.device, dtype=x.dtype)


def _channel_dim(scaler: Scaler) -> Optional[int]:
    if isinstance(scaler, ChannelVoltageScaler):
        return scaler.channel_dim
    return None


def _compute_delay_ratio(
    module: BaseNode,
    post_scaler: Scaler,
    x: torch.Tensor,
) -> torch.Tensor:
    v_threshold = _as_runtime_tensor(module.v_threshold, x)
    if v_threshold.numel() != 1:
        raise ValueError("Delay estimation requires scalar neuron v_threshold.")
    if not torch.isfinite(v_threshold).all() or (v_threshold <= 0).any():
        raise ValueError("Delay estimation requires finite positive v_threshold.")

    try:
        v_init = module.get_reset_value("v")
    except (KeyError, AttributeError):
        v_init = getattr(module, "v_reset", 0.0)
    if v_init is None:
        v_init = 0.0
    v_init = _as_runtime_tensor(v_init, x)

    scale = _scale_view(post_scaler, x)
    x_nonnegative = torch.clamp(x.detach(), min=0)
    original_activation = x_nonnegative * scale / v_threshold
    required_charge = torch.clamp(v_threshold - v_init, min=0)
    original_required_charge = required_charge * scale / v_threshold

    channel_dim = _channel_dim(post_scaler)
    if channel_dim is None or scale.dim() == 0:
        max_mean_activation = original_activation.mean()
    else:
        if channel_dim < 0:
            channel_dim += original_activation.dim()
        if channel_dim < 0 or channel_dim >= original_activation.dim():
            raise ValueError("channel_dim is out of range for delay estimation.")
        reduce_dims = tuple(
            dim for dim in range(original_activation.dim()) if dim != channel_dim
        )
        max_mean_activation = original_activation.mean(dim=reduce_dims).max()

    if max_mean_activation <= 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (original_required_charge.mean() / max_mean_activation).detach()


def _find_scaler_neuron_scaler_paths(
    model: nn.Module,
) -> List[Tuple[BaseNode, Scaler]]:
    if not isinstance(model, fx.GraphModule):
        return []

    modules = dict(model.named_modules())
    paths: List[Tuple[BaseNode, Scaler]] = []
    for node in model.graph.nodes:
        if node.op != "call_module" or not isinstance(node.target, str):
            continue
        module = modules.get(node.target)
        if not isinstance(module, BaseNode):
            continue
        if len(node.args) != 1 or not isinstance(node.args[0], fx.Node):
            continue
        pre_node = node.args[0]
        if pre_node.op != "call_module" or not isinstance(pre_node.target, str):
            continue
        pre_module = modules.get(pre_node.target)
        if not isinstance(pre_module, (VoltageScaler, ChannelVoltageScaler)):
            continue

        post_scalers = []
        for user in node.users:
            if user.op != "call_module" or not isinstance(user.target, str):
                continue
            user_module = modules.get(user.target)
            if isinstance(user_module, (VoltageScaler, ChannelVoltageScaler)):
                post_scalers.append(user_module)
        if len(post_scalers) == 1:
            paths.append((module, post_scalers[0]))
    return paths


def estimate_delay_start(
    model: nn.Module,
    dataloader: Iterable,
    device: Union[str, torch.device],
    time_steps: int,
    num_batches: int = 1,
) -> int:
    r"""
    **API Language** - :ref:`中文 <estimate_delay_start-cn>` | :ref:`English <estimate_delay_start-en>`

    ----

    .. _estimate_delay_start-cn:

    * **中文**

    估计 ANN2SNN 转换网络的 delayed-readout 起始时间步。该函数识别
    ``VoltageScaler/ChannelVoltageScaler -> BaseNode -> VoltageScaler/ChannelVoltageScaler``
    结构，为神经元临时注册 hook，在 probe 前向中估计每层从 reset 初始膜电位
    到有效放电所需的时间，并将各层估计值相加得到 ``delay_start``。

    该函数不会改变神经元公共 API，也不会改变正常 SNN 前向动力学。所有 hook
    都会在函数返回或异常抛出前移除。

    :param model: ANN2SNN 转换后的 FX ``GraphModule``。
    :type model: nn.Module
    :param dataloader: 校准数据加载器，batch 应为输入张量或首元素为输入张量的
        tuple/list。
    :type dataloader: Iterable
    :param device: 运行 probe forward 的设备。
    :type device: str or torch.device
    :param time_steps: SNN 总仿真步数。该参数仅用于裁剪返回值，确保 delayed
        readout 仍保留足够时间步。
    :type time_steps: int
    :param num_batches: 用于估计 delay 的校准 batch 数。
    :type num_batches: int
    :return: 可直接传给 delayed readout 的起始时间步。
    :rtype: int

    ----

    .. _estimate_delay_start-en:

    * **English**

    Estimate the delayed-readout start timestep for an ANN2SNN converted
    network. The function detects
    ``VoltageScaler/ChannelVoltageScaler -> BaseNode -> VoltageScaler/ChannelVoltageScaler``
    patterns, temporarily registers hooks on the neurons, estimates how many
    timesteps each layer needs to reach an effective spike output from its reset
    membrane potential, and sums the layer-wise estimates.

    This function does not change the public neuron API or normal SNN forward
    dynamics. All hooks are removed before returning or re-raising an exception.

    :param model: FX ``GraphModule`` converted by ANN2SNN.
    :type model: nn.Module
    :param dataloader: Calibration dataloader. Each batch should be an input
        tensor or a tuple/list whose first item is the input tensor.
    :type dataloader: Iterable
    :param device: Device for the probe forward pass.
    :type device: str or torch.device
    :param time_steps: Total SNN simulation steps. This parameter is used only
        to clamp the returned value so delayed readout still has enough
        timesteps.
    :type time_steps: int
    :param num_batches: Number of calibration batches used for delay estimation.
    :type num_batches: int
    :return: Start timestep for delayed readout.
    :rtype: int
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be positive.")
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    paths = _find_scaler_neuron_scaler_paths(model)
    if not paths:
        return 0

    model.eval().to(device)
    ratios: Dict[BaseNode, List[float]] = {module: [] for module, _ in paths}
    handles = []

    def make_hook(module: BaseNode, post_scaler: Scaler):
        def hook(_module, inputs, _output):
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("Delay estimation neuron hook expects one tensor input.")
            x = inputs[0]
            ratio = _compute_delay_ratio(module, post_scaler, x)
            ratios[module].append(float(ratio.detach().cpu().item()))
            max_value = float(_as_runtime_tensor(module.v_threshold, x).item())
            return torch.clamp(x, min=0, max=max_value)

        return hook

    try:
        for module, post_scaler in paths:
            handles.append(module.register_forward_hook(make_hook(module, post_scaler)))

        _reset_snn(model)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                x = _extract_batch_input(batch)
                if not isinstance(x, torch.Tensor):
                    raise TypeError("The extracted model input must be a tensor.")
                _reset_snn(model)
                model(x.to(device, non_blocking=True))
        delay = 0.0
        for values in ratios.values():
            if values:
                delay += sum(values) / len(values)
    finally:
        for handle in handles:
            handle.remove()
        _reset_snn(model)

    delay_start = int(math.ceil(delay))
    if time_steps < delay_start + _MIN_READOUT_STEPS:
        return max(time_steps - _MIN_READOUT_STEPS, 0)
    return min(delay_start, time_steps - 1)
