"""Opt-in Nsight Systems ranges for SNN training and inference scripts."""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from spikingjelly.activation_based.base import StepModule

__all__ = ["capture", "step", "region", "module_ranges"]


@contextmanager
def capture(enabled: bool = False) -> Iterator[None]:
    r"""
    **API Language** - :ref:`中文 <nsys-capture-cn>` | :ref:`English <nsys-capture-en>`

    ----

    .. _nsys-capture-cn:

    * **中文**

    在已启动的 CUDA 程序中限定一次 Nsight Systems 采集窗口。请在模型初始化和
    热身后进入此范围。启用时调用 CUDA profiler start/stop；关闭时不调用 CUDA。
    此函数不启动 Nsight Systems 进程，也不改变模型状态。

    :param enabled: 是否开启采集窗口，默认为 ``False``；开启时需要 CUDA。
    :type enabled: bool
    :raises RuntimeError: 开启后 CUDA profiler 无法启动或停止时抛出。

    ----

    .. _nsys-capture-en:

    * **English**

    Bound one Nsight Systems capture in an already launched CUDA program. Enter
    after model initialization and warmup. When enabled, call CUDA profiler
    start/stop; otherwise make no CUDA calls. This function does not launch
    Nsight Systems or change model state.

    :param enabled: Enable the capture window; defaults to ``False``. CUDA is
        required when enabled.
    :type enabled: bool
    :raises RuntimeError: If the CUDA profiler cannot start or stop when enabled.
    """
    if enabled:
        torch.cuda.profiler.start()
    try:
        yield
    finally:
        if enabled:
            torch.cuda.profiler.stop()


@contextmanager
def region(name: str, enabled: bool = False) -> Iterator[None]:
    r"""
    **API Language** - :ref:`中文 <nsys-region-cn>` | :ref:`English <nsys-region-en>`

    ----

    .. _nsys-region-cn:

    * **中文**

    在当前 CPU 线程上标记一个 NVTX 范围，适用于 forward、backward、reset
    等完整阶段。启用时需要 CUDA，退出范围时会配对关闭标记；关闭时无 CUDA 调用。
    不要用它逐个标记极短的神经元算子。

    :param name: 时间线中显示的阶段名称。
    :type name: str
    :param enabled: 是否写入 NVTX 标记，默认为 ``False``。
    :type enabled: bool
    :raises RuntimeError: 启用后 NVTX 范围无法写入时抛出。

    ----

    .. _nsys-region-en:

    * **English**

    Mark one NVTX range on the current CPU thread for a complete stage such as
    forward, backward, or reset. CUDA is required when enabled; the marker is
    paired on exit. Disabled ranges make no CUDA calls. Do not annotate every
    very short neuron operation.

    :param name: Stage name displayed on the timeline.
    :type name: str
    :param enabled: Emit an NVTX range; defaults to ``False``.
    :type enabled: bool
    :raises RuntimeError: If an enabled NVTX range cannot be emitted.
    """
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def step(index: int, phase: str, enabled: bool = False) -> Iterator[None]:
    r"""
    **API Language** - :ref:`中文 <nsys-step-cn>` | :ref:`English <nsys-step-en>`

    ----

    .. _nsys-step-cn:

    * **中文**

    标记一个完整训练或推理 step；范围名为 ``sj.step:<phase>:<index>``。
    与 :func:`capture` 配合时，一个采集窗口可包含多个 step。退出后不保留状态。

    :param index: 从零开始的采集窗口内 step 序号。
    :type index: int
    :param phase: ``"training"`` 或 ``"inference"``。
    :type phase: str
    :param enabled: 是否写入 NVTX 标记，默认为 ``False``。
    :type enabled: bool
    :return: 可用于 ``with`` 语句的 NVTX 范围。
    :rtype: Iterator[None]
    :raises ValueError: 当序号为负或阶段名称不受支持时抛出。

    ----

    .. _nsys-step-en:

    * **English**

    Mark a complete training or inference step with the name
    ``sj.step:<phase>:<index>``. Multiple steps can be placed inside one
    :func:`capture` window. No state is retained after exit.

    :param index: Zero-based step index within the capture window.
    :type index: int
    :param phase: ``"training"`` or ``"inference"``.
    :type phase: str
    :param enabled: Emit an NVTX range; defaults to ``False``.
    :type enabled: bool
    :return: An NVTX range usable with ``with``.
    :rtype: Iterator[None]
    :raises ValueError: If the index is negative or the phase is unsupported.
    """
    if index < 0 or phase not in ("training", "inference"):
        raise ValueError("step requires index >= 0 and phase training or inference")
    return region(f"sj.step:{phase}:{index}", enabled)


def _tensor_metadata(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "bytes": value.numel() * value.element_size(),
            "contiguous": value.is_contiguous(),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _tensor_metadata(item) for key, item in value.items()}
    return type(value).__name__


@contextmanager
def module_ranges(model: nn.Module, output: Path) -> Iterator[None]:
    r"""
    **API Language** - :ref:`中文 <nsys-module-ranges-cn>` | :ref:`English <nsys-module-ranges-en>`

    ----

    .. _nsys-module-ranges-cn:

    * **中文**

    在单独的 eager 诊断轮次为 ``StepModule``、SNN attention 及常见 Conv、
    Linear、BatchNorm、Pool 模块加入前向 NVTX 范围，并记录首次输入和输出的张量
    形状、stride、storage offset、dtype、device、字节数及连续性，以及模块的
    step mode 和 backend（若存在）。退出时移除钩子并写入 JSONL。
    Python 钩子会改变性能，不用于正式时延测量或 ``torch.compile`` 路径。

    :param model: 待检查的 SNN 模型；此上下文不自行执行前向，调用者在其中
        执行前向时可能改变 BatchNorm 统计量及神经元状态。
    :type model: nn.Module
    :param output: JSONL 输出路径；父目录不存在时创建。
    :type output: pathlib.Path
    :raises OSError: 输出文件无法写入时抛出。
    :raises RuntimeError: CUDA/NVTX 不可用时抛出。

    ----

    .. _nsys-module-ranges-en:

    * **English**

    Add forward NVTX ranges to ``StepModule``, SNN attention, and common Conv,
    Linear, BatchNorm, and Pool modules in a separate eager diagnostic pass. Record the
    first input and output tensor shapes, strides, storage offsets, dtypes,
    devices, byte sizes, and contiguity, plus module step modes and backends
    when present. Remove hooks and write JSONL on exit. Python hooks change
    performance and must not be used for formal timing or ``torch.compile``.

    :param model: SNN model to inspect. This context does not run a forward
        pass; a forward pass invoked by the caller may change BatchNorm
        statistics and neuron state.
    :type model: nn.Module
    :param output: JSONL output path; missing parent directories are created.
    :type output: pathlib.Path
    :raises OSError: If the output file cannot be written.
    :raises RuntimeError: If CUDA/NVTX is unavailable.
    """
    handles = []
    records = []
    seen = set()
    from spikingjelly.activation_based.layer import attention

    attention_types = (
        attention.TemporalWiseAttention,
        attention.MultiDimensionalAttention,
        attention.SpikingSelfAttention,
        attention.QKAttention,
        attention.SpikeDrivenSelfAttention,
    )
    layer_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
    )
    try:
        for name, module in model.named_modules():
            if not name or not isinstance(
                module, (StepModule, *attention_types, *layer_types)
            ):
                continue
            if isinstance(module, layer_types) and any(True for _ in module.children()):
                continue

            def before(_module, inputs, *, module_name=name):
                torch.cuda.nvtx.range_push(f"module:{module_name}")
                if module_name not in seen:
                    records.append(
                        {
                            "module": module_name,
                            "type": type(_module).__name__,
                            "step_mode": str(getattr(_module, "step_mode", "")) or None,
                            "backend": str(getattr(_module, "backend", "")) or None,
                            "event": "input",
                            "value": _tensor_metadata(inputs),
                        }
                    )

            def after(_module, _inputs, value, *, module_name=name):
                if module_name not in seen:
                    records.append(
                        {
                            "module": module_name,
                            "event": "output",
                            "value": _tensor_metadata(value),
                        }
                    )
                    seen.add(module_name)
                torch.cuda.nvtx.range_pop()

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
