"""PLIF experiment with device-local implementation selection."""

from typing import Optional

import torch

from ..dispatch import _register_dispatch, _update_cache_tag
from ..selection import _CudaSelection
from . import cpu as _cpu

_selection = _CudaSelection(
    __name__,
    "sj_plif",
    "SJ_PLIF_CUDA_IMPLEMENTATION",
    _cpu._forward_impl,
    _cpu._backward_impl,
    on_select=_update_cache_tag,
)
_dispatch = _register_dispatch("sj_plif", _cpu, _selection)
_forward = torch.ops.sj_plif.forward.default


def get_cuda_implementation(device: torch.device) -> dict[str, object]:
    r"""
    **API Language** - :ref:`中文 <experimental-plif-implementation-cn>` | :ref:`English <experimental-plif-implementation-en>`

    ----

    .. _experimental-plif-implementation-cn:

    * **中文**

    初始化指定 CUDA 设备的实验 PLIF 实现并返回诊断快照；后续调用复用该选择。
    默认优先级为原生 CUDA、Triton、CuPy。环境变量
    ``SJ_PLIF_CUDA_IMPLEMENTATION`` 在导入本模块时读取，改变选择须重启进程。

    :param device: NVIDIA CUDA 设备；省略索引时使用当前设备。
    :type device: torch.device
    :return: ``implementation`` 名称及 ``unavailable`` 候选失败原因字典的副本。
    :rtype: dict[str, object]
    :raises ValueError: 设备不是 CUDA 或环境变量的实现名无效。
    :raises RuntimeError: 没有可用实现，或当前为 ROCm。

    ----

    .. _experimental-plif-implementation-en:

    * **English**

    Initialize experimental PLIF for a CUDA device and return a diagnostic snapshot.
    Later calls reuse the selection. The default order is native CUDA, Triton, then
    CuPy. ``SJ_PLIF_CUDA_IMPLEMENTATION`` is read at module import; changing the
    selection requires a new process.

    :param device: NVIDIA CUDA device; an omitted index selects the current device.
    :type device: torch.device
    :return: Selected ``implementation`` and a copy of ``unavailable`` reasons.
    :rtype: dict[str, object]
    :raises ValueError: Device is not CUDA or the configured implementation is invalid.
    :raises RuntimeError: No implementation is available, or this is ROCm.
    """
    return _selection.diagnostics(device)


def plif(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool = True,
    threshold: float = 1.0,
    reset: Optional[float] = 0.0,
    detach_reset: bool = False,
    alpha: float = 4.0,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <experimental-plif-op-cn>` | :ref:`English <experimental-plif-op-en>`

    ----

    .. _experimental-plif-op-cn:

    * **中文**

    显式状态的实验多步 PLIF，使用可选替代梯度且仅支持一阶反向。
    不修改输入或状态。CUDA 设备首次使用时选择实现，后续复用；编译前先预热。
    非连续输入在实现入口转为连续存储；输出为互不别名的连续张量。

    :param x: CPU/NVIDIA CUDA FP32/FP16/BF16 输入 ``[T, ...]``，T >= 1，神经元维度非空。
    :type x: torch.Tensor
    :param v: 初态，形状为 ``x.shape[1:]``，使用 FP32 且 device 与 x 相同。
    :type v: torch.Tensor
    :param w: 可学习的零维 FP32/FP16/BF16 标量，与输入同设备；其 FP32 Sigmoid 为倒数时间常数，归约后梯度转为 w dtype。
        有限值及正负无穷均按 Torch Sigmoid 计算，饱和值 0/1 受支持。
    :type w: torch.Tensor
    :param decay_input: 是否对输入乘以 ``sigmoid(w)``，默认 True。
    :type decay_input: bool
    :param threshold: 有限发放阈值，默认 1。
    :type threshold: float
    :param reset: 有限硬重置电位，默认 0；None 表示软重置。
    :type reset: Optional[float]
    :param detach_reset: 是否分离重置分支的脉冲梯度，默认 False。
    :type detach_reset: bool
    :param alpha: 有限正替代梯度参数 alpha，默认 4。
    :type alpha: float
    :param store_v_seq: 默认 True 返回完整电位轨迹；False 只返回最终状态，形状与 v 相同。
    :type store_v_seq: bool
    :param surrogate_id: 替代梯度编号，默认 0：0 Sigmoid、1 ATan、2 PiecewiseQuadratic、
        3 PiecewiseExp、4 SoftSign、5 SuperSpike、6 Erf。计算使用 FP32。
    :type surrogate_id: int
    :return: 脉冲、电位输出及不可微分的充电电位 workspace。脉冲和 workspace 与 x 同形状；
        电位为完整轨迹（True）或与 v 同形状的最终状态（False）；电位和 workspace 使用 FP32，脉冲跟随 x dtype，所有输出与 x 同 device。
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: 标量参数无效或实现配置无效。
    :raises RuntimeError: 输入不满足约束或设备没有可用实现。

    ----

    .. _experimental-plif-op-en:

    * **English**

    Experimental explicit-state multi-step PLIF with a selected surrogate and first-order
    reverse-mode gradients only. Inputs are not mutated. CUDA selection occurs once
    per device; warm up before compilation. Noncontiguous inputs are made contiguous
    at the implementation boundary; outputs are contiguous and do not alias.

    :param x: CPU/NVIDIA CUDA FP32/FP16/BF16 ``[T, ...]`` input, T >= 1, with nonempty neuron dimensions.
    :type x: torch.Tensor
    :param v: Initial state shaped ``x.shape[1:]``, in FP32 on the input device.
    :type v: torch.Tensor
    :param w: Learnable zero-dimensional FP32/FP16/BF16 scalar on the input device. Its
        FP32 Sigmoid is the reciprocal time constant; the reduced gradient is cast to w dtype. Finite values and positive/negative
        infinity follow Torch Sigmoid, including saturation at 0/1.
    :type w: torch.Tensor
    :param decay_input: Scale the input by ``sigmoid(w)``; default True.
    :type decay_input: bool
    :param threshold: Finite firing threshold; default 1.
    :type threshold: float
    :param reset: Finite hard-reset voltage, default 0; None selects soft reset.
    :type reset: Optional[float]
    :param detach_reset: Detach spikes in the reset branch; default False.
    :type detach_reset: bool
    :param alpha: Finite positive surrogate alpha; default 4.
    :type alpha: float
    :param store_v_seq: Return the full voltage trace (default True), or only the
        final state with the shape of v when False.
    :type store_v_seq: bool
    :param surrogate_id: Surrogate ID, default 0: 0 Sigmoid, 1 ATan, 2 PiecewiseQuadratic,
        3 PiecewiseExp, 4 SoftSign, 5 SuperSpike, 6 Erf. Arithmetic uses FP32.
    :type surrogate_id: int
    :return: Spikes, post-reset voltage trace, and nondifferentiable charged-voltage
        workspace. Spikes/workspace have the input shape; voltage is the full trace
        (True) or the final state shaped like v (False). Voltage/workspace are FP32; spikes follow x dtype. All share the input device.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: Invalid scalar parameters or implementation configuration.
    :raises RuntimeError: Unsupported input or no implementation for the device.
    """
    return _forward(
        x,
        v,
        w,
        decay_input,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
