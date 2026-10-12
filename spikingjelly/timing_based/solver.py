import math
from typing import Callable, Optional, Tuple

import torch

__all__ = ["first_spike_time"]

_Response = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


def _evaluate(response, time, piece, parameters):
    result = response(time, piece, *parameters)
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not all(isinstance(v, torch.Tensor) for v in result)
    ):
        raise TypeError("response must return a (residual, slope) tensor tuple.")
    value, slope = result
    if (
        value.shape != time.shape
        or slope.shape != time.shape
        or value.dtype != torch.float64
        or slope.dtype != torch.float64
        or value.device != time.device
        or slope.device != time.device
    ):
        raise ValueError("response must return matching FP64 residuals and slopes.")
    if not bool(torch.isfinite(value).all() & torch.isfinite(slope).all()):
        raise RuntimeError("response returned a non-finite residual or slope.")
    return value, slope


class _FirstSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, response, boundaries, extrema, *parameters):
        ctx.set_materialize_grads(False)
        knots = boundaries.double()
        state = tuple(p.double() if p.is_floating_point() else p for p in parameters)
        lower, upper = knots[..., :-1], knots[..., 1:]
        pieces = torch.arange(lower.shape[-1], device=knots.device).expand_as(lower)
        left, _ = _evaluate(response, lower, pieces, state)
        right, _ = _evaluate(response, upper, pieces, state)
        eps = torch.finfo(torch.float64).eps
        roundoff = 64 * eps * (1 + left.abs() + right.abs())
        reaches = torch.where(extrema[..., 1], right > roundoff, right >= 0)
        bracket = (upper > lower) & (
            ((left < 0) & reaches) | ((left == 0) & (right > 0) & ~extrema[..., 0])
        )
        firing = bracket.any(-1, keepdim=True)
        chosen = bracket.to(torch.int8).argmax(-1, keepdim=True)
        lo, hi = lower.gather(-1, chosen), upper.gather(-1, chosen)
        start, end = lo.clone(), hi.clone()
        f_start = left.gather(-1, chosen)
        endpoint = (f_start == 0) | (right.gather(-1, chosen) == 0)
        exact = torch.where(f_start == 0, lo, hi)
        moving_root = firing & ~endpoint
        absolute = math.ulp(0.0)
        residual_tolerance = (
            64
            * eps
            * (1 + torch.minimum(f_start.abs(), right.gather(-1, chosen).abs()))
        )
        # Covers the full FP64 exponent range, including roots near zero.
        iterations = (
            math.ceil(math.log2(torch.finfo(torch.float64).max) - math.log2(absolute))
            + 2
        )
        for _ in range(iterations):
            middle = lo + (hi - lo) / 2
            tolerance = absolute + 8 * eps * torch.maximum(lo.abs(), hi.abs())
            moving = (
                moving_root & ((hi - lo) > tolerance) & (middle > lo) & (middle < hi)
            )
            if not bool(moving.any()):
                break
            value, _ = _evaluate(
                response, torch.where(moving, middle, lo), chosen, state
            )
            exact_middle = moving & (value == 0)
            lo = torch.where(exact_middle | (moving & (value < 0)), middle, lo)
            hi = torch.where(exact_middle | (moving & (value >= 0)), middle, hi)
        tolerance = absolute + 8 * eps * torch.maximum(lo.abs(), hi.abs())
        if bool((moving_root & (hi - lo > tolerance)).any()):
            raise RuntimeError("The first-spike root did not converge.")
        root = torch.where(endpoint, exact, hi)
        root = torch.where(firing, root, start)
        value, slope = _evaluate(response, root, chosen, state)
        residual_tolerance = residual_tolerance + slope.abs() * tolerance
        if bool(
            (
                firing
                & ((value.abs() > residual_tolerance) | (root < start) | (root > end))
            ).any()
        ):
            raise RuntimeError(
                "The first-spike root has an invalid residual or interval."
            )
        ctx.response = response
        ctx.save_for_backward(root, chosen, firing, *parameters)
        result = torch.where(firing, root, torch.inf).squeeze(-1).to(boundaries.dtype)
        if bool((firing.squeeze(-1) & ~torch.isfinite(result)).any()):
            raise RuntimeError("The first-spike time overflows its output dtype.")
        return result

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("first_spike_time supports first-order backward only.")
        root, chosen, firing, *parameters = ctx.saved_tensors
        if gradient is None:
            return (None,) * (3 + len(parameters))
        used = firing & (gradient[..., None] != 0)
        needs = ctx.needs_input_grad[3:]
        if not bool(used.any()) or not any(needs):
            return (None, None, None) + tuple(
                torch.zeros_like(p) if need else None
                for p, need in zip(parameters, needs)
            )
        with torch.enable_grad():
            state = tuple(
                (
                    p.detach().double() if p.is_floating_point() else p.detach()
                ).requires_grad_(need)
                for p, need in zip(parameters, needs)
            )
            value, slope = _evaluate(ctx.response, root, chosen, state)
            if bool((used & (slope <= 0)).any()):
                raise RuntimeError(
                    "A requested spike-time derivative is not transverse."
                )
            denominator = torch.where(used, slope.detach(), 1)
            factor = torch.where(used, -gradient.double()[..., None] / denominator, 0)
            if not bool(torch.isfinite(factor).all()):
                raise RuntimeError("The spike-time derivative is not finite.")
            requested = tuple(p for p, need in zip(state, needs) if need)
            derivatives = (
                torch.autograd.grad(value, requested, factor, allow_unused=True)
                if value.requires_grad
                else (None,) * len(requested)
            )
        result, position = [], 0
        for p, need in zip(parameters, needs):
            derivative = derivatives[position] if need else None
            position += int(need)
            if need:
                derivative = (
                    torch.zeros_like(p)
                    if derivative is None
                    else derivative.to(p.dtype)
                )
                if not bool(torch.isfinite(derivative).all()):
                    raise RuntimeError(
                        "Spike-time input gradients overflow their dtype."
                    )
            result.append(derivative)
        return (None, None, None, *result)


def first_spike_time(
    response: _Response,
    boundaries: torch.Tensor,
    parameters: Tuple[torch.Tensor, ...],
    extrema: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <timing-first-spike-cn>` | :ref:`English <timing-first-spike-en>`

    ----

    .. _timing-first-spike-cn:

    * **中文**

    在完整单调分段上寻找总阈值残差的最早上穿，不进行时间采样。调用方必须
    提供连续总响应（包括已接受输入及输出历史）的全部折点、驻点和窗口端点。
    本函数校验格式，不能从黑盒函数证明分段完整性。区间内必须单调；遗漏分段
    会使结果无效。仅有单个 PSP 的驻点并集不满足加权总响应契约。

    ``response(time,piece,*parameters)`` 返回残差 ``F`` 和偏导 ``F_t``，
    两者须为同设备、与 time 同形状的有限 FP64 张量。time/piece 的形状为
    ``[...,Q]`` （Q 为 K 或 1），piece 是对应原分段的 int64 索引。
    必须支持单侧端点求值；piece 决定折点采用哪一侧。每个前导索引是一条
    独立时间轴，禁止在响应中耦合不同时间轴的查询时刻。函数必须无状态、
    可重复执行，所有可微依赖须显式放入 parameters，不能隐藏在闭包内。

    仅支持一阶反向；create_graph=True 明确报错，包括参数来自可微变换的情况。
    FP64 二分只用于前向；一阶反向重算 ``-F_p/F_t``，传播到显式参数，
    包括可微的历史发放时刻。分段位置/类型仅负责选分支，不求导。
    未发放返回 inf，其梯度为零。相切不发放；窗口终点上穿计入。
    水平拐点上穿仍发放；只有请求时刻梯度时才拒绝其奇异导数。
    已认证极值的残差在
    ``64*eps*(1+abs(F_left)+abs(F_right))`` 内视为相切，
    小于该数值分辨率的超阈峰不保证可辨。折点选择较早区间的左侧分支。
    不保证拓扑切换处唯一导数；不含 reset、shunting、容量管理或替代梯度。

    :param response: 上述可微总残差/时间偏导函数，必须满足完整分段契约。
    :type response: Callable
    :param boundaries: 同设备 CPU/CUDA FP32/FP64，形状 ``[...,K+1]``，K>=1；
        有限、非递减的全部分段端点。同值可填充空区间。单位由调用方定义。
    :type boundaries: torch.Tensor
    :param parameters: 响应的显式张量依赖元组，与 boundaries 同设备且有限；
        浮点仅支持 FP32/FP64，内部转 FP64，返回梯度恢复各自 dtype。
    :type parameters: Tuple[torch.Tensor, ...]
    :param extrema: 同设备 bool、形状 ``[...,K,2]`` 的区间左/右侧局部极值/平台端点标记；默认
        None 表示调用方认证所有端点都非极值。相切点及平台端点必须标记，
        水平拐点不标记；窗口端点也须由核实现认证类型，折点两侧独立。
    :type extrema: Optional[torch.Tensor]
    :return: ``[...]`` 最早上穿时刻，与 boundaries 同 dtype/device，未发放 inf。
    :rtype: torch.Tensor
    :raises TypeError: 参数类型或 dtype 不受支持。
    :raises ValueError: 分段、设备、有限性或响应输出格式非法。
    :raises RuntimeError: 非有限响应、求根失败、被请求的非横截导数或梯度溢出。

    ----

    .. _timing-first-spike-en:

    * **English**

    Find the earliest upcrossing of a total threshold residual on a complete
    monotone partition without time sampling. The caller supplies every knot,
    stationary point and window endpoint of the continuous total response,
    including accepted inputs and output history. Format checks cannot certify
    completeness from a black box. Each interval must be monotone; an incomplete
    partition invalidates the result. The union of individual PSP stationary
    points does not satisfy the weighted-total-response contract.

    ``response(time,piece,*parameters)`` returns finite same-device FP64 residuals
    ``F`` and partial time derivatives ``F_t`` matching time's shape ``[...,Q]``
    (Q is K or 1). piece contains int64 original interval indices and selects the
    one-sided evaluation at knots. Each leading index has an independent time
    axis; queries must not couple these axes. The callable must be stateless and
    repeatable. All differentiable dependencies must be explicit parameters,
    including output history, rather than hidden in a closure.

    Only first-order backward is supported; create_graph=True raises, including
    when parameters come from differentiable transformations.
    FP64 bisection is forward-only. First-order backward recomputes ``-F_p/F_t``
    for explicit parameters, including differentiable previous spike times.
    Partition positions/types select branches and receive no gradient. Silence
    returns inf with zero gradient. Grazing does not fire; endpoint upcrossings
    count. Horizontal inflection crossings still fire; singular derivatives raise
    only when requested. A certified-extremum residual within
    ``64*eps*(1+abs(F_left)+abs(F_right))`` is treated as grazing; smaller peak
    overshoots cannot be guaranteed distinguishable. Knots select the earlier
    interval's left-sided branch. Topology changes need not have unique
    derivatives. No reset, shunting, capacity policy or surrogate is built in.

    :param response: Differentiable total residual/time-partial callable above,
        satisfying the complete-partition contract.
    :type response: Callable
    :param boundaries: Finite nondecreasing CPU/CUDA FP32/FP64 endpoints
        ``[...,K+1]``, K>=1, in caller-defined time units. Repeats pad empty intervals.
    :type boundaries: torch.Tensor
    :param parameters: Explicit finite tensor dependencies on the boundaries'
        device. Floating tensors must be FP32/FP64, internally promoted to FP64;
        gradients return in each parameter's dtype.
    :type parameters: Tuple[torch.Tensor, ...]
    :param extrema: Same-device bool local-extremum/plateau-end flags ``[...,K,2]``.
        None certifies every endpoint is non-extremal. Flag grazing and plateau
        endpoints, not horizontal inflections; the kernel must certify window
        endpoint types too. Sides of knots are independent.
    :type extrema: Optional[torch.Tensor]
    :return: Earliest upcrossing ``[...]`` matching boundaries' dtype/device;
        silence returns inf.
    :rtype: torch.Tensor
    :raises TypeError: For unsupported argument types or dtypes.
    :raises ValueError: For invalid partitions, devices, finite inputs or response format.
    :raises RuntimeError: For non-finite responses, failed roots, requested
        non-transverse derivatives or overflowing gradients.
    """
    if not callable(response) or not isinstance(parameters, tuple):
        raise TypeError("response must be callable and parameters must be a tuple.")
    if not isinstance(boundaries, torch.Tensor):
        raise TypeError("boundaries must be a tensor.")
    if boundaries.dtype not in (torch.float32, torch.float64):
        raise TypeError("boundaries must be FP32 or FP64.")
    if boundaries.device.type not in ("cpu", "cuda"):
        raise ValueError("The solver supports CPU and CUDA.")
    if boundaries.ndim < 1 or boundaries.shape[-1] < 2:
        raise ValueError("boundaries must have shape [..., K+1] with K >= 1.")
    if not bool(torch.isfinite(boundaries).all() & (boundaries.diff() >= 0).all()):
        raise ValueError("boundaries must be finite and nondecreasing.")
    for p in parameters:
        if not isinstance(p, torch.Tensor):
            raise TypeError("parameters must contain tensors.")
        if p.is_complex() or (
            p.is_floating_point() and p.dtype not in (torch.float32, torch.float64)
        ):
            raise TypeError("Floating parameters must be FP32 or FP64.")
        if p.device != boundaries.device or not bool(torch.isfinite(p).all()):
            raise ValueError("parameters must be finite on the boundaries' device.")
    flag_shape = (*boundaries.shape[:-1], boundaries.shape[-1] - 1, 2)
    if extrema is None:
        extrema = torch.zeros(flag_shape, dtype=torch.bool, device=boundaries.device)
    if not isinstance(extrema, torch.Tensor):
        raise TypeError("extrema must be a tensor or None.")
    if (
        extrema.dtype != torch.bool
        or extrema.shape != flag_shape
        or extrema.device != boundaries.device
    ):
        raise ValueError("extrema must be matching same-device bool flags.")
    return _FirstSpike.apply(response, boundaries.detach(), extrema, *parameters)
