import math

import torch
from torch.autograd.function import once_differentiable

__all__ = ["tempotron"]


def _parameters(duration, tau, tau_s, threshold):
    if any(
        not isinstance(x, (int, float)) or isinstance(x, bool)
        for x in (duration, tau, tau_s, threshold)
    ):
        raise TypeError("Time parameters and threshold must be fixed real scalars.")
    if not all(math.isfinite(x) and x > 0 for x in (duration, tau, tau_s, threshold)):
        raise ValueError(
            "duration, tau, tau_s, and v_threshold must be finite and positive."
        )
    if tau == tau_s:
        raise ValueError("tau and tau_s must be different.")
    slow, fast = max(tau, tau_s), min(tau, tau_s)
    delta = (slow - fast) / slow / fast
    if not math.isfinite(delta) or delta <= 0 or not math.isfinite(delta * slow):
        raise ValueError("Time constants are not numerically representable in FP64.")
    return slow, fast, delta


def _phi(delay, delta):
    z = delta * delay
    series = delay * (1 - z / 2 + z.square() / 6)
    return torch.where(
        z.abs() < math.sqrt(torch.finfo(torch.float64).eps),
        series,
        -torch.expm1(-z) / delta,
    )


def _voltage(voltage, current, delay, slow, delta, normalizer):
    decay = torch.exp(-delay / slow)
    return voltage * decay / normalizer + current * (
        decay * _phi(delay, delta) / normalizer
    )


def _stationary(voltage, current, slow, delta):
    nonzero = current != 0
    ratio = voltage / torch.where(nonzero, current, 1)
    denominator = 1 + delta * ratio
    valid = (
        nonzero
        & torch.isfinite(ratio)
        & torch.isfinite(denominator)
        & (denominator > 0)
    )
    z = (
        delta
        * (slow - torch.where(valid, ratio, 0))
        / torch.where(valid, denominator, 1)
    )
    valid = valid & torch.isfinite(z) & (z > -1)
    delay = torch.log1p(torch.where(valid, z, 0)) / delta
    return torch.where(valid, delay, torch.inf)


def _roots(
    voltage,
    current,
    lower,
    upper,
    mask,
    slow,
    fast,
    delta,
    normalizer,
    threshold,
    duration,
):
    eps = torch.finfo(torch.float64).eps
    width = upper - lower
    absolute = 8 * eps * torch.minimum(width, width.new_tensor(fast))
    absolute = absolute.clamp_min(math.ulp(0.0))
    lower = torch.where(mask, lower, 0)
    upper = torch.where(mask, upper, 0)
    iterations = (
        math.ceil(max(0.0, math.log2(duration) - math.log2(fast)) - math.log2(8 * eps))
        + 2
    )
    for _ in range(iterations):
        middle = lower + (upper - lower) / 2
        tolerance = absolute + 8 * eps * torch.maximum(lower.abs(), upper.abs())
        moving = (
            mask & ((upper - lower) > tolerance) & (middle > lower) & (middle < upper)
        )
        value = _voltage(voltage, current, middle, slow, delta, normalizer)
        below = value < threshold
        lower = torch.where(moving & below, middle, lower)
        upper = torch.where(moving & ~below, middle, upper)
    tolerance = absolute + 8 * eps * torch.maximum(lower.abs(), upper.abs())
    if bool(
        (
            mask
            & ((upper - lower) > tolerance)
            & (torch.nextafter(lower, upper) < upper)
        ).any()
    ):
        raise RuntimeError("Tempotron threshold root did not converge.")
    root = upper
    value = _voltage(voltage, current, root, slow, delta, normalizer)
    raw = value * normalizer
    q = current * torch.exp(-root / fast)
    slope = (q - raw / slow) / normalizer
    scale = (
        1
        + threshold
        + (voltage.abs() + (current * _phi(root, delta)).abs()) / normalizer
    )
    residual_tolerance = 64 * eps * scale + slope.abs() * tolerance
    if bool(
        (
            mask
            & (
                ~torch.isfinite(value)
                | ((value - threshold).abs() > residual_tolerance)
            )
        ).any()
    ):
        raise RuntimeError("Tempotron threshold root has an invalid voltage residual.")
    return root


def _psp(delay, slow, fast, delta, normalizer):
    decay = torch.exp(-delay / slow)
    fast_decay = torch.exp(-delta * delay)
    phi = _phi(delay, delta)
    value = decay * phi / normalizer
    first = decay * (fast_decay - phi / slow) / normalizer
    second = decay * (phi / slow / slow - (2 / slow + delta) * fast_decay) / normalizer
    return value, first, second


class _Tempotron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, times, indices, offsets, weight, duration, tau, tau_s, threshold):
        slow, fast, delta = _parameters(duration, tau, tau_s, threshold)
        ctx.set_materialize_grads(False)
        times64, weight64 = times.double(), weight.double()
        peak_delay = times64.new_tensor(delta * slow).log1p() / delta
        normalizer = torch.exp(-peak_delay / slow) * _phi(peak_delay, delta)
        if not bool(torch.isfinite(normalizer) & (normalizer > 0)):
            raise RuntimeError(
                "Tempotron PSP normalization is not finite and positive."
            )
        batch_count, output_count = offsets.numel() - 1, weight.shape[0]
        maxima = times64.new_zeros((batch_count, output_count))
        peak_times = maxima.clone()
        spike_times = torch.full_like(maxima, torch.inf)
        peak_types = torch.zeros_like(maxima, dtype=torch.int64)
        peak_events = torch.full_like(peak_types, -1)
        accepted = torch.zeros_like(peak_types)
        event_orders, valid_offsets = [], [0]
        boundaries = offsets.tolist()
        for batch, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            positions = torch.arange(begin, end, device=times.device)
            valid = (times64[positions] >= 0) & (times64[positions] <= duration)
            positions = positions[valid]
            positions = positions[torch.argsort(times64[positions], stable=True)]
            event_orders.append(positions)
            valid_offsets.append(valid_offsets[-1] + positions.numel())
            unique, counts = torch.unique_consecutive(
                times64[positions], return_counts=True
            )
            if unique.numel() == 0:
                continue
            starts = counts.cumsum(0) - counts
            first_events = positions[starts]
            grouped_weight = torch.segment_reduce(
                weight64[:, indices[positions]].T, "sum", lengths=counts
            ).T
            # Zero-current arrivals cannot change the trajectory; retain their
            # positions below so their weight derivatives and shunting stay correct.
            nonzero = (grouped_weight != 0).any(0)
            unique = unique[nonzero]
            first_events = first_events[nonzero]
            grouped_weight = grouped_weight[:, nonzero]
            if unique.numel() == 0:
                accepted[batch] = positions.numel()
                continue
            finite_weight = torch.isfinite(grouped_weight)
            invalid_prefix = (~finite_weight).to(torch.int64).cumsum(1) > 0
            grouped_weight = torch.where(finite_weight, grouped_weight, 0)
            endpoints = torch.cat((unique[1:], times64.new_tensor([duration])))
            right_events = torch.cat((first_events[1:], indices.new_tensor([-1])))
            right_events = torch.where(endpoints < duration, right_events, -1)
            voltage = times64.new_zeros(output_count)
            current = voltage.clone()
            origin = times64.new_zeros(())
            outputs_index = torch.arange(output_count, device=times.device)
            # ponytail: bounded event matrices; kernels await the kernel refactor merge.
            for start in range(0, unique.numel(), 128):
                end = min(start + 128, unique.numel())
                block_times = unique[start:end]
                active = torch.isinf(spike_times[batch])
                incoming = torch.where(active[:, None], grouped_weight[:, start:end], 0)
                delays = (block_times[:, None] - block_times[None, :]).clamp_min(0)
                kernel = torch.exp(-delays / slow) * _phi(delays, delta) / normalizer
                block_voltage = (
                    _voltage(
                        voltage[:, None],
                        current[:, None],
                        block_times - origin,
                        slow,
                        delta,
                        normalizer,
                    )
                    + incoming @ kernel.T
                )
                decay = torch.exp(-delays / fast).tril()
                block_current = (
                    current[:, None] * torch.exp(-(block_times - origin) / fast)
                    + incoming @ decay.T
                )
                finite_state = torch.isfinite(block_voltage) & torch.isfinite(
                    block_current
                )
                block_voltage = torch.where(finite_state, block_voltage, 0)
                block_current = torch.where(finite_state, block_current, 0)
                raw_voltage = block_voltage * normalizer
                interval = endpoints[start:end] - block_times
                critical = _stationary(raw_voltage, block_current, slow, delta)
                interior = (
                    torch.isfinite(critical) & (critical > 0) & (critical < interval)
                )
                endpoint_stationary = (
                    torch.isfinite(critical)
                    & (critical > 0)
                    & (
                        (critical - interval).abs()
                        <= 8
                        * torch.finfo(torch.float64).eps
                        * (
                            interval.abs()
                            + torch.minimum(interval, interval.new_tensor(fast))
                        )
                    )
                )
                local = torch.stack(
                    (
                        torch.zeros_like(critical),
                        torch.where(interior, critical, interval),
                        interval.expand_as(critical),
                    ),
                    -1,
                )
                values = _voltage(
                    raw_voltage[..., None],
                    block_current[..., None],
                    local,
                    slow,
                    delta,
                    normalizer,
                )
                finite_values = torch.isfinite(values).all(-1)
                values = torch.where(finite_values[..., None], values, 0)
                left, right = values[..., :-1], values[..., 1:]
                lower, upper = local[..., :-1], local[..., 1:]
                lower_slope = (
                    block_current[..., None] * torch.exp(-lower / fast)
                    - left * normalizer / slow
                ) / normalizer
                upper_slope = (
                    block_current[..., None] * torch.exp(-upper / fast)
                    - right * normalizer / slow
                ) / normalizer
                stationary_end = torch.stack(
                    (interior | endpoint_stationary, endpoint_stationary), -1
                )
                at_threshold = right == threshold
                reaches = (right > threshold) | (
                    at_threshold & ~stationary_end & (upper_slope > 0)
                )
                rising_start = torch.zeros_like(reaches)
                rising_start[..., 0] = (left[..., 0] == threshold) & (
                    lower_slope[..., 0] > 0
                )
                bracket = (
                    active[:, None, None]
                    & (upper > lower)
                    & (((left < threshold) & reaches) | rising_start)
                )
                firing = bracket.flatten(1).any(1)
                first = bracket.flatten(1).to(torch.int8).argmax(1)
                first_group, first_segment = first // 2, first % 2
                accepted_groups = active[:, None] & (
                    ~firing[:, None]
                    | (
                        torch.arange(end - start, device=times.device)
                        <= first_group[:, None]
                    )
                )
                if bool(
                    (
                        accepted_groups
                        & (
                            invalid_prefix[:, start:end]
                            | ~finite_state
                            | ~finite_values
                        )
                    ).any()
                ):
                    raise RuntimeError("Tempotron voltage or current is not finite.")
                if bool(firing.any()):
                    chosen_voltage = raw_voltage[outputs_index, first_group]
                    chosen_current = block_current[outputs_index, first_group]
                    chosen_lower = lower[outputs_index, first_group, first_segment]
                    chosen_upper = upper[outputs_index, first_group, first_segment]
                    chosen_origin = block_times[first_group]
                    starts_rising = rising_start[
                        outputs_index, first_group, first_segment
                    ]
                    endpoint_hit = (
                        at_threshold[outputs_index, first_group, first_segment]
                        & ~stationary_end[outputs_index, first_group, first_segment]
                        & ~starts_rising
                    )
                    root_mask = firing & ~endpoint_hit & ~starts_rising
                    root = (
                        _roots(
                            chosen_voltage,
                            chosen_current,
                            chosen_lower,
                            chosen_upper,
                            root_mask,
                            slow,
                            fast,
                            delta,
                            normalizer,
                            threshold,
                            duration,
                        )
                        if bool(root_mask.any())
                        else torch.zeros_like(chosen_lower)
                    )
                    root = torch.where(endpoint_hit, chosen_upper, root)
                    resolved_time = torch.where(
                        starts_rising,
                        torch.nextafter(
                            chosen_origin + chosen_lower, chosen_origin + chosen_upper
                        ),
                        chosen_origin + root,
                    )
                    if bool(
                        (
                            firing
                            & starts_rising
                            & (resolved_time <= chosen_origin + chosen_lower)
                        ).any()
                    ):
                        raise RuntimeError(
                            "Tempotron upcrossing time is not representable."
                        )
                    spike_times[batch] = torch.where(
                        firing, resolved_time, spike_times[batch]
                    )
                    tail_critical = critical[outputs_index, first_group]
                    tail_end = duration - chosen_origin
                    tail_interior = (
                        torch.isfinite(tail_critical)
                        & (tail_critical > resolved_time - chosen_origin)
                        & (tail_critical < tail_end)
                    )
                    tail_time = torch.where(tail_interior, tail_critical, tail_end)
                    tail_max = _voltage(
                        chosen_voltage,
                        chosen_current,
                        tail_time,
                        slow,
                        delta,
                        normalizer,
                    )
                    maxima[batch] = torch.where(firing, tail_max, maxima[batch])
                    peak_times[batch] = torch.where(
                        firing, chosen_origin + tail_time, peak_times[batch]
                    )
                    peak_types[batch] = torch.where(
                        firing, tail_interior.to(torch.int64), peak_types[batch]
                    )
                    peak_events[batch] = torch.where(firing, -1, peak_events[batch])
                values[..., 1] = torch.where(interior, values[..., 1], -torch.inf)
                candidate_max, candidate_index = values.flatten(1).max(1)
                candidate_group, candidate_phase = (
                    candidate_index // 3,
                    candidate_index % 3,
                )
                better = active & ~firing & (candidate_max > maxima[batch])
                selected_local = (
                    local.flatten(1).gather(1, candidate_index[:, None]).squeeze(1)
                )
                chosen_event = torch.where(
                    candidate_phase == 0,
                    first_events[start:end][candidate_group],
                    right_events[start:end][candidate_group],
                )
                chosen_type = torch.where(
                    candidate_phase == 1, 1, torch.where(chosen_event >= 0, 2, 0)
                )
                maxima[batch] = torch.where(better, candidate_max, maxima[batch])
                peak_times[batch] = torch.where(
                    better,
                    block_times[candidate_group] + selected_local,
                    peak_times[batch],
                )
                peak_types[batch] = torch.where(better, chosen_type, peak_types[batch])
                peak_events[batch] = torch.where(
                    better & (chosen_type == 2), chosen_event, peak_events[batch]
                )
                remaining = torch.isinf(spike_times[batch])
                voltage = torch.where(remaining, raw_voltage[:, -1], 0)
                current = torch.where(remaining, block_current[:, -1], 0)
                origin = block_times[-1]
                if not bool(remaining.any()):
                    break
            accepted[batch] = torch.searchsorted(
                times64[positions], spike_times[batch], right=False
            )
        order = torch.cat(event_orders) if event_orders else indices.new_empty(0)
        valid_offsets = offsets.new_tensor(valid_offsets)
        ctx.save_for_backward(
            times,
            indices,
            weight,
            order,
            valid_offsets,
            peak_times,
            spike_times,
            accepted,
            peak_types,
            peak_events,
            normalizer,
        )
        ctx.parameters = slow, fast, delta
        outputs = (
            maxima.to(times.dtype),
            peak_times.to(times.dtype),
            spike_times.to(times.dtype),
        )
        if not bool(
            torch.isfinite(outputs[0]).all() & torch.isfinite(outputs[1]).all()
        ):
            raise RuntimeError("Tempotron outputs overflow their dtype.")
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_max, grad_peak, grad_spike):
        (
            times,
            indices,
            weight,
            order,
            offsets,
            peak,
            spike,
            accepted,
            kinds,
            peak_events,
            normalizer,
        ) = ctx.saved_tensors
        slow, fast, delta = ctx.parameters
        shape = peak.shape
        gradients = []
        for gradient in (grad_max, grad_peak, grad_spike):
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Tempotron received a non-finite upstream gradient.")
            gradients.append(
                peak.new_zeros(shape) if gradient is None else gradient.double()
            )
        gmax, gpeak, gspike = gradients
        time_gradient, weight_gradient = (
            torch.zeros_like(times, dtype=torch.float64),
            torch.zeros_like(weight, dtype=torch.float64),
        )
        bounds = offsets.tolist()
        for batch, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            positions = order[begin:end]
            count = positions.numel()
            peak_slope = peak.new_zeros(weight.shape[0])
            peak_curvature = peak_slope.clone()
            spike_slope = peak_slope.clone()
            finite_spike = torch.isfinite(spike[batch])
            safe_spike = torch.where(finite_spike, spike[batch], 0)
            use_peak = (kinds[batch] == 1) & (gpeak[batch] != 0)
            use_spike = finite_spike & (gspike[batch] != 0)
            compute_peak = grad_peak is not None and bool(use_peak.any())
            compute_spike = grad_spike is not None and bool(use_spike.any())
            compute_boundary = grad_max is not None and bool(
                ((kinds[batch] == 2) & (gmax[batch] != 0)).any()
            )
            derivative_count = (
                count if compute_peak or compute_spike or compute_boundary else 0
            )
            for start in range(0, derivative_count, 1024):
                pos = positions[start : start + 1024]
                event_times = times[pos].double()[None, :]
                event_weight = weight[:, indices[pos]].double()
                prefix = (
                    torch.arange(start, start + pos.numel(), device=times.device)[
                        None, :
                    ]
                    < accepted[batch, :, None]
                )
                if compute_peak or compute_boundary:
                    peak_mask = prefix & (event_times < peak[batch, :, None])
                    _, first, second = _psp(
                        (peak[batch, :, None] - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    if compute_boundary:
                        peak_slope += (
                            torch.where(peak_mask, first, 0) * event_weight
                        ).sum(1)
                    if compute_peak:
                        peak_curvature += (
                            torch.where(peak_mask, second, 0) * event_weight
                        ).sum(1)
                if compute_spike:
                    spike_mask = (
                        prefix
                        & finite_spike[:, None]
                        & (event_times < safe_spike[:, None])
                    )
                    _, spike_first, _ = _psp(
                        (safe_spike[:, None] - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    spike_slope += (
                        torch.where(spike_mask, spike_first, 0) * event_weight
                    ).sum(1)
            if compute_peak and bool(
                (
                    use_peak & (~torch.isfinite(peak_curvature) | (peak_curvature >= 0))
                ).any()
            ):
                raise RuntimeError("Tempotron peak-time derivative is singular.")
            if compute_spike and bool(
                (use_spike & (~torch.isfinite(spike_slope) | (spike_slope <= 0))).any()
            ):
                raise RuntimeError("Tempotron spike-time derivative is singular.")
            peak_factor = torch.where(use_peak, gpeak[batch], 0) / torch.where(
                use_peak, peak_curvature, 1
            )
            spike_factor = torch.where(use_spike, gspike[batch], 0) / torch.where(
                use_spike, spike_slope, 1
            )
            contribution_count = (
                count if grad_max is not None or compute_peak or compute_spike else 0
            )
            for start in range(0, contribution_count, 1024):
                pos = positions[start : start + 1024]
                event_times = times[pos].double()[None, :]
                columns = indices[pos]
                event_weight = weight[:, columns].double()
                prefix = (
                    torch.arange(start, start + pos.numel(), device=times.device)[
                        None, :
                    ]
                    < accepted[batch, :, None]
                )
                dw, dt = torch.zeros_like(event_weight), torch.zeros_like(event_weight)
                if grad_max is not None or compute_peak:
                    peak_mask = prefix & (event_times < peak[batch, :, None])
                    value, first, second = _psp(
                        (peak[batch, :, None] - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    dw = torch.where(
                        peak_mask,
                        gmax[batch, :, None] * value
                        - (peak_factor[:, None] * first if compute_peak else 0),
                        0,
                    )
                    dt = torch.where(
                        peak_mask,
                        event_weight
                        * (
                            -gmax[batch, :, None] * first
                            + (peak_factor[:, None] * second if compute_peak else 0)
                        ),
                        0,
                    )
                if compute_spike:
                    spike_mask = (
                        prefix
                        & finite_spike[:, None]
                        & (event_times < safe_spike[:, None])
                    )
                    spike_value, spike_first, _ = _psp(
                        (safe_spike[:, None] - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    dw -= torch.where(
                        spike_mask, spike_factor[:, None] * spike_value, 0
                    )
                    dt += torch.where(
                        spike_mask,
                        spike_factor[:, None] * event_weight * spike_first,
                        0,
                    )
                if ctx.needs_input_grad[3]:
                    weight_gradient.index_add_(1, columns, dw)
                if ctx.needs_input_grad[0]:
                    time_gradient.index_add_(0, pos, dt.sum(0))
            boundary = kinds[batch] == 2
            if ctx.needs_input_grad[0] and bool(boundary.any()):
                time_gradient.index_add_(
                    0,
                    peak_events[batch, boundary],
                    gpeak[batch, boundary]
                    + gmax[batch, boundary] * peak_slope[boundary],
                )
        if ctx.needs_input_grad[0]:
            time_gradient = time_gradient.to(times.dtype)
            if not bool(torch.isfinite(time_gradient).all()):
                raise RuntimeError(
                    "Tempotron event-time gradients overflow their dtype."
                )
        if ctx.needs_input_grad[3]:
            weight_gradient = weight_gradient.to(weight.dtype)
            if not bool(torch.isfinite(weight_gradient).all()):
                raise RuntimeError("Tempotron weight gradients overflow their dtype.")
        return (
            time_gradient if ctx.needs_input_grad[0] else None,
            None,
            None,
            weight_gradient if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
            None,
        )


def tempotron(
    spike_times: torch.Tensor,
    input_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    weight: torch.Tensor,
    duration: float,
    tau: float = 15.0,
    tau_s: float = 3.75,
    v_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <timing-functional-tempotron-cn>` | :ref:`English <timing-functional-tempotron-en>`

    ----

    .. _timing-functional-tempotron-cn:

    * **中文**

    用连续事件求解独立 Tempotron 试次，不展开均匀时间轴。电压由单位峰值的
    双指数 PSP 加权求和；只有上穿阈值才发放，相切不发放。发放后屏蔽同刻和
    更晚的输入，已有 PSP 继续演化，不执行硬重置。每条输入连接可有多个事件。
    同刻事件聚合，输出发放优先。

    支持 CPU/CUDA 上的 FP32/FP64，所有张量必须同设备，时刻和权重同 dtype。
    内部使用 FP64，以有界局部事件矩阵分块求解，避免逐事件启动小算子。
    CUDA 仍可能有块级控制流程同步，不使用专用 GPU 核。反向只重算被使用输出
    所需的导数；仅支持对权重及输入时刻的一阶反向。

    峰值并列时选择最早分支；事件边界使用稳定排序中的首个原事件作为时间分支，
    固定窗口端点的峰值时间梯度为零。非光滑点不保证唯一导数。未发放的 ``t_spike``
    分支梯度为零，不内置 loss、替代梯度、裁剪或沉默恢复。

    :param spike_times: 有限事件时刻，形状 ``[E]``；与时间常数和 duration
        同单位。窗外事件忽略且梯度为零。
    :type spike_times: torch.Tensor
    :param input_indices: 输入连接索引，int64 ``[E]``，范围 ``[0, I)``。
    :type input_indices: torch.Tensor
    :param batch_offsets: int64 ``[B+1]``，从 0 开始、非递减、以 E 结束。
        每段可无序或为空。
    :type batch_offsets: torch.Tensor
    :param weight: 有限突触权重，形状 ``[O,I]``，I、O 均为正；重复连接共享权重。
    :type weight: torch.Tensor
    :param duration: 正且有限的观察窗终点，窗口为 ``[0,duration]``。
    :type duration: float
    :param tau: 正且有限的膜时间常数，默认 15；必须与 tau_s 不同。
    :type tau: float
    :param tau_s: 正且有限的突触时间常数，默认 3.75；允许大于 tau。
    :type tau_s: float
    :param v_threshold: 正且有限的阈值，默认 1；不影响 PSP 单位峰值归一化。
    :type v_threshold: float
    :return: ``(v_max,t_peak,t_spike)``，均为 ``[B,O]``，dtype/device 与时刻
        相同。峰值属于实际 shunting 轨迹，未发放时 t_spike 为 inf；空样本返回
        ``(0,0,inf)``。调用间无状态持久化。
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises TypeError: dtype 或参数类型不受支持。
    :raises ValueError: 形状、设备、事件索引、分段边界、有限性或超参数非法。
    :raises RuntimeError: 数值状态、求根、被使用的时间导数或梯度不可表示。

    ----

    .. _timing-functional-tempotron-en:

    * **English**

    Solve independent continuous-event Tempotron trials without a uniform time grid.
    Voltage is a weighted sum of unit-peak double-exponential PSPs. Only upward
    threshold crossings fire; grazing does not. A spike shunts simultaneous and
    later inputs. Existing PSPs continue without a hard reset. Each connection may
    carry multiple events. Simultaneous events are grouped; output firing wins ties.

    Supports CPU/CUDA FP32/FP64 with all tensors on one device and matching time
    and weight dtypes. Computation uses FP64 and bounded local event matrices to
    solve blocks, avoiding tiny per-event launches. CUDA may still synchronize
    block-level control flow; no specialized GPU kernel is used. Backward recomputes
    only derivatives needed by used outputs. Only first-order weight and input-time
    derivatives are supported.

    Equal peaks select the earliest branch. Event-boundary time derivatives select
    the first original event in stable order; fixed-window peak times have zero
    derivative. Nonsmooth points need not have unique derivatives. Non-firing ``t_spike``
    branches have zero gradient. No loss, surrogate, clipping or silent-neuron
    recovery is built in.

    :param spike_times: Finite times of shape ``[E]``, in the same units as duration
        and the time constants. Out-of-window events are ignored with zero gradient.
    :type spike_times: torch.Tensor
    :param input_indices: int64 connection indices of shape ``[E]`` in ``[0,I)``.
    :type input_indices: torch.Tensor
    :param batch_offsets: Nondecreasing int64 boundaries ``[B+1]`` starting at 0
        and ending at E. Segments may be unordered or empty.
    :type batch_offsets: torch.Tensor
    :param weight: Finite weights ``[O,I]`` with positive O and I; repeated
        connections share their weight.
    :type weight: torch.Tensor
    :param duration: Finite positive endpoint of the window ``[0,duration]``.
    :type duration: float
    :param tau: Finite positive membrane time constant, default 15; differs from tau_s.
    :type tau: float
    :param tau_s: Finite positive synaptic time constant, default 3.75; may exceed tau.
    :type tau_s: float
    :param v_threshold: Finite positive threshold, default 1; independent of unit-peak
        PSP normalization.
    :type v_threshold: float
    :return: ``(v_max,t_peak,t_spike)``, each ``[B,O]`` with the input times' dtype
        and device. Peaks follow the actual shunted trajectory. Non-firing spike
        times are inf; empty trials return ``(0,0,inf)``. No state persists across calls.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises TypeError: For unsupported dtypes or parameter types.
    :raises ValueError: For invalid shapes, devices, indices, offsets, finite values
        or hyperparameters.
    :raises RuntimeError: For unrepresentable numerical states, roots, requested
        time derivatives or gradients.
    """
    _parameters(duration, tau, tau_s, v_threshold)
    tensors = (spike_times, input_indices, batch_offsets, weight)
    if any(
        t.device != spike_times.device for t in tensors
    ) or spike_times.device.type not in ("cpu", "cuda"):
        raise ValueError("All tensors must be on the same CPU or CUDA device.")
    if (
        spike_times.dtype not in (torch.float32, torch.float64)
        or weight.dtype != spike_times.dtype
        or input_indices.dtype != torch.int64
        or batch_offsets.dtype != torch.int64
    ):
        raise TypeError(
            "Times/weights must share FP32 or FP64; indices/offsets must be int64."
        )
    if (
        spike_times.ndim != 1
        or input_indices.shape != spike_times.shape
        or batch_offsets.ndim != 1
        or batch_offsets.numel() < 1
        or weight.ndim != 2
        or min(weight.shape) < 1
    ):
        raise ValueError("Expected times/indices [E], offsets [B+1], and weight [O,I].")
    if not bool(torch.isfinite(spike_times).all() & torch.isfinite(weight).all()):
        raise ValueError("Times and weights must be finite.")
    if (
        int(batch_offsets[0]) != 0
        or int(batch_offsets[-1]) != spike_times.numel()
        or bool((batch_offsets[1:] < batch_offsets[:-1]).any())
    ):
        raise ValueError("Offsets must start at 0, be nondecreasing, and end at E.")
    if bool(((input_indices < 0) | (input_indices >= weight.shape[1])).any()):
        raise ValueError("Input indices must be in [0,in_features).")
    return _Tempotron.apply(
        spike_times,
        input_indices,
        batch_offsets,
        weight,
        duration,
        tau,
        tau_s,
        v_threshold,
    )
