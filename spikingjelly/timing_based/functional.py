import math

import torch
from torch.autograd.function import once_differentiable

from .._ops.tempotron import _roots, _states, _weight_gradient
from .._ops.tempotron.reference import _phi, _psp, _stationary, _voltage

__all__ = ["tempotron", "srm"]

_EVENT_BLOCK_SIZE = {"cpu": 128, "cuda": 512}
_BACKWARD_BLOCK_SIZE = 16384


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
        event_batches = torch.repeat_interleave(
            torch.arange(batch_count, device=times.device),
            offsets.diff(),
            output_size=times.numel(),
        )
        order = torch.arange(times.numel(), device=times.device)
        order = order[(times64 >= 0) & (times64 <= duration)]
        order = order[torch.argsort(times64[order], stable=True)]
        order = order[torch.argsort(event_batches[order], stable=True)]
        sorted_batches = event_batches[order]
        valid_counts = torch.bincount(sorted_batches, minlength=batch_count)
        valid_offsets = torch.cat((offsets.new_zeros(1), valid_counts.cumsum(0)))
        if order.numel():
            new_group = torch.ones(order.numel(), dtype=torch.bool, device=times.device)
            new_group[1:] = (sorted_batches[1:] != sorted_batches[:-1]) | (
                times64[order[1:]] != times64[order[:-1]]
            )
            starts = new_group.nonzero().flatten()
            counts = (
                torch.cat((starts[1:], starts.new_tensor([order.numel()]))) - starts
            )
            first_events = order[starts]
            unique = times64[first_events]
            group_batches = event_batches[first_events]
            grouped_weight = torch.segment_reduce(
                weight64[:, indices[order]].T, "sum", lengths=counts
            ).T
            # Keep zero-current boundary metadata without accumulating decay roundoff.
            nonzero = (grouped_weight != 0).any(0)
            zero_events = first_events[~nonzero]
            unique, first_events, group_batches = (
                unique[nonzero],
                first_events[nonzero],
                group_batches[nonzero],
            )
            grouped_weight = grouped_weight[:, nonzero]
        else:
            zero_events = order
            unique = times64.new_empty(0)
            first_events = order
            group_batches = sorted_batches
            grouped_weight = weight64.new_empty((output_count, 0))
        group_counts = torch.bincount(group_batches, minlength=batch_count)
        group_offsets = torch.cat((offsets.new_zeros(1), group_counts.cumsum(0)))
        max_groups = int(group_counts.max()) if batch_count else 0
        if max_groups:
            next_same = torch.cat(
                (
                    group_batches[1:] == group_batches[:-1],
                    group_batches.new_zeros(1, dtype=torch.bool),
                )
            )
            endpoints = torch.where(
                next_same,
                torch.cat((unique[1:], unique.new_tensor([duration]))),
                duration,
            )
            right_events = torch.where(
                next_same & (endpoints < duration),
                torch.cat((first_events[1:], first_events.new_tensor([-1]))),
                -1,
            )
            invalid_weight = ~torch.isfinite(grouped_weight)
            grouped_weight = torch.where(invalid_weight, 0, grouped_weight)
        voltage = maxima.clone()
        current = maxima.clone()
        origin = times64.new_zeros(batch_count)
        batches_index = torch.arange(batch_count, device=times.device)[:, None]
        outputs_index = torch.arange(output_count, device=times.device)[None, :]
        block_size = _EVENT_BLOCK_SIZE[times.device.type]
        for start in range(0, max_groups, block_size):
            width = min(block_size, max_groups - start)
            local_groups = start + torch.arange(width, device=times.device)
            valid = local_groups[None, :] < group_counts[:, None]
            gather = (group_offsets[:-1, None] + local_groups).clamp_max(
                unique.numel() - 1
            )
            block_times = torch.where(valid, unique[gather], duration)
            block_endpoints = torch.where(valid, endpoints[gather], duration)
            block_first = first_events[gather]
            block_right = right_events[gather]
            block_invalid = (
                invalid_weight[:, gather].permute(1, 0, 2) & valid[:, None, :]
            ).to(torch.int64).cumsum(-1) > 0
            last = valid.sum(1).sub(1).clamp_min(0)
            active = torch.isinf(spike_times) & valid.any(-1)[:, None]
            incoming = torch.where(
                active[..., None] & valid[:, None, :],
                grouped_weight[:, gather].permute(1, 0, 2),
                0,
            )
            block_voltage, block_current = _states(
                block_times,
                incoming,
                voltage,
                current,
                origin,
                slow,
                fast,
                delta,
                normalizer,
            )
            finite_state = torch.isfinite(block_voltage) & torch.isfinite(block_current)
            block_voltage = torch.where(finite_state, block_voltage, 0)
            block_current = torch.where(finite_state, block_current, 0)
            raw_voltage = block_voltage * normalizer
            interval = (block_endpoints - block_times)[:, None, :]
            critical = _stationary(raw_voltage, block_current, slow, delta)
            interior = torch.isfinite(critical) & (critical > 0) & (critical < interval)
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
                active[..., None, None]
                & valid[:, None, :, None]
                & (upper > lower)
                & (((left < threshold) & reaches) | rising_start)
            )
            firing = bracket.flatten(2).any(-1)
            first = bracket.flatten(2).to(torch.int8).argmax(-1)
            first_group, first_segment = first // 2, first % 2
            accepted_groups = active[..., None] & (
                ~firing[..., None]
                | (torch.arange(width, device=times.device) <= first_group[..., None])
            )
            if bool(
                (
                    accepted_groups
                    & valid[:, None, :]
                    & (block_invalid | ~finite_state | ~finite_values)
                ).any()
            ):
                raise RuntimeError("Tempotron voltage or current is not finite.")
            if bool(firing.any()):
                chosen_voltage = raw_voltage[batches_index, outputs_index, first_group]
                chosen_current = block_current[
                    batches_index, outputs_index, first_group
                ]
                chosen_lower = lower[
                    batches_index, outputs_index, first_group, first_segment
                ]
                chosen_upper = upper[
                    batches_index, outputs_index, first_group, first_segment
                ]
                chosen_origin = block_times[batches_index, first_group]
                starts_rising = rising_start[
                    batches_index, outputs_index, first_group, first_segment
                ]
                endpoint_hit = (
                    at_threshold[
                        batches_index, outputs_index, first_group, first_segment
                    ]
                    & ~stationary_end[
                        batches_index, outputs_index, first_group, first_segment
                    ]
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
                spike_times = torch.where(firing, resolved_time, spike_times)
                tail_critical = critical[batches_index, outputs_index, first_group]
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
                maxima = torch.where(firing, tail_max, maxima)
                peak_times = torch.where(firing, chosen_origin + tail_time, peak_times)
                peak_types = torch.where(
                    firing, tail_interior.to(torch.int64), peak_types
                )
                peak_events = torch.where(firing, -1, peak_events)
            values[..., 1] = torch.where(interior, values[..., 1], -torch.inf)
            candidate_max, candidate_index = (
                torch.where(valid[:, None, :, None], values, -torch.inf)
                .flatten(2)
                .max(-1)
            )
            candidate_group, candidate_phase = (
                candidate_index // 3,
                candidate_index % 3,
            )
            better = active & ~firing & (candidate_max > maxima)
            selected_local = (
                local.flatten(2).gather(2, candidate_index[..., None]).squeeze(2)
            )
            chosen_event = torch.where(
                candidate_phase == 0,
                block_first[batches_index, candidate_group],
                block_right[batches_index, candidate_group],
            )
            chosen_type = torch.where(
                candidate_phase == 1, 1, torch.where(chosen_event >= 0, 2, 0)
            )
            maxima = torch.where(better, candidate_max, maxima)
            peak_times = torch.where(
                better,
                block_times[batches_index, candidate_group] + selected_local,
                peak_times,
            )
            peak_types = torch.where(better, chosen_type, peak_types)
            peak_events = torch.where(
                better & (chosen_type == 2), chosen_event, peak_events
            )
            remaining = (
                torch.isinf(spike_times) & (group_counts > start + width)[:, None]
            )
            voltage = torch.where(
                remaining, raw_voltage[batches_index, outputs_index, last[:, None]], 0
            )
            current = torch.where(
                remaining, block_current[batches_index, outputs_index, last[:, None]], 0
            )
            origin = block_times[torch.arange(batch_count, device=times.device), last]
            if not bool(remaining.any()):
                break
        if order.numel():
            accepted.index_add_(
                0,
                sorted_batches,
                (times64[order, None] < spike_times[sorted_batches]).to(torch.int64),
            )
        if zero_events.numel() and bool((peak_types == 1).any()):
            branches = torch.full_like(peak_events, times.numel())
            for start in range(0, zero_events.numel(), _BACKWARD_BLOCK_SIZE):
                pos = zero_events[start : start + _BACKWARD_BLOCK_SIZE]
                batch = event_batches[pos]
                at_peak = (
                    (peak_types[batch] == 1)
                    & (times64[pos, None] == peak_times[batch])
                    & (times64[pos, None] < spike_times[batch])
                )
                branches.scatter_reduce_(
                    0,
                    batch[:, None].expand(-1, output_count),
                    torch.where(at_peak, pos[:, None], times.numel()),
                    reduce="amin",
                )
            boundary = branches < times.numel()
            peak_types = torch.where(boundary, 2, peak_types)
            peak_events = torch.where(boundary, branches, peak_events)
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
        if not ctx.needs_input_grad[0] and grad_peak is None and grad_spike is None:
            weight_gradient = (
                _weight_gradient(
                    times,
                    indices,
                    weight,
                    order,
                    offsets,
                    peak,
                    accepted,
                    gmax,
                    slow,
                    delta,
                    normalizer,
                )
                if ctx.needs_input_grad[3] and grad_max is not None
                else weight_gradient
            )
        else:
            event_batches = torch.repeat_interleave(
                torch.arange(shape[0], device=times.device),
                offsets.diff(),
                output_size=order.numel(),
            )
            event_ranks = (
                torch.arange(order.numel(), device=times.device)
                - offsets[event_batches]
            )
            peak_slope = peak.new_zeros(shape)
            peak_curvature = peak_slope.clone()
            spike_slope = peak_slope.clone()
            finite_spike = torch.isfinite(spike)
            safe_spike = torch.where(finite_spike, spike, 0)
            use_peak = (kinds == 1) & (gpeak != 0)
            use_spike = finite_spike & (gspike != 0)
            compute_peak = grad_peak is not None and bool(use_peak.any())
            compute_spike = grad_spike is not None and bool(use_spike.any())
            compute_boundary = (
                ctx.needs_input_grad[0]
                and grad_max is not None
                and bool(((kinds == 2) & (gmax != 0)).any())
            )
            derivative_count = (
                order.numel()
                if compute_peak or compute_spike or compute_boundary
                else 0
            )
            for start in range(0, derivative_count, _BACKWARD_BLOCK_SIZE):
                end = start + _BACKWARD_BLOCK_SIZE
                pos, batch = order[start:end], event_batches[start:end]
                event_times = times[pos].double()[None, :]
                event_weight = weight[:, indices[pos]].double()
                prefix = event_ranks[start:end][None, :] < accepted[batch].T
                if compute_peak or compute_boundary:
                    peak_mask = prefix & (event_times < peak[batch].T)
                    _, first, second = _psp(
                        (peak[batch].T - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    if compute_boundary:
                        peak_slope.index_add_(
                            0,
                            batch,
                            (torch.where(peak_mask, first, 0) * event_weight).T,
                        )
                    if compute_peak:
                        peak_curvature.index_add_(
                            0,
                            batch,
                            (torch.where(peak_mask, second, 0) * event_weight).T,
                        )
                if compute_spike:
                    spike_mask = (
                        prefix
                        & finite_spike[batch].T
                        & (event_times < safe_spike[batch].T)
                    )
                    _, spike_first, _ = _psp(
                        (safe_spike[batch].T - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    spike_slope.index_add_(
                        0,
                        batch,
                        (torch.where(spike_mask, spike_first, 0) * event_weight).T,
                    )
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
            peak_factor = torch.where(use_peak, gpeak, 0) / torch.where(
                use_peak, peak_curvature, 1
            )
            spike_factor = torch.where(use_spike, gspike, 0) / torch.where(
                use_spike, spike_slope, 1
            )
            contribution_count = (
                order.numel()
                if grad_max is not None or compute_peak or compute_spike
                else 0
            )
            for start in range(0, contribution_count, _BACKWARD_BLOCK_SIZE):
                end = start + _BACKWARD_BLOCK_SIZE
                pos, batch = order[start:end], event_batches[start:end]
                event_times = times[pos].double()[None, :]
                columns = indices[pos]
                event_weight = weight[:, columns].double()
                prefix = event_ranks[start:end][None, :] < accepted[batch].T
                dw, dt = torch.zeros_like(event_weight), torch.zeros_like(event_weight)
                if grad_max is not None or compute_peak:
                    peak_mask = prefix & (event_times < peak[batch].T)
                    value, first, second = _psp(
                        (peak[batch].T - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    dw = torch.where(
                        peak_mask,
                        gmax[batch].T * value
                        - (peak_factor[batch].T * first if compute_peak else 0),
                        0,
                    )
                    dt = torch.where(
                        peak_mask,
                        event_weight
                        * (
                            -gmax[batch].T * first
                            + (peak_factor[batch].T * second if compute_peak else 0)
                        ),
                        0,
                    )
                if compute_spike:
                    spike_mask = (
                        prefix
                        & finite_spike[batch].T
                        & (event_times < safe_spike[batch].T)
                    )
                    spike_value, spike_first, _ = _psp(
                        (safe_spike[batch].T - event_times).clamp_min(0),
                        slow,
                        fast,
                        delta,
                        normalizer,
                    )
                    dw -= torch.where(
                        spike_mask, spike_factor[batch].T * spike_value, 0
                    )
                    dt += torch.where(
                        spike_mask,
                        spike_factor[batch].T * event_weight * spike_first,
                        0,
                    )
                if ctx.needs_input_grad[3]:
                    weight_gradient.index_add_(1, columns, dw)
                if ctx.needs_input_grad[0]:
                    time_gradient.index_add_(0, pos, dt.sum(0))
            boundary = kinds == 2
            if ctx.needs_input_grad[0] and bool(boundary.any()):
                time_gradient.index_add_(
                    0,
                    peak_events[boundary],
                    gpeak[boundary] + gmax[boundary] * peak_slope[boundary],
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
    内部使用 FP64，整批样本并行推进有界事件块（CPU 128、CUDA 512 个事件组），
    不展开均匀时间轴。CUDA 仍有块级同步；Torch 参考临时矩阵随 batch 大小及块宽平方增长。
    安装 Triton 时，CUDA 用仿射前缀扫描计算块状态，并融合阈值求根和仅使用 v_max 的权重反向；
    否则使用 PyTorch 参考实现。时刻导数仍按整批事件重算；
    仅支持对权重及输入时刻的一阶反向。

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
    and weight dtypes. Computation uses FP64 and advances samples together in
    bounded event blocks
    (128 groups on CPU, 512 on CUDA), without a uniform time grid. CUDA still has
    block-level synchronization. Torch reference temporary matrices scale with batch size
    and squared block width. With Triton installed, CUDA uses affine prefix scans
    for block states and fused threshold roots and v_max-only weight backward;
    otherwise it uses the Torch
    reference. Time derivatives are still recomputed across batched events.
    Only first-order weight and input-time derivatives are supported.

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
    _validate_events(spike_times, input_indices, batch_offsets, weight)
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


def _validate_events(spike_times, input_indices, batch_offsets, weight):
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


def _srm_parameters(duration, threshold, max_spikes):
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in (duration, threshold)
    ):
        raise TypeError("duration and v_threshold must be fixed real scalars.")
    if not all(math.isfinite(value) and value > 0 for value in (duration, threshold)):
        raise ValueError("duration and v_threshold must be finite and positive.")
    if (
        not isinstance(max_spikes, int)
        or isinstance(max_spikes, bool)
        or max_spikes < 1
    ):
        raise ValueError("max_spikes must be a positive integer.")
    if not math.isfinite(threshold * max_spikes):
        raise ValueError("The output capacity is not representable in FP64.")


def srm(
    spike_times: torch.Tensor,
    input_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    weight: torch.Tensor,
    duration: float,
    v_threshold: float = 1.0,
    max_spikes: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <timing-functional-srm-cn>` | :ref:`English <timing-functional-srm-en>`

    ----

    .. _timing-functional-srm-cn:

    * **中文**

    连续事件 SRM，输入核为 ``epsilon(u)=max(u,0)``，历史输出核为
    ``eta(u)=-v_threshold*H(u)``。电压上穿阈值时发放并归零，输入电流
    保留，继续接收输入；每条连接及每个输出允许多个脉冲。静息电位为零，
    每次调用是独立试次。同刻输出优先于输入，相切不发放，窗口终点上穿计入。

    对未复位响应的第 k 个阈值 ``k*v_threshold`` 找到首次上穿区间，解析
    求时刻，不展开时间网格。内部 FP64，返回时刻/权重的所选分支由 PyTorch
    自动微分；事件创建、删除或排序切换处不保证唯一导数。没有 loss、替代
    梯度或沉默恢复。容量不足报错，不截断输出。该固定核模型不含泄漏或
    绝对不应期，也不支持任意响应核。

    :param spike_times: 有限 FP32/FP64 时刻 ``[E]``，窗外事件忽略且梯度为零。
    :type spike_times: torch.Tensor
    :param input_indices: int64 ``[E]``，范围 ``[0,I)``，可重复或无序。
    :type input_indices: torch.Tensor
    :param batch_offsets: int64 ``[B+1]``，从 0 到 E 的非递减边界，支持空段。
    :type batch_offsets: torch.Tensor
    :param weight: 有限 ``[O,I]`` 权重，与时刻同 dtype/device，单位为电压/时间。
        所有输入须在同一 CPU/CUDA 设备，I、O 均为正。
    :type weight: torch.Tensor
    :param duration: 有限正数观察窗终点，窗口 ``[0,duration]``，单位与时刻一致。
    :type duration: float
    :param v_threshold: 有限正数阈值与复位幅度，默认 1，单位为电压。
    :type v_threshold: float
    :param max_spikes: 每样本每输出的最大事件容量，默认 64，正整数。
    :type max_spikes: int
    :return: 紧凑输出 ``(times,indices,offsets)``，形状 ``[S]``、``[S]``、
        ``[B+1]``，同输入设备；时刻同输入浮点 dtype，元数据为 int64。
        按样本、输出索引、发放次序排列，样本内不保证全局时间有序。
        静息/空样本无事件，可直接作为下一 SRM 层输入。
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises TypeError: 参数类型或输入 dtype 不受支持。
    :raises ValueError: 参数、形状、设备、有限性、索引或 CSR 边界非法。
    :raises RuntimeError: 超出事件容量，或状态/所选根不可表示。

    ----

    .. _timing-functional-srm-en:

    * **English**

    Continuous-event SRM with input kernel ``epsilon(u)=max(u,0)`` and output
    history kernel ``eta(u)=-v_threshold*H(u)``. An upward threshold crossing
    emits a spike and resets voltage to zero, retaining input current and
    accepting further inputs. Connections and outputs allow multiple spikes.
    Rest is zero and calls are independent trials. Outputs precede simultaneous
    inputs; grazing does not fire; upward crossings at the window endpoint count.

    Locate the first upward crossing of each level ``k*v_threshold`` in the
    unreset response and solve its time analytically without a time grid.
    Internals use FP64; PyTorch differentiates the selected time/weight branch.
    Event creation, deletion and ordering boundaries need not have unique
    derivatives. No loss, surrogate or silent-neuron recovery is built in.
    Capacity overflow raises instead of truncating. This fixed-kernel model has
    no leak or absolute refractory period and does not accept arbitrary kernels.

    :param spike_times: Finite FP32/FP64 times ``[E]``. Out-of-window events are
        ignored with zero gradient.
    :type spike_times: torch.Tensor
    :param input_indices: int64 ``[E]`` in ``[0,I)``; repetitions and disorder allowed.
    :type input_indices: torch.Tensor
    :param batch_offsets: Nondecreasing int64 ``[B+1]`` from 0 to E; empty segments allowed.
    :type batch_offsets: torch.Tensor
    :param weight: Finite ``[O,I]`` weights matching time dtype/device, in voltage/time
        units. All inputs share a CPU/CUDA device; I and O are positive.
    :type weight: torch.Tensor
    :param duration: Finite positive endpoint of ``[0,duration]``, in the times' units.
    :type duration: float
    :param v_threshold: Finite positive voltage threshold and reset amplitude, default 1.
    :type v_threshold: float
    :param max_spikes: Positive per-trial, per-output event capacity, default 64.
    :type max_spikes: int
    :return: Compact ``(times,indices,offsets)`` of shapes ``[S]``, ``[S]``, ``[B+1]``
        on the input device. Times share the input floating dtype; metadata is int64.
        Ordered by trial, output index and spike number, not globally by time within
        trials. Resting/empty trials emit nothing. Directly usable by another SRM.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises TypeError: For unsupported parameter types or dtypes.
    :raises ValueError: For invalid parameters, shapes, devices, finite values,
        indices or CSR boundaries.
    :raises RuntimeError: For capacity overflow or unrepresentable states/selected roots.
    """
    _srm_parameters(duration, v_threshold, max_spikes)
    _validate_events(spike_times, input_indices, batch_offsets, weight)
    batch_count, output_count = batch_offsets.numel() - 1, weight.shape[0]
    batches = torch.repeat_interleave(
        torch.arange(batch_count, device=spike_times.device),
        batch_offsets.diff(),
        output_size=spike_times.numel(),
    )
    order = ((spike_times >= 0) & (spike_times <= duration)).nonzero().flatten()
    order = order[torch.argsort(spike_times[order], stable=True)]
    order = order[torch.argsort(batches[order], stable=True)]
    counts = torch.bincount(batches[order], minlength=batch_count)
    width = int(counts.max()) if batch_count else 0
    empty_times = spike_times[:0] + weight.reshape(-1)[0] * 0
    empty_indices = input_indices[:0]
    empty_offsets = batch_offsets.new_zeros(batch_count + 1)
    if not width:
        return empty_times, empty_indices, empty_offsets

    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    rows = batches[order]
    columns = torch.arange(order.numel(), device=order.device) - offsets[rows]
    times = spike_times.double().new_full((batch_count, width), duration)
    times[rows, columns] = spike_times[order].double()
    indices = input_indices.new_zeros((batch_count, width))
    indices[rows, columns] = input_indices[order]
    valid = torch.arange(width, device=order.device)[None, :] < counts[:, None]
    incoming = torch.where(
        valid[:, None, :], weight.double()[:, indices].permute(1, 0, 2), 0
    )
    current = incoming.cumsum(-1)
    left_voltage = torch.cat(
        (
            current.new_zeros((batch_count, output_count, 1)),
            (current[..., :-1] * times.diff()[:, None, :]).cumsum(-1),
        ),
        dim=-1,
    )
    right_times = torch.cat(
        (times[:, 1:], times.new_full((batch_count, 1), duration)), 1
    )
    right_voltage = left_voltage + current * (right_times - times)[:, None, :]
    if not bool(torch.isfinite(current).all() & torch.isfinite(right_voltage).all()):
        raise RuntimeError("SRM state is not representable in FP64.")
    # A constant reset subtracts one threshold: spike k crosses raw level k*threshold.
    running_max = right_voltage.cummax(-1).values.clamp_min(0)
    spike_counts = torch.floor(running_max[..., -1] / v_threshold)
    if bool((spike_counts > max_spikes).any()):
        raise RuntimeError("SRM exceeded max_spikes; increase the output capacity.")
    spike_counts = spike_counts.to(torch.int64)
    capacity = int(spike_counts.max())
    if not capacity:
        return empty_times, empty_indices, empty_offsets
    levels = (
        torch.arange(1, capacity + 1, device=order.device, dtype=torch.float64)
        * v_threshold
    ).expand(batch_count, output_count, capacity)
    active = (
        torch.arange(capacity, device=order.device)[None, None, :]
        < spike_counts[..., None]
    )
    interval = torch.searchsorted(running_max.contiguous(), levels.contiguous())
    interval = interval.clamp_max(width - 1)
    slope = current.gather(-1, interval)
    if bool((active & (slope <= 0)).any()):
        raise RuntimeError("SRM selected a non-rising threshold crossing.")
    voltage = torch.where(active, left_voltage.gather(-1, interval), 0)
    slope = torch.where(active, slope, 1)
    selected_times = times[:, None, :].expand_as(current).gather(-1, interval)
    roots = selected_times + (levels - voltage) / slope
    endpoints = right_times[:, None, :].expand_as(current).gather(-1, interval)
    tolerance = 8 * torch.finfo(torch.float64).eps * duration
    if bool(
        (
            active
            & (
                ~torch.isfinite(roots)
                | (roots <= selected_times)
                | (roots > endpoints + tolerance)
            )
        ).any()
    ):
        raise RuntimeError("SRM selected root is not representable in its interval.")
    output_times = roots[active]
    output_times = output_times.to(spike_times.dtype)
    if not bool(torch.isfinite(output_times).all()):
        raise RuntimeError("SRM output times overflow their dtype.")
    output_indices = torch.arange(output_count, device=order.device)[
        None, :, None
    ].expand_as(active)[active]
    totals = spike_counts.sum(1)
    output_offsets = torch.cat((totals.new_zeros(1), totals.cumsum(0)))
    return output_times, output_indices, output_offsets
