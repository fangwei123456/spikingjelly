import math

import torch
from torch.autograd.function import once_differentiable

__all__ = ["tempotron"]

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
            # Retain zero-current events in order for their weight derivatives.
            nonzero = (grouped_weight != 0).any(0)
            unique, first_events, group_batches = (
                unique[nonzero],
                first_events[nonzero],
                group_batches[nonzero],
            )
            grouped_weight = grouped_weight[:, nonzero]
        else:
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
            delays = (block_times[:, :, None] - block_times[:, None, :]).clamp_min(0)
            kernel = torch.exp(-delays / slow) * _phi(delays, delta) / normalizer
            block_voltage = _voltage(
                voltage[..., None],
                current[..., None],
                (block_times - origin[:, None])[:, None, :],
                slow,
                delta,
                normalizer,
            ) + incoming @ kernel.transpose(-1, -2)
            decay = torch.exp(-delays / fast).tril()
            block_current = current[..., None] * torch.exp(
                -((block_times - origin[:, None])[:, None, :]) / fast
            ) + incoming @ decay.transpose(-1, -2)
            del delays, kernel, decay
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
        event_batches = torch.repeat_interleave(
            torch.arange(shape[0], device=times.device),
            offsets.diff(),
            output_size=order.numel(),
        )
        event_ranks = (
            torch.arange(order.numel(), device=times.device) - offsets[event_batches]
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
        compute_boundary = grad_max is not None and bool(
            ((kinds == 2) & (gmax != 0)).any()
        )
        derivative_count = (
            order.numel() if compute_peak or compute_spike or compute_boundary else 0
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
                        0, batch, (torch.where(peak_mask, first, 0) * event_weight).T
                    )
                if compute_peak:
                    peak_curvature.index_add_(
                        0, batch, (torch.where(peak_mask, second, 0) * event_weight).T
                    )
            if compute_spike:
                spike_mask = (
                    prefix & finite_spike[batch].T & (event_times < safe_spike[batch].T)
                )
                _, spike_first, _ = _psp(
                    (safe_spike[batch].T - event_times).clamp_min(0),
                    slow,
                    fast,
                    delta,
                    normalizer,
                )
                spike_slope.index_add_(
                    0, batch, (torch.where(spike_mask, spike_first, 0) * event_weight).T
                )
        if compute_peak and bool(
            (use_peak & (~torch.isfinite(peak_curvature) | (peak_curvature >= 0))).any()
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
                    prefix & finite_spike[batch].T & (event_times < safe_spike[batch].T)
                )
                spike_value, spike_first, _ = _psp(
                    (safe_spike[batch].T - event_times).clamp_min(0),
                    slow,
                    fast,
                    delta,
                    normalizer,
                )
                dw -= torch.where(spike_mask, spike_factor[batch].T * spike_value, 0)
                dt += torch.where(
                    spike_mask, spike_factor[batch].T * event_weight * spike_first, 0
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
    不展开均匀时间轴。CUDA 仍有块级同步；临时矩阵内存随 batch 大小及块宽平方增长，
    不使用专用 GPU 核。反向按整批事件重算被使用输出所需的导数；
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
    block-level synchronization. Temporary matrix memory scales with batch size
    and squared block width; no specialized GPU kernel is used. Backward processes
    events across the batch, recomputing only derivatives needed by used outputs.
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
