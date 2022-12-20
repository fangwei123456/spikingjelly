from typing import Callable, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import neuron, monitor, base


def stdp_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike      # shape = [batch_size, N_in]
    trace_post = trace_post - trace_post / tau_post + out_spike # shape = [batch_size, N_out]

    # [batch_size, N_out, N_in] -> [N_out, N_in]
    delta_w_pre = -f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
    delta_w_post = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def mstdp_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike      # shape = [batch_size, C_in]
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike  # shape = [batch_size, C_out]

    # [batch_size, N_out, N_in]
    eligibility = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)) -\
                  f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1))
    return trace_pre, trace_post, eligibility


def mstdpet_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float, tau_trace: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike

    eligibility = f_post(weight) * torch.outer(out_spike, trace_pre) -\
                  f_pre(weight) * torch.outer(trace_post, in_spike)
    return trace_pre, trace_post, eligibility


def stdp_conv2d_single_step(
    conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv2d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv2d has not been implemented!'
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device = in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]  # shape = [batch_size, C_in, h_out, w_out]
            post_spike = out_spike  # shape = [batch_size, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]   # shape = [batch_size_out, C_in]

            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]    # shape = [batch_size, C_in, h_out, w_out]
            tr_post = trace_post    # shape = [batch_size, C_out, h_out, w_out]

            delta_w_pre = - (f_pre(weight) *
			                (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                            .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4]))
            delta_w_post = f_post(weight) * \
                           (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2))\
                           .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w

def stdp_conv1d_single_step(
    conv: nn.Conv1d, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1, ):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv1d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv1d has not been implemented!'
        )

    stride_l = conv.stride[0]

    if conv.padding == (0, ):
        pass
    else:
        pL = conv.padding[0]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pL, pL))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for l in range(conv.weight.shape[2]):
        l_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + l
        pre_spike = in_spike[:, :, l:l_end:stride_l]  # shape = [batch_size, C_in, l_out]
        post_spike = out_spike  # shape = [batch_size, C_out, l_out]
        weight = conv.weight.data[:, :, l]   # shape = [batch_size_out, C_in]

        tr_pre = trace_pre[:, :, l:l_end:stride_l]    # shape = [batch_size, C_in, l_out]
        tr_post = trace_post    # shape = [batch_size, C_out, l_out]

        delta_w_pre = - (f_pre(weight) *
                        (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                        .permute([1, 2, 0, 3]).sum(dim = [2, 3]))
        delta_w_post = f_post(weight) * \
                       (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2))\
                       .permute([1, 2, 0, 3]).sum(dim = [2, 3])
        delta_w[:, :, l] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w

def stdp_multi_step(
    layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
    in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None], 
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x 
):
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]

    if isinstance(layer, nn.Linear):
        stdp_single_step = stdp_linear_single_step

    elif isinstance(layer, nn.Conv1d):
        stdp_single_step = stdp_conv1d_single_step

    elif isinstance(layer, nn.Conv2d):
        stdp_single_step = stdp_conv2d_single_step


    for t in range(T):
        trace_pre, trace_post, dw = stdp_single_step(
            layer, in_spike[t], out_spike[t], trace_pre, trace_post,
            tau_pre, tau_post, f_pre, f_post
        )
        delta_w += dw

    return trace_pre, trace_post, delta_w


class STDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

    def reset(self):
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Linear):
                stdp_f = stdp_linear_single_step
            elif isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Conv1d):
                stdp_f = stdp_conv1d_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if isinstance(self.synapse, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                stdp_f = stdp_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]

            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class MSTDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str, batch_size: float,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.step_mode = step_mode
        self.batch_size = batch_size
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

    def reset(self):
        super(MSTDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, reward, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Conv2d):
                # stdp_f = mstdp_conv2d_single_step
                raise NotImplementedError(self.synapse)
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = mstdp_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if (isinstance(self.synapse, nn.Conv2d) or 
                isinstance(self.synapse, nn.Linear)):
                # stdp_f = mstdp_multi_step
                raise NotImplementedError(self.synapse)
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            if not hasattr(self, "eligibility"):
                self.eligibility = torch.zeros(
                    self.batch_size, *self.synapse.weight.shape, device=self.synapse.weight.device
                )

            dw = (reward.view(-1, 1, 1) * self.eligibility).sum(0)     # [batch_size, N_out, N_in] -> [N_out, N_in]

            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)
        
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]

            self.trace_pre, self.trace_post, self.eligibility = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class MSTDPETLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str, synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre: float, tau_post: float, tau_trace: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_trace = tau_trace
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)
        self.register_memory('trace_e', None)

    def reset(self):
        super(MSTDPETLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, reward, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Conv2d):
                # stdp_f = mstdpet_conv2d_single_step
                raise NotImplementedError(self.synapse)
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = mstdpet_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if (isinstance(self.synapse, nn.Conv2d) or 
                isinstance(self.synapse, nn.Linear)):
                # stdp_f = mstdpet_multi_step
                raise NotImplementedError(self.synapse)
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            if not hasattr(self, "eligibility"):
                self.eligibility = torch.zeros(
                    *self.synapse.weight.shape, device=self.synapse.weight.device
                )

            if self.trace_e is None:
                self.trace_e = 0.

            self.trace_e = self.trace_e * math.exp(-1 / self.tau_trace) + self.eligibility / self.tau_trace

            dw = reward * self.trace_e

            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)
        
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)

            self.trace_pre, self.trace_post, self.eligibility = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post, self.tau_trace,
                self.f_pre, self.f_post
            )

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
