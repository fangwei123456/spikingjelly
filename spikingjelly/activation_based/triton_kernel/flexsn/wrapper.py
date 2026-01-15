import torch
from torch import autograd

try:
    import triton
except BaseException as e:
    import logging
    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.flexsn.wrapper: {e}")
    triton = dummy.DummyTriton

from ..triton_utils import type_dict, contiguous_and_device_guard
from ..triton_utils import amp_custom_fwd, amp_custom_bwd
from .info import FlexSNInfo


__all__ = ["FlexSNFunction"]


def flexsn_inference(f, info: FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = x_example[0].numel()
    dtype = x_example.dtype
    outputs = [
        torch.empty_like(x_example) for _ in range(info.num_outputs + info.num_states)
    ]
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    f[grid](
        *args,
        *outputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(outputs)


def flexsn_forward(f, info: FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = x_example[0].numel()
    returns = [torch.empty_like(x_example) for _ in range(info.num_fwd_kernel_returns)]
    dtype = x_example.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    f[grid](
        *args,
        *returns,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(returns)


def flexsn_backward(f, info: FlexSNInfo, *args) -> tuple:
    grad_example = args[0]
    T = grad_example.shape[0]
    NCL = grad_example[0].numel()
    grad_inputs = [torch.empty_like(grad_example) for _ in range(info.num_inputs)]
    grad_inputs += [torch.empty_like(grad_example[0]) for _ in range(info.num_states)]
    dtype = grad_example.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    f[grid](
        *args,
        *grad_inputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(grad_inputs)


class FlexSNFunction(autograd.Function):
    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx,
        fn_inf,
        fn_fwd,
        fn_bwd,
        info: FlexSNInfo,
        *args,  # len = num_inputs; including initial states
    ):
        if any(ctx.needs_input_grad):
            results = flexsn_forward(fn_fwd, info, *args)
            outputs_states = results[: info.num_outputs + info.num_states]
            to_save = []
            for i in info.c2k_return_mapping:
                to_save.append(results[i])
            ctx.save_for_backward(*to_save)
            ctx.fn_bwd = fn_bwd
            ctx.info = info
        else:
            outputs_states = flexsn_inference(fn_inf, info, *args)
        if len(outputs_states) == 1:
            return outputs_states[0]
        else:
            return outputs_states

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, *args):  # len(args) = num_outputs + num_states
        required_results = ctx.saved_tensors
        fn_bwd = ctx.fn_bwd
        grads = flexsn_backward(fn_bwd, ctx.info, *args, *required_results)
        return None, None, None, None, *grads
