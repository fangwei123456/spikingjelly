import torch
from torch import autograd

try:
    import triton
except BaseException as e:
    import logging
    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.flexsn.wrapper: {e}")
    triton = dummy.DummyImport()

from ..triton_utils import type_dict, contiguous_and_device_guard
from ..triton_utils import amp_custom_fwd, amp_custom_bwd
from .info import FlexSNInfo


__all__ = [
    "flexsn_inference",
    "flexsn_inference_final_state",
    "flexsn_forward",
    "flexsn_backward",
    "FlexSNFunction",
]


def _num_elements_per_step(x: torch.Tensor) -> int:
    if x.shape[0] > 0:
        return x[0].numel()
    return x.new_empty(x.shape[1:]).numel()


def flexsn_inference(f, info: FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    outputs = [
        torch.empty_like(x_example) for _ in range(info.num_outputs + info.num_states)
    ]
    if T == 0:
        return tuple(outputs)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    f[grid](
        *args,
        *outputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(outputs)


def flexsn_inference_final_state(f, info: FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    output_seqs = [torch.empty_like(x_example) for _ in range(info.num_outputs)]
    init_states = args[info.num_inputs : info.num_inputs + info.num_states]
    final_states = [
        init_states[i].new_empty(init_states[i].shape)
        if i < len(init_states)
        else x_example.new_empty(x_example.shape[1:])
        for i in range(info.num_states)
    ]
    if T == 0:
        final_states = [
            (
                init_states[i].clone()
                if i < len(init_states)
                else x_example.new_zeros(x_example.shape[1:])
            )
            for i in range(info.num_states)
        ]
        return tuple([*output_seqs, *final_states])
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    f[grid](
        *args,
        *output_seqs,
        *final_states,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple([*output_seqs, *final_states])


def flexsn_forward(f, info: FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    returns = [torch.empty_like(x_example) for _ in range(info.num_fwd_kernel_returns)]
    dtype = x_example.dtype
    if T == 0:
        return tuple(returns)
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
    NCL = _num_elements_per_step(grad_example)
    grad_inputs = [
        (
            torch.zeros_like(grad_example)
            if T == 0
            else torch.empty_like(grad_example)
        )
        for _ in range(info.num_inputs)
    ]
    grad_state_seq_examples = args[
        info.num_outputs : info.num_outputs + info.num_states
    ]
    # State-sequence gradients include the leading time dimension. The wrapper
    # returns gradients for the initial states, so their templates are shape[1:].
    grad_inputs += [
        (
            grad_state_seq_examples[i].new_zeros(grad_state_seq_examples[i].shape[1:])
            if T == 0
            else grad_state_seq_examples[i].new_empty(
                grad_state_seq_examples[i].shape[1:]
            )
        )
        if i < len(grad_state_seq_examples) and grad_state_seq_examples[i] is not None
        else (
            grad_example.new_zeros(grad_example.shape[1:])
            if T == 0
            else grad_example.new_empty(grad_example.shape[1:])
        )
        for i in range(info.num_states)
    ]
    dtype = grad_example.dtype
    if T == 0:
        return tuple(grad_inputs)
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
