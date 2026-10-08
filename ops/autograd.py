"""Autograd helpers shared by registered neuron families."""

import torch


def _save_for_higher_order(ctx, inputs, intermediates):
    positions = tuple(
        index for index, value in enumerate(inputs) if isinstance(value, torch.Tensor)
    )
    ctx.input_template = tuple(
        None if isinstance(value, torch.Tensor) else value for value in inputs
    )
    ctx.tensor_positions = positions
    ctx.intermediate_count = len(intermediates)
    ctx.save_for_backward(*intermediates, *(inputs[index] for index in positions))


def _higher_order_grad(ctx, forward, output_grads):
    saved = ctx.saved_tensors
    tensors = saved[ctx.intermediate_count :]
    inputs = list(ctx.input_template)
    for index, tensor in zip(ctx.tensor_positions, tensors, strict=True):
        inputs[index] = tensor

    with torch.enable_grad():
        outputs = forward(*inputs)
        active = tuple(
            (output, grad)
            for output, grad in zip(outputs, output_grads, strict=True)
            if grad is not None
        )
        differentiable_inputs = tuple(
            tensor for tensor in tensors if tensor.requires_grad
        )
        if not active or not differentiable_inputs:
            return (None,) * len(inputs)
        gradients = torch.autograd.grad(
            tuple(output for output, _ in active),
            differentiable_inputs,
            tuple(grad for _, grad in active),
            create_graph=True,
            allow_unused=True,
        )

    result = [None] * len(inputs)
    for index, gradient in zip(
        (i for i in ctx.tensor_positions if inputs[i].requires_grad),
        gradients,
        strict=True,
    ):
        result[index] = gradient
    return tuple(result)
