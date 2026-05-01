import torch


def replay_and_grad(op, tensor_args, static_args, grad_outputs):
    replay_inputs = []
    grad_inputs = []
    grad_input_indices = []
    for x in tensor_args:
        t = x.detach().requires_grad_(x.requires_grad)
        replay_inputs.append(t)
        if t.requires_grad:
            grad_inputs.append(t)
            grad_input_indices.append(len(replay_inputs) - 1)
    with torch.enable_grad():
        outputs = op(*replay_inputs, *static_args)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    if len(grad_inputs) == 0:
        return tuple(None for _ in replay_inputs)
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=grad_inputs,
        grad_outputs=grad_outputs[: len(outputs)],
        allow_unused=True,
    )
    mapped_grads = [None for _ in replay_inputs]
    for idx, g in zip(grad_input_indices, grads):
        mapped_grads[idx] = g
    return tuple(mapped_grads)

