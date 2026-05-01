import torch


def sg_registry_key(sg) -> str:
    return repr(
        (
            sg.__class__.__module__,
            sg.__class__.__qualname__,
            getattr(sg, "spiking", True),
            getattr(sg, "_sg_params", {}),
        )
    )


def replay_and_grad(op, tensor_args, static_args, grad_outputs):
    replay_inputs = []
    grad_inputs = []
    for x in tensor_args:
        if x is None:
            replay_inputs.append(None)
            continue
        t = x.detach().requires_grad_(x.requires_grad)
        replay_inputs.append(t)
        grad_inputs.append(t)

    with torch.enable_grad():
        outputs = op(*replay_inputs, *static_args)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=grad_inputs,
        grad_outputs=grad_outputs[: len(outputs)],
        allow_unused=True,
    )
    grad_map = {}
    j = 0
    for i, x in enumerate(replay_inputs):
        if x is not None:
            grad_map[i] = grads[j]
            j += 1
    return tuple(grad_map.get(i, None) for i in range(len(replay_inputs)))


