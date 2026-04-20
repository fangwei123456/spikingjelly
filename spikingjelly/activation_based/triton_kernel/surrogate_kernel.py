import torch

from .. import surrogate

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging

    from . import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.compress: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()


__all__ = [
    "sg_triton",
    "SG_TRITON_IDS",
    "resolve_sg_triton_id_and_alpha",
]

SG_TRITON_IDS: dict[type[surrogate.SurrogateFunctionBase], int] = {
    surrogate.Sigmoid: 0,
    surrogate.ATan: 1,
}


@triton.jit
def sg_triton(h, alpha, sg_triton_id: tl.constexpr):
    if sg_triton_id == 0:  # Sigmoid
        sg = tl.sigmoid(h.to(tl.float32) * alpha)
        sg = alpha * sg * (1.0 - sg)
    elif sg_triton_id == 1:  # ATan
        sg = 3.141592653589793 * h * alpha / 2.0
        sg = alpha / 2.0 / tl.fma(sg, sg, 1.0)
    else:
        sg = tl.zeros_like(h)
    return sg.to(h.dtype)


def resolve_sg_triton_id_and_alpha(surrogate_function) -> tuple[int, float]:
    sg_type = type(surrogate_function)
    sg_triton_id = None
    for supported_type, candidate_id in SG_TRITON_IDS.items():
        if isinstance(surrogate_function, supported_type):
            sg_triton_id = candidate_id
            break
    if sg_triton_id is None:
        supported_names = tuple(t.__name__ for t in SG_TRITON_IDS)
        raise NotImplementedError(
            f"Triton backend only supports surrogate functions "
            f"{supported_names}, but got {sg_type.__name__}."
        )

    if not hasattr(surrogate_function, "alpha"):
        raise TypeError(
            "Triton backend requires surrogate_function.alpha, but got "
            f"{sg_type.__name__} without 'alpha'. Please use a surrogate function with alpha "
            "(e.g., surrogate.Sigmoid / surrogate.ATan)."
        )

    alpha = surrogate_function.alpha
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1:
            raise TypeError(
                "surrogate_function.alpha must be a scalar for Triton backend, "
                f"but got tensor with shape={tuple(alpha.shape)}."
            )
        alpha = alpha.item()

    if not isinstance(alpha, (int, float)):
        raise TypeError(
            "surrogate_function.alpha must be a real scalar for Triton backend, "
            f"but got {type(alpha).__name__}."
        )

    return sg_triton_id, float(alpha)
