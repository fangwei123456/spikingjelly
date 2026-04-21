import math

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

# Maps surrogate class → integer id used as tl.constexpr in sg_triton.
# Adding a new surrogate: (1) assign the next id here, (2) add an elif branch
# in sg_triton below, (3) update resolve_sg_triton_id_and_alpha if needed.
SG_TRITON_IDS: dict[type[surrogate.SurrogateFunctionBase], int] = {
    surrogate.Sigmoid: 0,
    surrogate.ATan: 1,
    surrogate.PiecewiseQuadratic: 2,
    surrogate.PiecewiseExp: 3,
    surrogate.SoftSign: 4,
    surrogate.SuperSpike: 5,
    surrogate.Erf: 6,
}


@triton.jit
def sg_triton(h, alpha, sg_triton_id: tl.constexpr):
    """Surrogate gradient g'(h) in Triton JIT.

    All transcendentals upcast to float32 to avoid fp16 precision issues.
    The result is cast back to the input dtype before returning.
    """
    if sg_triton_id == 0:  # Sigmoid:  alpha * sigmoid(alpha*h) * (1 - sigmoid(alpha*h))
        sg = tl.sigmoid(h.to(tl.float32) * alpha)
        sg = alpha * sg * (1.0 - sg)

    elif sg_triton_id == 1:  # ATan:  alpha / (2 * (1 + (pi/2 * alpha * h)^2))
        sg = 3.141592653589793 * h * alpha / 2.0
        sg = alpha / 2.0 / tl.fma(sg, sg, 1.0)

    elif sg_triton_id == 2:  # PiecewiseQuadratic:  max(0, -alpha^2*|h| + alpha)
        sg = tl.maximum(0.0, alpha - alpha * alpha * tl.abs(h.to(tl.float32)))

    elif sg_triton_id == 3:  # PiecewiseExp:  alpha/2 * exp(-alpha * |h|)
        sg = alpha / 2.0 * tl.exp(-alpha * tl.abs(h.to(tl.float32)))

    elif sg_triton_id == 4:  # SoftSign:  1 / (2 * alpha * (1/alpha + |h|)^2)
        # Only (1/alpha + |h|) is squared; 2*alpha is a linear scale factor.
        # Previous: denom = 2α*(1/α+|h|), sg = 1/denom² → off by factor 2α.
        temp = 1.0 / alpha + tl.abs(h.to(tl.float32))
        sg = 1.0 / (2.0 * alpha * temp * temp)

    elif sg_triton_id == 5:  # SuperSpike:  alpha / (|h| + 1)^2
        denom = tl.abs(h.to(tl.float32)) + 1.0
        sg = alpha / (denom * denom)

    elif sg_triton_id == 6:  # Erf:  alpha/sqrt(pi) * exp(-(alpha*h)^2)
        # 1/sqrt(pi) ≈ 0.5641895835477563; use x*x, ** unsupported on tl tensors
        ha = h.to(tl.float32) * alpha
        sg = 0.5641895835477563 * alpha * tl.exp(-ha * ha)

    else:
        # Unknown id — zero gradient (safe fallback; resolve() catches this earlier)
        sg = tl.zeros_like(h)

    return sg.to(h.dtype)


def resolve_sg_triton_id_and_alpha(surrogate_function) -> tuple[int, float]:
    """Return (sg_triton_id, alpha) for a surrogate function.

    Raises NotImplementedError for unsupported surrogate types.
    """
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
            f"{supported_names}, but got {sg_type.__name__}. "
            f"Use backend='torch' for other surrogate functions."
        )

    if not hasattr(surrogate_function, "alpha"):
        raise TypeError(
            "Triton backend requires surrogate_function.alpha, but got "
            f"{sg_type.__name__} without 'alpha'."
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
