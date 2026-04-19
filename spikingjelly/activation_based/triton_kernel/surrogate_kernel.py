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
    "sigmoid_surrogate_kernel",
    "atan_surrogate_kernel",
    "SG_TRITON_REGISTRY",
    "get_sg_kernel",
]


@triton.jit
def sigmoid_surrogate_kernel(h, alpha):
    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * alpha)
    sg = alpha * sg * (1.0 - sg)
    return sg.to(h.dtype)


@triton.jit
def atan_surrogate_kernel(h, alpha):
    sg = 3.141592653589793 * h * alpha / 2.0
    sg = alpha / 2.0 / tl.fma(sg, sg, 1.0)
    return sg.to(h.dtype)


SG_TRITON_REGISTRY: dict = {
    "Sigmoid": sigmoid_surrogate_kernel,
    "ATan": atan_surrogate_kernel,
}

def get_sg_kernel(sg_type: str):
    try:
        return SG_TRITON_REGISTRY[sg_type]
    except KeyError as e:
        raise NotImplementedError(
            f"Triton backend only supports surrogate functions "
            f"{tuple(SG_TRITON_REGISTRY.keys())}, but got {sg_type}."
        ) from e
