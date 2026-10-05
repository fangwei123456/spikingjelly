import triton
import triton.language as tl


@triton.jit
def _surrogate_gradient(x, alpha, SURROGATE: tl.constexpr):
    if SURROGATE == 0:
        s = tl.div_rn(1.0, 1.0 + tl.exp(-alpha * x))
        return (1.0 - s) * s * alpha
    elif SURROGATE == 1:
        z = x * (1.5707963267948966 * alpha)
        # Match production Triton's division: div_rn is slower here and also
        # loses subnormal derivatives on Triton 3.3.
        return (0.5 * alpha) / (1.0 + z * z)
    elif SURROGATE == 2:
        return tl.maximum(0.0, alpha - alpha * alpha * tl.abs(x))
    elif SURROGATE == 3:
        return (0.5 * alpha) * tl.exp(-alpha * tl.abs(x))
    elif SURROGATE == 4:
        z = tl.div_rn(1.0, alpha) + tl.abs(x)
        return tl.div_rn(1.0, 2.0 * alpha * z * z)
    elif SURROGATE == 5:
        z = 1.0 + tl.abs(x)
        return tl.div_rn(alpha, z * z)
    elif SURROGATE == 6:
        z = alpha * x
        return (0.5641895835477563 * alpha) * tl.exp(-z * z)
    else:
        tl.static_assert(False, "Unsupported surrogate")
