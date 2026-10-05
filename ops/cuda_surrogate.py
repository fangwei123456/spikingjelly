import ast
from typing import Callable

from spikingjelly.activation_based import surrogate


def _surrogate_cuda_code(
    surrogate_function: surrogate.SurrogateFunctionBase, dtype: str
) -> str:
    cuda_codes = getattr(surrogate_function, "cuda_codes", None)
    if not callable(cuda_codes):
        raise TypeError("CuPy backend requires callable surrogate_function.cuda_codes.")
    codes = cuda_codes(y=f"const {dtype} grad_s_to_h", x="over_th", dtype=dtype)
    # Inductor prints op arguments in generated comments; raw newlines break Python.
    return repr(codes)


def _decode_cuda_code(codes: str) -> str:
    source = ast.literal_eval(codes)
    if not isinstance(source, str):
        raise ValueError("CuPy surrogate code must decode to a string.")
    return source


def _cuda_codes_callable(codes: str) -> Callable[..., str]:
    source = _decode_cuda_code(codes)

    def cuda_codes(**_) -> str:
        return source

    return cuda_codes
