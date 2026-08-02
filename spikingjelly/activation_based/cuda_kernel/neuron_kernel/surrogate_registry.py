import threading
from typing import Callable

from ... import surrogate

_LOCK = threading.Lock()
_ID_TO_CODES: dict[tuple[int, str], str] = {}
_CODES_TO_ID: dict[tuple[str, str], int] = {}


def _resolve_cuda_code_id(
    surrogate_function: surrogate.SurrogateFunctionBase, dtype: str
) -> int:
    cuda_codes = getattr(surrogate_function, "cuda_codes", None)
    if not callable(cuda_codes):
        raise TypeError("CuPy backend requires callable surrogate_function.cuda_codes.")

    codes = cuda_codes(y=f"const {dtype} grad_s_to_h", x="over_th", dtype=dtype)
    codes_key = (dtype, codes)
    with _LOCK:
        surrogate_id = _CODES_TO_ID.get(codes_key)
        if surrogate_id is None:
            surrogate_id = len(_CODES_TO_ID)
            _CODES_TO_ID[codes_key] = surrogate_id
            _ID_TO_CODES[(surrogate_id, dtype)] = codes
    return surrogate_id


def _cuda_codes(surrogate_id: int, dtype: str) -> str:
    try:
        return _ID_TO_CODES[(surrogate_id, dtype)]
    except KeyError as error:
        raise RuntimeError(
            f"Unknown surrogate_id={surrogate_id} for dtype={dtype}."
        ) from error


def _cuda_codes_callable(surrogate_id: int, dtype: str) -> Callable[..., str]:
    codes = _cuda_codes(surrogate_id, dtype)

    def cuda_codes(**_) -> str:
        return codes

    return cuda_codes
