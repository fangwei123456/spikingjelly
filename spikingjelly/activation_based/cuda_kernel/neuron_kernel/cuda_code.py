from ..auto_cuda import cfunction


def _neuronal_hard_reset(
    v_next: str, h: str, spike: str, v_reset: str, dtype: str
) -> str:
    if dtype == "float":
        return f"{v_next} = {h} * (1.0f - {spike}) + {v_reset} * {spike};"
    if dtype == "half2":
        return f"{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), {spike}), __hmul2({v_reset}, {spike}));"
    raise NotImplementedError(dtype)


def _neuronal_soft_reset(v_next: str, h: str, spike: str, v_th: str, dtype: str) -> str:
    if dtype == "float":
        return f"{v_next} = {h} - {v_th} * {spike};"
    if dtype == "half2":
        return f"{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));"
    raise NotImplementedError(dtype)


def _neuronal_fire(spike: str, v: str, v_th: str, dtype: str) -> str:
    if dtype == "float":
        return cfunction.heaviside(y=spike, x=f"({v} - {v_th})", dtype=dtype)
    if dtype == "half2":
        return cfunction.heaviside(y=spike, x=f"__hsub2({v}, {v_th})", dtype=dtype)
    raise NotImplementedError(dtype)
