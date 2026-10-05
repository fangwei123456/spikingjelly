from functools import lru_cache
from pathlib import Path

import cupy
import torch

_TYPES = {
    torch.float32: "float",
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
}


@lru_cache(maxsize=72)
def _kernel(path, name, dtype, *template_parameters):
    source = Path(path)
    code = (source.parent.parent / "_cuda.cuh").read_text(encoding="utf-8")
    code += "\n" + source.read_text(encoding="utf-8").replace(
        '#include "../_cuda.cuh"', ""
    )
    arguments = _TYPES[dtype]
    for value in template_parameters:
        arguments += f", {value}"
    expression = f"{name}<{arguments}>"
    module = cupy.RawModule(
        code=code,
        options=("--std=c++17", "--fmad=false"),
        name_expressions=(expression,),
    )
    return module.get_function(expression)
