try:
    import triton
except BaseException as e:
    import logging
    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.flexsn.template: {e}")
    triton = dummy.DummyImport()

from ..torch2triton import compile_triton_code_str
from .info import FlexSNInfo


__all__ = [
    "get_flexsn_inference_kernel",
    "get_flexsn_inference_final_state_kernel",
    "get_flexsn_forward_kernel",
    "get_flexsn_forward_final_state_kernel",
    "get_flexsn_backward_kernel",
    "get_flexsn_backward_final_state_kernel",
]


INDENTATION = " " * 4


def _join_signature(parts):
    return f",\n{INDENTATION}".join(parts)


def _join_args(parts):
    return ", ".join(parts)

init_state_load_template = """
    {name}_init_ptrs = tl.make_block_ptr(
        {name}_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    {name} = tl.load(
        {name}_init_ptrs, boundary_check=(1,), padding_option="zero"
    )
"""

grad_init_state_store_template = """
    {name}_init_ptrs = tl.make_block_ptr(
        {name}_init_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(t, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    convert_and_store({name}_init_ptrs, {name}_accumulate, boundary_check=(1,))
    # tl.store({name}_ptrs, {name}, boundary_check=(1,))
"""

store_template = """
        {name}_ptrs = tl.make_block_ptr(
            {name}_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        convert_and_store({name}_ptrs, {name}, boundary_check=(1,))
        # tl.store({name}_ptrs, {name}, boundary_check=(1,))
"""

final_state_store_template = """
    {name}_final_ptrs = tl.make_block_ptr(
        {name}_final_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    convert_and_store({name}_final_ptrs, {name}, boundary_check=(1,))
"""

grad_final_state_load_template = """
    {name}_final_ptrs = tl.make_block_ptr(
        {name}_final_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0)
    )
    {name}_accumulate = tl.load(
        {name}_final_ptrs, boundary_check=(1,), padding_option="zero"
    )
"""

load_template = """
        {name}_ptrs = tl.make_block_ptr(
            {name}_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        {name} = tl.load(
            {name}_ptrs, boundary_check=(1,), padding_option="zero"
        )
"""

BASE_AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
    ]"""

BASE_AUTOTUNE_TAG = "b256w4"
INFERENCE_AUTOTUNE_CONFIGS = BASE_AUTOTUNE_CONFIGS
INFERENCE_AUTOTUNE_TAG = BASE_AUTOTUNE_TAG
BACKWARD_AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_NCL": 128}, num_warps=2),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ]"""
BACKWARD_AUTOTUNE_TAG = "bw128_2_256_4_512_8"
INFERENCE_FINAL_STATE_AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ]"""
INFERENCE_FINAL_STATE_AUTOTUNE_TAG = "i512w8"
FORWARD_FINAL_STATE_AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_NCL": 128}, num_warps=2),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
    ]"""
FORWARD_FINAL_STATE_AUTOTUNE_TAG = "f128w2_256w4"

kernel_template = """import triton
import triton.language as tl


@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)

{core_str}

@triton.autotune(
    configs={autotune_configs},
    key={autotune_key},
    restore_value=[{autotune_restore}],
)
@triton.jit
def flexsn_{kernel_type}_kernel_{hash}(
    {kernel_input_signature}, # inputs (including init states)
    {kernel_output_signature}, # outputs
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr{meta_signature},
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    {init_state_loads}

    for t in tl.static_range({loop_range}):
        {loads}

        {computes}

        {stores}

    {tail}
"""


def get_flexsn_inference_kernel(
    core_str: str, core_name: str, info: FlexSNInfo, verbose: bool = False
):
    hash = f"{core_name[-8:]}_{INFERENCE_AUTOTUNE_TAG}"
    num_inputs = info.num_inputs
    num_states = info.num_states
    num_outputs = info.num_outputs

    kernel_input_signature = _join_signature(
        [f"x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"v{i}_init_ptr" for i in range(num_states)]
    )

    kernel_output_signature = _join_signature(
        [f"s{i}_seq_ptr" for i in range(num_outputs)]
        + [f"v{i}_seq_ptr" for i in range(num_states)]
    )

    autotune_restore = _join_args(
        [f'"s{i}_seq_ptr"' for i in range(num_outputs)]
        + [f'"v{i}_seq_ptr"' for i in range(num_states)]
    )

    init_state_loads = "".join(
        [
            init_state_load_template.format(
                name=f"v{i}",
            )
            for i in range(num_states)
        ]
    )

    loads = "".join([load_template.format(name=f"x{i}") for i in range(num_inputs)])
    stores = "".join([store_template.format(name=f"s{i}") for i in range(num_outputs)])
    stores += "".join([store_template.format(name=f"v{i}") for i in range(num_states)])

    lhs = _join_args(
        [f"s{i}" for i in range(num_outputs)] + [f"v{i}" for i in range(num_states)]
    )
    core_args = _join_args(
        [f"x{i}" for i in range(num_inputs)] + [f"v{i}" for i in range(num_states)]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=INFERENCE_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype"]',
        meta_signature="",
        autotune_restore=autotune_restore,
        kernel_type="inference",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="0, T, 1",
        loads=loads,
        computes=f"{lhs} = {core_name}({core_args})",
        stores=stores,
        tail="",
    ).strip()
    kernel_name = f"flexsn_inference_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn inference kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


def get_flexsn_inference_final_state_kernel(
    core_str: str, core_name: str, info: FlexSNInfo, verbose: bool = False
):
    hash = f"{core_name[-8:]}_{INFERENCE_FINAL_STATE_AUTOTUNE_TAG}"
    num_inputs = info.num_inputs
    num_states = info.num_states
    num_outputs = info.num_outputs

    kernel_input_signature = _join_signature(
        [f"x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"v{i}_init_ptr" for i in range(num_states)]
    )

    kernel_output_signature = _join_signature(
        [f"s{i}_seq_ptr" for i in range(num_outputs)]
        + [f"v{i}_final_ptr" for i in range(num_states)]
    )

    autotune_restore = _join_args(
        [f'"s{i}_seq_ptr"' for i in range(num_outputs)]
        + [f'"v{i}_final_ptr"' for i in range(num_states)]
    )

    init_state_loads = "".join(
        [init_state_load_template.format(name=f"v{i}") for i in range(num_states)]
    )

    loads = "".join([load_template.format(name=f"x{i}") for i in range(num_inputs)])
    stores = "".join([store_template.format(name=f"s{i}") for i in range(num_outputs)])
    tail = "".join(
        [final_state_store_template.format(name=f"v{i}") for i in range(num_states)]
    )

    lhs = _join_args(
        [f"s{i}" for i in range(num_outputs)] + [f"v{i}" for i in range(num_states)]
    )
    core_args = _join_args(
        [f"x{i}" for i in range(num_inputs)] + [f"v{i}" for i in range(num_states)]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=INFERENCE_FINAL_STATE_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype"]',
        meta_signature="",
        autotune_restore=autotune_restore,
        kernel_type="inference_final_state",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="0, T, 1",
        loads=loads,
        computes=f"{lhs} = {core_name}({core_args})",
        stores=stores,
        tail=tail,
    ).strip()
    kernel_name = f"flexsn_inference_final_state_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn inference-final-state kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


def get_flexsn_forward_kernel(
    core_str: str,
    core_name: str,
    info: FlexSNInfo,
    verbose: bool = False,
):
    hash = f"{core_name[-8:]}_{BASE_AUTOTUNE_TAG}"
    num_inputs = info.num_inputs
    num_states = info.num_states
    fwd_kernel_returns = info.fwd_kernel_returns  # unique
    fwd_core_recipients = info.fwd_core_recipients  # `_` for duplicates

    kernel_input_signature = _join_signature(
        [f"x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"v{i}_init_ptr" for i in range(num_states)]
    )

    kernel_output_signature = f",\n{INDENTATION}".join(
        [f"{r}_seq_ptr" for r in fwd_kernel_returns]
    )

    autotune_restore = ", ".join([f'"{r}_seq_ptr"' for r in fwd_kernel_returns])

    init_state_loads = "".join(
        [
            init_state_load_template.format(
                name=f"v{i}",
            )
            for i in range(num_states)
        ]
    )

    loads = "".join([load_template.format(name=f"x{i}") for i in range(num_inputs)])
    stores = "".join([store_template.format(name=r) for r in fwd_kernel_returns])
    lhs = _join_args([r for r in fwd_core_recipients])
    core_args = _join_args(
        [f"x{i}" for i in range(num_inputs)] + [f"v{i}" for i in range(num_states)]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=BASE_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype"]',
        meta_signature="",
        autotune_restore=autotune_restore,
        kernel_type="forward",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="0, T, 1",
        loads=loads,
        computes=f"{lhs} = {core_name}({core_args})",
        stores=stores,
        tail="",
    ).strip()
    kernel_name = f"flexsn_forward_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generating flexsn forward kernel:")
        print("```")
        print(kernel_str)
        print("```")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


def get_flexsn_forward_final_state_kernel(
    core_str: str,
    core_name: str,
    info: FlexSNInfo,
    verbose: bool = False,
):
    hash = f"{core_name[-8:]}_{FORWARD_FINAL_STATE_AUTOTUNE_TAG}"
    num_inputs = info.num_inputs
    num_states = info.num_states
    num_outputs = info.num_outputs

    saved_non_output_indices = []
    seen_saved_non_output_indices = set()
    for idx in info.c2k_return_mapping:
        if idx < num_outputs or idx in seen_saved_non_output_indices:
            continue
        saved_non_output_indices.append(idx)
        seen_saved_non_output_indices.add(idx)

    saved_non_output_returns = [
        info.fwd_kernel_returns[idx] for idx in saved_non_output_indices
    ]

    kernel_input_signature = _join_signature(
        [f"x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"v{i}_init_ptr" for i in range(num_states)]
    )

    kernel_output_signature = f",\n{INDENTATION}".join(
        [f"s{i}_seq_ptr" for i in range(num_outputs)]
        + [f"v{i}_final_ptr" for i in range(num_states)]
        + [f"{name}_seq_ptr" for name in saved_non_output_returns]
    )

    autotune_restore = _join_args(
        [f'"s{i}_seq_ptr"' for i in range(num_outputs)]
        + [f'"v{i}_final_ptr"' for i in range(num_states)]
        + [f'"{name}_seq_ptr"' for name in saved_non_output_returns]
    )

    init_state_loads = "".join(
        [init_state_load_template.format(name=f"v{i}") for i in range(num_states)]
    )

    loads = "".join([load_template.format(name=f"x{i}") for i in range(num_inputs)])
    stores = "".join([store_template.format(name=f"s{i}") for i in range(num_outputs)])
    stores += "".join(
        [store_template.format(name=name) for name in saved_non_output_returns]
    )
    tail = "".join(
        [final_state_store_template.format(name=f"v{i}") for i in range(num_states)]
    )

    lhs = _join_args([r for r in info.fwd_core_recipients])
    core_args = _join_args(
        [f"x{i}" for i in range(num_inputs)] + [f"v{i}" for i in range(num_states)]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=FORWARD_FINAL_STATE_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype"]',
        meta_signature="",
        autotune_restore=autotune_restore,
        kernel_type="forward_final_state",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="0, T, 1",
        loads=loads,
        computes=f"{lhs} = {core_name}({core_args})",
        stores=stores,
        tail=tail,
    ).strip()
    kernel_name = f"flexsn_forward_final_state_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generating flexsn forward-final-state kernel:")
        print("```")
        print(kernel_str)
        print("```")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


def get_flexsn_backward_kernel(
    core_str: str,
    core_name: str,
    info: FlexSNInfo,
    verbose: bool = False,
):
    hash = f"{core_name[-8:]}_{BACKWARD_AUTOTUNE_TAG}"
    num_outputs = info.num_outputs
    num_inputs = info.num_inputs
    num_states = info.num_states
    n = len(info.c2k_return_mapping)  # number of intermediate results

    assert n + num_outputs + num_states == len(info.fwd_core_returns)

    kernel_input_signature_parts = [f"grad_s{i}_seq_ptr" for i in range(num_outputs)]
    kernel_input_signature_parts += [f"grad_v{i}_seq_ptr" for i in range(num_states)]
    if n > 0:
        kernel_input_signature_parts += [f"res{i}_b_seq_ptr" for i in range(n)]
    kernel_input_signature = _join_signature(kernel_input_signature_parts)
    # res{i}_b slightly different from res{i}_f in the forward kernel
    # as res{i}_b might be from s{i} or v{i}

    kernel_output_signature = _join_signature(
        [f"grad_x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"grad_v{i}_init_ptr" for i in range(num_states)]
    )

    autotune_restore = _join_args(
        [f'"grad_x{i}_seq_ptr"' for i in range(num_inputs)]
        + [f'"grad_v{i}_init_ptr"' for i in range(num_states)]
    )

    init_state_loads = f"\n{INDENTATION}".join(
        [
            f"grad_v{i}_accumulate = tl.zeros([1, BLOCK_NCL], dtype=dtype)"
            for i in range(num_states)
        ]
    )

    loads = "".join(
        [load_template.format(name=f"grad_s{i}") for i in range(num_outputs)]
    )
    loads += "".join(
        [load_template.format(name=f"grad_v{i}") for i in range(num_states)]
    )
    loads += "".join([load_template.format(name=f"res{i}_b") for i in range(n)])

    stores = "".join(
        [store_template.format(name=f"grad_x{i}") for i in range(num_inputs)]
    )

    computes = f"\n{INDENTATION}{INDENTATION}".join(
        [
            f"grad_v{i}_accumulate = grad_v{i}_accumulate + grad_v{i}"
            for i in range(num_states)
        ]
    )  # accumulate gradients of states
    lhs = _join_args(
        [f"grad_x{i}" for i in range(num_inputs)]
        + [f"grad_v{i}_accumulate" for i in range(num_states)]
    )
    _core_args_parts = []
    if n > 0:
        _core_args_parts += [f"res{i}_b" for i in range(n)]
    _core_args_parts += [f"grad_s{i}" for i in range(num_outputs)]
    _core_args_parts += [f"grad_v{i}_accumulate" for i in range(num_states)]
    core_args = ", ".join(_core_args_parts)
    computes += f"\n{INDENTATION}{INDENTATION}{lhs} = {core_name}({core_args})"

    tail = f"\n{INDENTATION}".join(
        [
            grad_init_state_store_template.format(name=f"grad_v{i}")
            for i in range(num_states)
        ]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=BACKWARD_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype", "NCL_BUCKET"]',
        meta_signature=",\n    NCL_BUCKET: tl.constexpr",
        autotune_restore=autotune_restore,
        kernel_type="backward",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="T-1, -1, -1",
        loads=loads,
        computes=computes,
        stores=stores,
        tail=tail,
    ).strip()
    kernel_name = f"flexsn_backward_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn backward kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


def get_flexsn_backward_final_state_kernel(
    core_str: str,
    core_name: str,
    info: FlexSNInfo,
    verbose: bool = False,
):
    hash = f"{core_name[-8:]}_{BACKWARD_AUTOTUNE_TAG}_final"
    num_outputs = info.num_outputs
    num_inputs = info.num_inputs
    num_states = info.num_states
    n = len(info.c2k_return_mapping)

    kernel_input_signature = f",\n{INDENTATION}".join(
        [f"grad_s{i}_seq_ptr" for i in range(num_outputs)]
    )
    if num_states > 0:
        kernel_input_signature += f",\n{INDENTATION}"
        kernel_input_signature += f",\n{INDENTATION}".join(
            [f"grad_v{i}_final_ptr" for i in range(num_states)]
        )
    if n > 0:
        kernel_input_signature += f",\n{INDENTATION}"
        kernel_input_signature += f",\n{INDENTATION}".join(
            [f"res{i}_b_seq_ptr" for i in range(n)]
        )

    kernel_output_signature = _join_signature(
        [f"grad_x{i}_seq_ptr" for i in range(num_inputs)]
        + [f"grad_v{i}_init_ptr" for i in range(num_states)]
    )

    autotune_restore = _join_args(
        [f'"grad_x{i}_seq_ptr"' for i in range(num_inputs)]
        + [f'"grad_v{i}_init_ptr"' for i in range(num_states)]
    )

    init_state_loads = "".join(
        [grad_final_state_load_template.format(name=f"grad_v{i}") for i in range(num_states)]
    )

    loads = "".join(
        [load_template.format(name=f"grad_s{i}") for i in range(num_outputs)]
    )
    loads += "".join([load_template.format(name=f"res{i}_b") for i in range(n)])

    stores = "".join(
        [store_template.format(name=f"grad_x{i}") for i in range(num_inputs)]
    )

    lhs = _join_args(
        [f"grad_x{i}" for i in range(num_inputs)]
        + [f"grad_v{i}_accumulate" for i in range(num_states)]
    )
    _core_args_parts = []
    if n > 0:
        _core_args_parts += [f"res{i}_b" for i in range(n)]
    _core_args_parts += [f"grad_s{i}" for i in range(num_outputs)]
    _core_args_parts += [f"grad_v{i}_accumulate" for i in range(num_states)]
    core_args = ", ".join(_core_args_parts)
    computes = f"{lhs} = {core_name}({core_args})"

    tail = f"\n{INDENTATION}".join(
        [
            grad_init_state_store_template.format(name=f"grad_v{i}")
            for i in range(num_states)
        ]
    )

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_configs=BACKWARD_AUTOTUNE_CONFIGS,
        autotune_key='["T", "dtype", "NCL_BUCKET"]',
        meta_signature=",\n    NCL_BUCKET: tl.constexpr",
        autotune_restore=autotune_restore,
        kernel_type="backward_final_state",
        hash=hash,
        kernel_input_signature=kernel_input_signature,
        kernel_output_signature=kernel_output_signature,
        init_state_loads=init_state_loads,
        loop_range="T-1, -1, -1",
        loads=loads,
        computes=computes,
        stores=stores,
        tail=tail,
    ).strip()
    kernel_name = f"flexsn_backward_final_state_kernel_{hash}"

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn backward-final-state kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe
