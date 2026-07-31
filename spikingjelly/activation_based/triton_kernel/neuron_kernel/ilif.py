import torch

from ..triton_utils import (
    convert_and_store,
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except BaseException as e:
    import logging

    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.neuron_kernel.ilif: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()
    libdevice = dummy.DummyImport()


__all__ = ["single_step_ilif", "multistep_ilif"]


@triton.jit
def _single_step_ilif_forward_kernel(
    x_ptr,
    v_init_ptr,
    spike_ptr,
    h_ptr,
    v_ptr,
    decay,
    v_threshold,
    max_spike_count,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = offsets < NCL
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    v_init = tl.load(v_init_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    decay = tl.full([1], decay, dtype=compute_dtype)
    threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    max_count = tl.full([1], max_spike_count, dtype=compute_dtype)

    h = decay * v_init + x
    scaled_h = tl.maximum(tl.minimum(h / threshold, max_count), 0.0)
    spike = libdevice.rint(scaled_h.to(tl.float32)).to(compute_dtype)
    v = h - spike * threshold

    tl.store(spike_ptr + offsets, spike, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)
    if save_intermediates:
        tl.store(h_ptr + offsets, h, mask=mask)


@triton.jit
def _single_step_ilif_backward_kernel(
    grad_spike_ptr,
    grad_v_ptr,
    h_ptr,
    grad_x_ptr,
    grad_v_init_ptr,
    decay,
    v_threshold,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    grad_min: tl.constexpr,
    grad_max: tl.constexpr,
    detach_reset: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = offsets < NCL
    grad_spike = tl.load(grad_spike_ptr + offsets, mask=mask, other=0.0).to(
        compute_dtype
    )
    grad_v = tl.load(grad_v_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    decay = tl.full([1], decay, dtype=compute_dtype)
    threshold = tl.full([1], v_threshold, dtype=compute_dtype)

    scaled_h = h / threshold
    sg = tl.where(
        (scaled_h >= grad_min) & (scaled_h <= grad_max),
        1.0 / threshold,
        0.0,
    ).to(compute_dtype)
    if detach_reset:
        grad_h = tl.fma(grad_spike, sg, grad_v)
    else:
        grad_h = tl.fma(grad_spike - threshold * grad_v, sg, grad_v)

    tl.store(grad_x_ptr + offsets, grad_h, mask=mask)
    tl.store(grad_v_init_ptr + offsets, grad_h * decay, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "compute_dtype", "save_intermediates", "store_v_seq"],
    restore_value=["spike_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_ilif_forward_kernel_static(
    x_seq_ptr,
    v_init_ptr,
    spike_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    decay,
    v_threshold,
    max_spike_count,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    max_spike_count = tl.full([1], max_spike_count, dtype=compute_dtype)
    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero").to(
        compute_dtype
    )

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        h = decay * v + x
        scaled_h = tl.maximum(tl.minimum(h / v_threshold, max_spike_count), 0.0)
        spike = libdevice.rint(scaled_h.to(tl.float32)).to(compute_dtype)
        v = h - spike * v_threshold

        spike_ptrs = tl.make_block_ptr(
            spike_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(spike_ptrs, spike, boundary_check=(1,))
        if store_v_seq:
            v_ptrs = tl.make_block_ptr(
                v_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

    if not store_v_seq:
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_last_ptrs, v, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["NCL", "compute_dtype", "save_intermediates", "store_v_seq"],
    restore_value=["spike_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_ilif_forward_kernel_dynamic(
    x_seq_ptr,
    v_init_ptr,
    spike_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    decay,
    v_threshold,
    max_spike_count,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    save_intermediates: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    max_spike_count = tl.full([1], max_spike_count, dtype=compute_dtype)
    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero").to(
        compute_dtype
    )

    for t in tl.range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        h = decay * v + x
        scaled_h = tl.maximum(tl.minimum(h / v_threshold, max_spike_count), 0.0)
        spike = libdevice.rint(scaled_h.to(tl.float32)).to(compute_dtype)
        v = h - spike * v_threshold

        spike_ptrs = tl.make_block_ptr(
            spike_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(spike_ptrs, spike, boundary_check=(1,))
        if store_v_seq:
            v_ptrs = tl.make_block_ptr(
                v_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

    if not store_v_seq:
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_last_ptrs, v, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["T", "NCL", "compute_dtype", "detach_reset", "store_v_seq"],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr"],
)
@triton.jit
def _multistep_ilif_backward_kernel_static(
    grad_spike_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    decay,
    v_threshold,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    grad_min: tl.constexpr,
    grad_max: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    else:
        grad_v_last_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v_acc = tl.load(
            grad_v_last_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_spike_ptrs = tl.make_block_ptr(
            grad_spike_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_spike = tl.load(
            grad_spike_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = tl.make_block_ptr(
                grad_v_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            grad_v_acc += tl.load(
                grad_v_ptrs, boundary_check=(1,), padding_option="zero"
            ).to(compute_dtype)
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        scaled_h = h / v_threshold
        sg = tl.where(
            (scaled_h >= grad_min) & (scaled_h <= grad_max),
            1.0 / v_threshold,
            0.0,
        ).to(compute_dtype)
        if detach_reset:
            grad_h = tl.fma(grad_spike, sg, grad_v_acc)
        else:
            grad_h = tl.fma(
                grad_spike - v_threshold * grad_v_acc,
                sg,
                grad_v_acc,
            )
        grad_v_acc = grad_h * decay
        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(grad_x_ptrs, grad_h, boundary_check=(1,))

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v_init_ptrs, grad_v_acc, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=["NCL", "compute_dtype", "detach_reset", "store_v_seq"],
    restore_value=["grad_x_seq_ptr", "grad_v_init_ptr"],
)
@triton.jit
def _multistep_ilif_backward_kernel_dynamic(
    grad_spike_seq_ptr,
    grad_v_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    grad_v_init_ptr,
    decay,
    v_threshold,
    T,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    compute_dtype: tl.constexpr,
    grad_min: tl.constexpr,
    grad_max: tl.constexpr,
    detach_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros([1, BLOCK_NCL], dtype=compute_dtype)
    else:
        grad_v_last_ptrs = tl.make_block_ptr(
            grad_v_seq_ptr,
            shape=(1, NCL),
            strides=(NCL, 1),
            offsets=(0, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v_acc = tl.load(
            grad_v_last_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)

    for t in tl.range(T - 1, -1, -1):
        grad_spike_ptrs = tl.make_block_ptr(
            grad_spike_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_spike = tl.load(
            grad_spike_ptrs, boundary_check=(1,), padding_option="zero"
        ).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = tl.make_block_ptr(
                grad_v_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            grad_v_acc += tl.load(
                grad_v_ptrs, boundary_check=(1,), padding_option="zero"
            ).to(compute_dtype)
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero").to(
            compute_dtype
        )
        scaled_h = h / v_threshold
        sg = tl.where(
            (scaled_h >= grad_min) & (scaled_h <= grad_max),
            1.0 / v_threshold,
            0.0,
        ).to(compute_dtype)
        if detach_reset:
            grad_h = tl.fma(grad_spike, sg, grad_v_acc)
        else:
            grad_h = tl.fma(
                grad_spike - v_threshold * grad_v_acc,
                sg,
                grad_v_acc,
            )
        grad_v_acc = grad_h * decay
        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(grad_x_ptrs, grad_h, boundary_check=(1,))

    grad_v_init_ptrs = tl.make_block_ptr(
        grad_v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v_init_ptrs, grad_v_acc, boundary_check=(1,))


def _select_forward_kernel(T: int):
    if use_static_range_for_triton_neuron_kernel(T):
        return _multistep_ilif_forward_kernel_static
    return _multistep_ilif_forward_kernel_dynamic


def _select_backward_kernel(T: int):
    if use_static_range_for_triton_neuron_kernel(T):
        return _multistep_ilif_backward_kernel_static
    return _multistep_ilif_backward_kernel_dynamic


def _launch_forward_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    spike_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_out: torch.Tensor,
    *,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    save_intermediates: bool,
    store_v_seq: bool,
) -> None:
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()

    def grid(meta):
        return (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    kernel = wrap_triton(_select_forward_kernel(T))
    with torch.cuda.device(x_seq.device):
        kernel[grid](
            x_seq,
            v_init,
            spike_seq,
            h_seq,
            v_out,
            decay,
            v_threshold,
            max_spike_count,
            T=T,
            NCL=NCL,
            compute_dtype=type_dict[x_seq.dtype],
            save_intermediates=save_intermediates,
            store_v_seq=store_v_seq,
        )


def _launch_backward_kernel(
    grad_spike_seq: torch.Tensor,
    grad_v_out: torch.Tensor,
    h_seq: torch.Tensor,
    grad_x_seq: torch.Tensor,
    grad_v_init: torch.Tensor,
    *,
    decay: float,
    v_threshold: float,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> None:
    T = grad_spike_seq.shape[0]
    NCL = grad_spike_seq[0].numel()

    def grid(meta):
        return (triton.cdiv(NCL, meta["BLOCK_NCL"]),)

    kernel = wrap_triton(_select_backward_kernel(T))
    with torch.cuda.device(grad_spike_seq.device):
        kernel[grid](
            grad_spike_seq,
            grad_v_out,
            h_seq,
            grad_x_seq,
            grad_v_init,
            decay,
            v_threshold,
            T=T,
            NCL=NCL,
            compute_dtype=type_dict[grad_spike_seq.dtype],
            grad_min=grad_min,
            grad_max=grad_max,
            detach_reset=detach_reset,
            store_v_seq=store_v_seq,
        )


@register_op("sj::multistep_ilif_forward_no_grad")
def multistep_ilif_forward_no_grad(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()
    spike_seq = torch.empty_like(x_seq)
    v_out = torch.empty_like(x_seq) if store_v_seq else torch.empty_like(v_init)
    _launch_forward_kernel(
        x_seq,
        v_init,
        spike_seq,
        v_out,
        v_out,
        decay=decay,
        v_threshold=v_threshold,
        max_spike_count=max_spike_count,
        save_intermediates=False,
        store_v_seq=store_v_seq,
    )
    return spike_seq, v_out


@torch.library.register_fake("sj::multistep_ilif_forward_no_grad")
def _multistep_ilif_forward_no_grad_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    store_v_seq: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape if store_v_seq else v_init.shape),
    )


@register_op("sj::multistep_ilif_forward")
def multistep_ilif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()
    spike_seq = torch.empty_like(x_seq)
    v_out = torch.empty_like(x_seq) if store_v_seq else torch.empty_like(v_init)
    h_seq = torch.empty_like(x_seq)
    _launch_forward_kernel(
        x_seq,
        v_init,
        spike_seq,
        h_seq,
        v_out,
        decay=decay,
        v_threshold=v_threshold,
        max_spike_count=max_spike_count,
        save_intermediates=True,
        store_v_seq=store_v_seq,
    )
    return spike_seq, v_out, h_seq


@torch.library.register_fake("sj::multistep_ilif_forward")
def _multistep_ilif_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
    store_v_seq: bool,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape if store_v_seq else v_init.shape),
        x_seq.new_empty(x_seq.shape),
    )


def _launch_single_step_ilif_forward(
    x: torch.Tensor,
    v_init: torch.Tensor,
    spike: torch.Tensor,
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    save_intermediates: bool,
) -> None:
    NCL = x.numel()
    grid = (triton.cdiv(NCL, 256),)
    with torch.cuda.device(x.device):
        wrap_triton(_single_step_ilif_forward_kernel)[grid](
            x,
            v_init,
            spike,
            h,
            v,
            decay,
            v_threshold,
            max_spike_count,
            NCL=NCL,
            BLOCK_NCL=256,
            compute_dtype=type_dict[x.dtype],
            save_intermediates=save_intermediates,
        )


def _launch_single_step_ilif_backward(
    grad_spike: torch.Tensor,
    grad_v: torch.Tensor,
    h: torch.Tensor,
    grad_x: torch.Tensor,
    grad_v_init: torch.Tensor,
    *,
    decay: float,
    v_threshold: float,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
) -> None:
    NCL = grad_spike.numel()
    grid = (triton.cdiv(NCL, 256),)
    with torch.cuda.device(grad_spike.device):
        wrap_triton(_single_step_ilif_backward_kernel)[grid](
            grad_spike,
            grad_v,
            h,
            grad_x,
            grad_v_init,
            decay,
            v_threshold,
            NCL=NCL,
            BLOCK_NCL=256,
            compute_dtype=type_dict[grad_spike.dtype],
            grad_min=grad_min,
            grad_max=grad_max,
            detach_reset=detach_reset,
        )


@register_op("sj::single_step_ilif_inference")
def single_step_ilif_inference(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    v_init = v_init.contiguous()
    spike = torch.empty_like(x)
    v = torch.empty_like(v_init)
    _launch_single_step_ilif_forward(
        x,
        v_init,
        spike,
        v,
        v,
        decay=decay,
        v_threshold=v_threshold,
        max_spike_count=max_spike_count,
        save_intermediates=False,
    )
    return spike, v


@torch.library.register_fake("sj::single_step_ilif_inference")
def _single_step_ilif_inference_fake(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
):
    return x.new_empty(x.shape), v_init.new_empty(v_init.shape)


@register_op("sj::single_step_ilif_forward")
def single_step_ilif_forward(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    v_init = v_init.contiguous()
    spike = torch.empty_like(x)
    v = torch.empty_like(v_init)
    h = torch.empty_like(x)
    _launch_single_step_ilif_forward(
        x,
        v_init,
        spike,
        h,
        v,
        decay=decay,
        v_threshold=v_threshold,
        max_spike_count=max_spike_count,
        save_intermediates=True,
    )
    return spike, v, h


@torch.library.register_fake("sj::single_step_ilif_forward")
def _single_step_ilif_forward_fake(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
):
    return (
        x.new_empty(x.shape),
        v_init.new_empty(v_init.shape),
        x.new_empty(x.shape),
    )


def _setup_single_step_context(ctx, inputs, output):
    (
        ctx.decay,
        ctx.v_threshold,
        _,
        ctx.grad_min,
        ctx.grad_max,
        ctx.detach_reset,
    ) = inputs[2:]
    ctx.save_for_backward(output[2])


def _single_step_ilif_backward(ctx, grad_spike, grad_v, grad_h):
    (h,) = ctx.saved_tensors
    grad_spike = grad_spike.contiguous()
    grad_v = grad_v.contiguous()
    grad_x = torch.empty_like(grad_spike)
    grad_v_init = torch.empty_like(grad_v)
    _launch_single_step_ilif_backward(
        grad_spike,
        grad_v,
        h.contiguous(),
        grad_x,
        grad_v_init,
        decay=ctx.decay,
        v_threshold=ctx.v_threshold,
        grad_min=ctx.grad_min,
        grad_max=ctx.grad_max,
        detach_reset=ctx.detach_reset,
    )
    return grad_x, grad_v_init, None, None, None, None, None, None


torch.library.register_autograd(
    "sj::single_step_ilif_forward",
    _single_step_ilif_backward,
    setup_context=_setup_single_step_context,
)


def single_step_ilif(
    x: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <single_step_ilif-cn>` | :ref:`English <single_step_ilif-en>`

    ----

    .. _single_step_ilif-cn:

    * **中文**

    使用专用 Triton kernel 执行单步 I-LIF 状态转移。输出是
    ``[0, max_spike_count]`` 内的整数计数，不是二值脉冲。``x`` 和
    ``v_init`` 必须是 shape、dtype 和 device 相同的 CUDA FP32、FP16 或
    BF16 张量。

    :param x: 单步 CUDA 输入，形状为 ``[N, *]``
    :type x: torch.Tensor
    :param v_init: 与 ``x`` 同形状的初始膜电位
    :type v_init: torch.Tensor
    :param decay: 膜电位衰减系数
    :type decay: float
    :param v_threshold: 发放阈值
    :type v_threshold: float
    :param max_spike_count: 每个元素的最大整数输出
    :type max_spike_count: int
    :param grad_min: 矩形替代梯度窗口下界
    :type grad_min: float
    :param grad_max: 矩形替代梯度窗口上界
    :type grad_max: float
    :param detach_reset: 是否在反向传播中分离复位项
    :type detach_reset: bool
    :return: ``(integer_count, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: 当 ``x`` 和 ``v_init`` 的 shape、dtype 或 CUDA device
        不一致，或输入不在 CUDA 上时
    :raises NotImplementedError: 当 dtype 不受 Triton 后端支持时

    ----

    .. _single_step_ilif-en:

    * **English**

    Run one I-LIF state transition with the dedicated Triton kernel. The output
    is an integer count in ``[0, max_spike_count]``, not a binary spike. ``x``
    and ``v_init`` must be CUDA FP32, FP16, or BF16 tensors with matching shape,
    dtype, and device.

    :param x: Single-step CUDA input shaped ``[N, *]``
    :type x: torch.Tensor
    :param v_init: Initial membrane voltage with the same shape as ``x``
    :type v_init: torch.Tensor
    :param decay: Membrane-voltage decay factor
    :type decay: float
    :param v_threshold: Firing threshold
    :type v_threshold: float
    :param max_spike_count: Maximum integer output per element
    :type max_spike_count: int
    :param grad_min: Lower bound of the rectangular surrogate-gradient window
    :type grad_min: float
    :param grad_max: Upper bound of the rectangular surrogate-gradient window
    :type grad_max: float
    :param detach_reset: Whether to detach the reset term in backward
    :type detach_reset: bool
    :return: ``(integer_count, v_next)``
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If ``x`` and ``v_init`` differ in shape, dtype, or CUDA
        device, or if the input is not on CUDA
    :raises NotImplementedError: If the dtype is not supported by the Triton
        backend
    """
    if x.device.type != "cuda":
        raise ValueError("single_step_ilif requires CUDA tensors.")
    if v_init.shape != x.shape or v_init.dtype != x.dtype or v_init.device != x.device:
        raise ValueError("x and v_init must have the same shape, dtype, and device.")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(x.dtype)
    need_grad = torch.is_grad_enabled() and (x.requires_grad or v_init.requires_grad)
    if need_grad:
        spike, v, _ = single_step_ilif_forward(
            x,
            v_init,
            decay,
            v_threshold,
            max_spike_count,
            grad_min,
            grad_max,
            detach_reset,
        )
        return spike, v
    return single_step_ilif_inference(
        x,
        v_init,
        decay,
        v_threshold,
        max_spike_count,
    )


def _setup_context(ctx, inputs, output):
    (
        ctx.decay,
        ctx.v_threshold,
        _,
        ctx.grad_min,
        ctx.grad_max,
        ctx.detach_reset,
        ctx.store_v_seq,
    ) = inputs[2:]
    ctx.save_for_backward(output[2])


def _multistep_ilif_backward(ctx, grad_spike_seq, grad_v_out, grad_h_seq):
    (h_seq,) = ctx.saved_tensors
    grad_spike_seq = grad_spike_seq.contiguous()
    grad_v_out = grad_v_out.contiguous()
    grad_x_seq = torch.empty_like(grad_spike_seq)
    grad_v_init = torch.empty_like(h_seq[0])
    _launch_backward_kernel(
        grad_spike_seq,
        grad_v_out,
        h_seq.contiguous(),
        grad_x_seq,
        grad_v_init,
        decay=ctx.decay,
        v_threshold=ctx.v_threshold,
        grad_min=ctx.grad_min,
        grad_max=ctx.grad_max,
        detach_reset=ctx.detach_reset,
        store_v_seq=ctx.store_v_seq,
    )
    return grad_x_seq, grad_v_init, None, None, None, None, None, None, None


torch.library.register_autograd(
    "sj::multistep_ilif_forward",
    _multistep_ilif_backward,
    setup_context=_setup_context,
)


def multistep_ilif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: float,
    v_threshold: float,
    max_spike_count: int,
    grad_min: float,
    grad_max: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v_init.requires_grad
    )
    if need_grad:
        spike_seq, v_out, _ = multistep_ilif_forward(
            x_seq,
            v_init,
            decay,
            v_threshold,
            max_spike_count,
            grad_min,
            grad_max,
            detach_reset,
            store_v_seq,
        )
    else:
        spike_seq, v_out = multistep_ilif_forward_no_grad(
            x_seq,
            v_init,
            decay,
            v_threshold,
            max_spike_count,
            store_v_seq,
        )
    return spike_seq, v_out
