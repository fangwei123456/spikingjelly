from spikingjelly.logger import logger
import torch

from ..._neuron_layout import _empty_like
from .utils import (
    _spatial_offsets,
    _time_offset,
    _triton_layout_args,
    _neuron_grid,
    _neuron_indices,
    _block_minor,
)

from ..triton_utils import (
    do_bench_cudagraph,
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except (ImportError, OSError) as e:
    from .. import dummy

    logger.debug("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()
    libdevice = dummy.DummyImport()


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "T",
        "NCL",
        "compute_dtype",
        "save_intermediates",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    spike_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    max_spike_count = tl.full([1], max_spike_count, dtype=compute_dtype)
    v_init_ptrs = v_init_ptr + v_init_ptr_offsets
    v = tl.load(v_init_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.static_range(0, T, 1):
        x_ptrs = x_seq_ptr + x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h = decay * v + x
        scaled_h = tl.maximum(tl.minimum(h / v_threshold, max_spike_count), 0.0)
        spike = libdevice.rint(scaled_h.to(tl.float32)).to(compute_dtype)
        v = h - spike * v_threshold

        spike_ptrs = (
            spike_seq_ptr + spike_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        )
        tl.store(spike_ptrs, spike, mask=mask)
        if store_v_seq:
            v_ptrs = v_seq_ptr + v_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 4)
            tl.store(v_ptrs, v, mask=mask)
        if save_intermediates:
            h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
            tl.store(h_ptrs, h, mask=mask)

    if not store_v_seq:
        v_last_ptrs = v_seq_ptr + v_seq_ptr_offsets
        tl.store(v_last_ptrs, v, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "NCL",
        "compute_dtype",
        "save_intermediates",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    spike_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    max_spike_count = tl.full([1], max_spike_count, dtype=compute_dtype)
    v_init_ptrs = v_init_ptr + v_init_ptr_offsets
    v = tl.load(v_init_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.range(0, T, 1):
        x_ptrs = x_seq_ptr + x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 0)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h = decay * v + x
        scaled_h = tl.maximum(tl.minimum(h / v_threshold, max_spike_count), 0.0)
        spike = libdevice.rint(scaled_h.to(tl.float32)).to(compute_dtype)
        v = h - spike * v_threshold

        spike_ptrs = (
            spike_seq_ptr + spike_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        )
        tl.store(spike_ptrs, spike, mask=mask)
        if store_v_seq:
            v_ptrs = v_seq_ptr + v_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 4)
            tl.store(v_ptrs, v, mask=mask)
        if save_intermediates:
            h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
            tl.store(h_ptrs, h, mask=mask)

    if not store_v_seq:
        v_last_ptrs = v_seq_ptr + v_seq_ptr_offsets
        tl.store(v_last_ptrs, v, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "T",
        "NCL",
        "compute_dtype",
        "detach_reset",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    grad_spike_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    grad_v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    grad_x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    grad_v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros(indices.shape, dtype=compute_dtype)
    else:
        grad_v_last_ptrs = grad_v_seq_ptr + grad_v_seq_ptr_offsets
        grad_v_acc = tl.load(grad_v_last_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_spike_ptrs = (
            grad_spike_seq_ptr
            + grad_spike_seq_ptr_offsets
            + _time_offset(t, NCL, STRIDES, 0)
        )
        grad_spike = tl.load(grad_spike_ptrs, mask=mask, other=0.0).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = (
                grad_v_seq_ptr
                + grad_v_seq_ptr_offsets
                + _time_offset(t, NCL, STRIDES, 1)
            )
            grad_v_acc += tl.load(grad_v_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        h = tl.load(h_ptrs, mask=mask, other=0.0).to(compute_dtype)
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
        grad_x_ptrs = (
            grad_x_seq_ptr + grad_x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
        )
        tl.store(grad_x_ptrs, grad_h, mask=mask)

    grad_v_init_ptrs = grad_v_init_ptr + grad_v_init_ptr_offsets
    tl.store(grad_v_init_ptrs, grad_v_acc, mask=mask)


@triton.autotune(
    do_bench=do_bench_cudagraph,
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [4, 8]
    ],
    key=[
        "BLOCK_MINOR",
        "NCL",
        "compute_dtype",
        "detach_reset",
        "store_v_seq",
        "SIZES",
        "STRIDES",
    ],
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
    BLOCK_MINOR: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
):
    indices, mask = _neuron_indices(NCL, BLOCK_NCL, SIZES, BLOCK_MINOR)
    grad_spike_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 0)
    grad_v_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 1)
    h_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 2)
    grad_x_seq_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 3)
    grad_v_init_ptr_offsets = _spatial_offsets(indices, SIZES, STRIDES, 4)
    decay = tl.full([1], decay, dtype=compute_dtype)
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    if store_v_seq:
        grad_v_acc = tl.zeros(indices.shape, dtype=compute_dtype)
    else:
        grad_v_last_ptrs = grad_v_seq_ptr + grad_v_seq_ptr_offsets
        grad_v_acc = tl.load(grad_v_last_ptrs, mask=mask, other=0.0).to(compute_dtype)

    for t in tl.range(T - 1, -1, -1):
        grad_spike_ptrs = (
            grad_spike_seq_ptr
            + grad_spike_seq_ptr_offsets
            + _time_offset(t, NCL, STRIDES, 0)
        )
        grad_spike = tl.load(grad_spike_ptrs, mask=mask, other=0.0).to(compute_dtype)
        if store_v_seq:
            grad_v_ptrs = (
                grad_v_seq_ptr
                + grad_v_seq_ptr_offsets
                + _time_offset(t, NCL, STRIDES, 1)
            )
            grad_v_acc += tl.load(grad_v_ptrs, mask=mask, other=0.0).to(compute_dtype)
        h_ptrs = h_seq_ptr + h_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 2)
        h = tl.load(h_ptrs, mask=mask, other=0.0).to(compute_dtype)
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
        grad_x_ptrs = (
            grad_x_seq_ptr + grad_x_seq_ptr_offsets + _time_offset(t, NCL, STRIDES, 3)
        )
        tl.store(grad_x_ptrs, grad_h, mask=mask)

    grad_v_init_ptrs = grad_v_init_ptr + grad_v_init_ptr_offsets
    tl.store(grad_v_init_ptrs, grad_v_acc, mask=mask)


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
    sizes, strides = _triton_layout_args(x_seq, x_seq, v_init, spike_seq, h_seq, v_out)
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()

    def grid(meta):
        return _neuron_grid(NCL, sizes, meta["BLOCK_NCL"], meta["BLOCK_MINOR"])

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
            BLOCK_MINOR=_block_minor(sizes, strides),
            SIZES=sizes,
            STRIDES=strides,
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
    sizes, strides = _triton_layout_args(
        h_seq, grad_spike_seq, grad_v_out, h_seq, grad_x_seq, grad_v_init
    )
    T = grad_spike_seq.shape[0]
    NCL = grad_spike_seq[0].numel()

    def grid(meta):
        return _neuron_grid(NCL, sizes, meta["BLOCK_NCL"], meta["BLOCK_MINOR"])

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
            BLOCK_MINOR=_block_minor(sizes, strides),
            SIZES=sizes,
            STRIDES=strides,
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
    spike_seq = _empty_like(x_seq)
    v_out = _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False)
    _launch_forward_kernel(
        x_seq,
        v_init,
        spike_seq,
        v_out,  # h_seq is not written when save_intermediates=False.
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
        _empty_like(x_seq),
        _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False),
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
    spike_seq = _empty_like(x_seq)
    v_out = _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False)
    h_seq = _empty_like(x_seq)
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
        _empty_like(x_seq),
        _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False),
        _empty_like(x_seq),
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
    grad_x_seq = _empty_like(h_seq)
    grad_v_init = _empty_like(h_seq[0], sequence=False)
    _launch_backward_kernel(
        grad_spike_seq,
        grad_v_out,
        h_seq,
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


def _multistep_ilif(
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
