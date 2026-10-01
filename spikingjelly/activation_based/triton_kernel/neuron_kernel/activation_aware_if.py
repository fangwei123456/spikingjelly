import math
from spikingjelly.logger import logger
import torch

from ..._neuron_layout import _empty_like, _layout_args
from .utils import _spatial_offsets

from ..triton_utils import (
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

try:
    import triton
    import triton.language as tl
except (ImportError, OSError) as e:
    from .. import dummy

    logger.debug("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()


__all__ = []


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps)
        for block_n, num_warps in ((128, 4), (256, 8))
    ],
    key=[
        "T",
        "N",
        "compute_dtype",
        "soft_reset",
        "store_v_seq",
        "threshold_is_scalar",
        "offset_is_scalar",
    ],
)
@triton.jit
def _multistep_activation_aware_if_forward_static(
    x_seq_ptr,
    v_init_ptr,
    threshold_ptr,
    offset_ptr,
    spike_seq_ptr,
    v_out_ptr,
    v_reset,
    T: tl.constexpr,
    N: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
    threshold_is_scalar: tl.constexpr,
    offset_is_scalar: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
    LOGICAL: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 1)
    spike_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 2)
    v_out_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 3)
    channel_offsets = (
        _spatial_offsets(offsets, SIZES, LOGICAL) // INNER_SIZE % CHANNEL_SIZE
    )
    v = tl.load(v_init_ptr + v_init_ptr_offsets, mask=mask, other=0.0).to(compute_dtype)
    if threshold_is_scalar:
        threshold = tl.load(threshold_ptr).to(compute_dtype)
    else:
        threshold = tl.load(
            threshold_ptr + channel_offsets * THRESHOLD_STRIDE, mask=mask, other=1.0
        ).to(compute_dtype)
    if offset_is_scalar:
        offset = tl.load(offset_ptr).to(compute_dtype)
    else:
        offset = tl.load(
            offset_ptr + channel_offsets * OFFSET_STRIDE, mask=mask, other=0.0
        ).to(compute_dtype)
    reset = tl.full([1], v_reset, compute_dtype)

    for t in tl.static_range(0, T, 1):
        x = tl.load(
            x_seq_ptr + x_seq_ptr_offsets + t * tl.full((), STRIDES[0][0], tl.int64),
            mask=mask,
            other=0.0,
        ).to(compute_dtype)
        h = v + x
        spike = tl.where(h + offset >= threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(
            spike_seq_ptr
            + spike_seq_ptr_offsets
            + t * tl.full((), STRIDES[2][0], tl.int64),
            spike,
            mask=mask,
        )
        if store_v_seq:
            tl.store(
                v_out_ptr
                + v_out_ptr_offsets
                + t * tl.full((), STRIDES[3][0], tl.int64),
                v,
                mask=mask,
            )
    if not store_v_seq:
        tl.store(v_out_ptr + v_out_ptr_offsets, v, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps)
        for block_n, num_warps in ((128, 4), (256, 8))
    ],
    key=[
        "N",
        "compute_dtype",
        "soft_reset",
        "store_v_seq",
        "threshold_is_scalar",
        "offset_is_scalar",
    ],
)
@triton.jit
def _multistep_activation_aware_if_forward_dynamic(
    x_seq_ptr,
    v_init_ptr,
    threshold_ptr,
    offset_ptr,
    spike_seq_ptr,
    v_out_ptr,
    v_reset,
    T,
    N: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    store_v_seq: tl.constexpr,
    threshold_is_scalar: tl.constexpr,
    offset_is_scalar: tl.constexpr,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
    LOGICAL: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 0)
    v_init_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 1)
    spike_seq_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 2)
    v_out_ptr_offsets = _spatial_offsets(offsets, SIZES, STRIDES, 3)
    channel_offsets = (
        _spatial_offsets(offsets, SIZES, LOGICAL) // INNER_SIZE % CHANNEL_SIZE
    )
    v = tl.load(v_init_ptr + v_init_ptr_offsets, mask=mask, other=0.0).to(compute_dtype)
    if threshold_is_scalar:
        threshold = tl.load(threshold_ptr).to(compute_dtype)
    else:
        threshold = tl.load(
            threshold_ptr + channel_offsets * THRESHOLD_STRIDE, mask=mask, other=1.0
        ).to(compute_dtype)
    if offset_is_scalar:
        offset = tl.load(offset_ptr).to(compute_dtype)
    else:
        offset = tl.load(
            offset_ptr + channel_offsets * OFFSET_STRIDE, mask=mask, other=0.0
        ).to(compute_dtype)
    reset = tl.full([1], v_reset, compute_dtype)

    for t in tl.range(0, T, 1):
        x = tl.load(
            x_seq_ptr + x_seq_ptr_offsets + t * tl.full((), STRIDES[0][0], tl.int64),
            mask=mask,
            other=0.0,
        ).to(compute_dtype)
        h = v + x
        spike = tl.where(h + offset >= threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(
            spike_seq_ptr
            + spike_seq_ptr_offsets
            + t * tl.full((), STRIDES[2][0], tl.int64),
            spike,
            mask=mask,
        )
        if store_v_seq:
            tl.store(
                v_out_ptr
                + v_out_ptr_offsets
                + t * tl.full((), STRIDES[3][0], tl.int64),
                v,
                mask=mask,
            )
    if not store_v_seq:
        tl.store(v_out_ptr + v_out_ptr_offsets, v, mask=mask)


def _launch_activation_aware_if_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    spike_seq: torch.Tensor,
    v_out: torch.Tensor,
    *,
    channel_size: int,
    inner_size: int,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
) -> None:
    T = x_seq.shape[0]
    N = x_seq[0].numel()
    kernel = (
        _multistep_activation_aware_if_forward_static
        if use_static_range_for_triton_neuron_kernel(T)
        else _multistep_activation_aware_if_forward_dynamic
    )

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    sizes, strides = _layout_args(x_seq, x_seq, v_init, spike_seq, v_out, sequence=True)
    order = sorted(range(1, x_seq.ndim), key=lambda d: (x_seq.stride(d), d))
    logical = (0,) + tuple(int(math.prod(x_seq.shape[d + 1 :])) for d in order)
    with torch.cuda.device(x_seq.device):
        wrap_triton(kernel)[grid](
            x_seq,
            v_init,
            threshold,
            offset,
            spike_seq,
            v_out,
            v_reset,
            T=T,
            N=N,
            SIZES=sizes,
            STRIDES=strides,
            LOGICAL=(logical,),
            THRESHOLD_STRIDE=threshold.stride(0) if threshold.ndim else 0,
            OFFSET_STRIDE=offset.stride(0) if offset.ndim else 0,
            CHANNEL_SIZE=channel_size,
            INNER_SIZE=inner_size,
            compute_dtype=type_dict[x_seq.dtype],
            soft_reset=soft_reset,
            store_v_seq=store_v_seq,
            threshold_is_scalar=threshold.dim() == 0,
            offset_is_scalar=offset.dim() == 0,
        )


@register_op("sj::multistep_activation_aware_if_inference")
def _multistep_activation_aware_if_inference(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channel_size: int,
    inner_size: int,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = threshold.to(device=x_seq.device, dtype=x_seq.dtype)
    offset = offset.to(device=x_seq.device, dtype=x_seq.dtype)
    spike_seq = _empty_like(x_seq)
    v_out = _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False)
    _launch_activation_aware_if_forward(
        x_seq,
        v_init,
        threshold,
        offset,
        spike_seq,
        v_out,
        channel_size=channel_size,
        inner_size=inner_size,
        v_reset=v_reset,
        soft_reset=soft_reset,
        store_v_seq=store_v_seq,
    )
    return spike_seq, v_out


@torch.library.register_fake("sj::multistep_activation_aware_if_inference")
def _multistep_activation_aware_if_inference_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channel_size: int,
    inner_size: int,
    v_reset: float,
    soft_reset: bool,
    store_v_seq: bool,
):
    del threshold, offset, channel_size, inner_size, v_reset, soft_reset
    return (
        _empty_like(x_seq),
        _empty_like(x_seq) if store_v_seq else _empty_like(v_init, sequence=False),
    )


def _multistep_activation_aware_if(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    *,
    channel_size: int,
    inner_size: int,
    v_reset: float | None,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    soft_reset = v_reset is None
    reset = 0.0 if soft_reset else v_reset
    return _multistep_activation_aware_if_inference(
        x_seq,
        v_init,
        threshold,
        offset,
        channel_size,
        inner_size,
        reset,
        soft_reset,
        store_v_seq,
    )
