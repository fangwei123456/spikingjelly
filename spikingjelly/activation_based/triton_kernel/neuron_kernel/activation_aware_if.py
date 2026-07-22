import torch

from ..triton_utils import (
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
)

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging

    from .. import dummy

    logging.info(
        "spikingjelly.activation_based.triton_kernel.neuron_kernel."
        f"activation_aware_if: {e}"
    )
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
        "save_v_seq",
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
    v_final_ptr,
    v_seq_ptr,
    v_reset,
    T: tl.constexpr,
    N: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_v_seq: tl.constexpr,
    threshold_is_scalar: tl.constexpr,
    offset_is_scalar: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    channel_offsets = offsets // INNER_SIZE % CHANNEL_SIZE
    v = tl.load(v_init_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    if threshold_is_scalar:
        threshold = tl.load(threshold_ptr).to(compute_dtype)
    else:
        threshold = tl.load(threshold_ptr + channel_offsets, mask=mask, other=1.0).to(
            compute_dtype
        )
    if offset_is_scalar:
        offset = tl.load(offset_ptr).to(compute_dtype)
    else:
        offset = tl.load(offset_ptr + channel_offsets, mask=mask, other=0.0).to(
            compute_dtype
        )
    reset = tl.full([1], v_reset, compute_dtype)

    for t in tl.static_range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(compute_dtype)
        h = v + x
        spike = tl.where(h + offset >= threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(spike_seq_ptr + t * N + offsets, spike, mask=mask)
        if save_v_seq:
            tl.store(v_seq_ptr + t * N + offsets, v, mask=mask)
    tl.store(v_final_ptr + offsets, v, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps)
        for block_n, num_warps in ((128, 4), (256, 8))
    ],
    key=[
        "N",
        "compute_dtype",
        "soft_reset",
        "save_v_seq",
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
    v_final_ptr,
    v_seq_ptr,
    v_reset,
    T,
    N: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_v_seq: tl.constexpr,
    threshold_is_scalar: tl.constexpr,
    offset_is_scalar: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    channel_offsets = offsets // INNER_SIZE % CHANNEL_SIZE
    v = tl.load(v_init_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
    if threshold_is_scalar:
        threshold = tl.load(threshold_ptr).to(compute_dtype)
    else:
        threshold = tl.load(threshold_ptr + channel_offsets, mask=mask, other=1.0).to(
            compute_dtype
        )
    if offset_is_scalar:
        offset = tl.load(offset_ptr).to(compute_dtype)
    else:
        offset = tl.load(offset_ptr + channel_offsets, mask=mask, other=0.0).to(
            compute_dtype
        )
    reset = tl.full([1], v_reset, compute_dtype)

    for t in tl.range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(compute_dtype)
        h = v + x
        spike = tl.where(h + offset >= threshold, 1.0, 0.0).to(compute_dtype)
        if soft_reset:
            v = h - spike * threshold
        else:
            v = spike * reset + (1.0 - spike) * h
        tl.store(spike_seq_ptr + t * N + offsets, spike, mask=mask)
        if save_v_seq:
            tl.store(v_seq_ptr + t * N + offsets, v, mask=mask)
    tl.store(v_final_ptr + offsets, v, mask=mask)


def _launch_activation_aware_if_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    spike_seq: torch.Tensor,
    v_final: torch.Tensor,
    v_seq: torch.Tensor,
    *,
    channel_size: int,
    inner_size: int,
    v_reset: float,
    soft_reset: bool,
    save_v_seq: bool,
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

    with torch.cuda.device(x_seq.device):
        kernel[grid](
            x_seq,
            v_init,
            threshold,
            offset,
            spike_seq,
            v_final,
            v_seq,
            v_reset,
            T=T,
            N=N,
            CHANNEL_SIZE=channel_size,
            INNER_SIZE=inner_size,
            compute_dtype=type_dict[x_seq.dtype],
            soft_reset=soft_reset,
            save_v_seq=save_v_seq,
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
    save_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()
    threshold = threshold.to(device=x_seq.device, dtype=x_seq.dtype).contiguous()
    offset = offset.to(device=x_seq.device, dtype=x_seq.dtype).contiguous()
    spike_seq = torch.empty_like(x_seq)
    v_final = torch.empty_like(v_init)
    v_seq = torch.empty_like(x_seq) if save_v_seq else x_seq.new_empty((0,))
    _launch_activation_aware_if_forward(
        x_seq,
        v_init,
        threshold,
        offset,
        spike_seq,
        v_final,
        v_seq,
        channel_size=channel_size,
        inner_size=inner_size,
        v_reset=v_reset,
        soft_reset=soft_reset,
        save_v_seq=save_v_seq,
    )
    return spike_seq, v_final, v_seq


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
    save_v_seq: bool,
):
    del threshold, offset, channel_size, inner_size, v_reset, soft_reset
    v_seq_shape = x_seq.shape if save_v_seq else (0,)
    return (
        torch.empty_like(x_seq),
        torch.empty_like(v_init),
        x_seq.new_empty(v_seq_shape),
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
    save_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    soft_reset = v_reset is None
    reset = 0.0 if soft_reset else v_reset
    spike_seq, v_final, v_seq = _multistep_activation_aware_if_inference(
        x_seq,
        v_init,
        threshold,
        offset,
        channel_size,
        inner_size,
        reset,
        soft_reset,
        save_v_seq,
    )
    return spike_seq, v_final, (v_seq if save_v_seq else None)
