from __future__ import annotations

import torch

from .triton_utils import (
    register_op,
    type_dict,
    use_static_range_for_triton_neuron_kernel,
    wrap_triton,
)

try:
    import triton
    import triton.language as tl
except (ImportError, OSError) as e:
    import logging

    from . import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.spikezip_kernel: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()

__all__ = ["multi_step_stbif"]

_STBIF_STATIC_RANGE_MAX_T = 16


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["T", "N", "dtype"],
    restore_value=["out_seq_ptr", "q_final_ptr", "acc_q_final_ptr", "cur_output_ptr"],
)
@triton.jit
def _multi_step_stbif_kernel_static(
    x_seq_ptr,
    q_init_ptr,
    acc_q_init_ptr,
    out_seq_ptr,
    q_final_ptr,
    acc_q_final_ptr,
    cur_output_ptr,
    work_flags_ptr,
    q_threshold,
    pos_max,
    neg_min,
    T: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    q = tl.load(q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    work = tl.full([BLOCK_N], False, dtype=tl.int1)

    for t in tl.static_range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = tl.floor(acc_q + 0.5)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        work = work | (normalized != 0.0) | (cur != 0.0)
        tl.store(
            out_seq_ptr + t * N + offsets, (cur * q_threshold).to(dtype), mask=mask
        )

    tl.store(q_final_ptr + offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + offsets, cur, mask=mask)
    tl.store(work_flags_ptr + pid_n, tl.max(work.to(tl.int32), axis=0))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [4, 8]
    ],
    key=["N", "dtype"],
    restore_value=["out_seq_ptr", "q_final_ptr", "acc_q_final_ptr", "cur_output_ptr"],
)
@triton.jit
def _multi_step_stbif_kernel_dynamic(
    x_seq_ptr,
    q_init_ptr,
    acc_q_init_ptr,
    out_seq_ptr,
    q_final_ptr,
    acc_q_final_ptr,
    cur_output_ptr,
    work_flags_ptr,
    q_threshold,
    pos_max,
    neg_min,
    T,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    q = tl.load(q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc_q = tl.load(acc_q_init_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cur = tl.zeros([BLOCK_N], dtype=tl.float32)
    work = tl.full([BLOCK_N], False, dtype=tl.int1)

    for t in tl.range(0, T, 1):
        x = tl.load(x_seq_ptr + t * N + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = x / q_threshold
        q = q + normalized
        acc_q = tl.floor(acc_q + 0.5)
        pos = (q >= 1.0) & (acc_q < pos_max)
        neg = (q < 0.0) & (acc_q > neg_min)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        acc_q = acc_q + cur
        q = q - pos.to(tl.float32) + neg.to(tl.float32)
        work = work | (normalized != 0.0) | (cur != 0.0)
        tl.store(
            out_seq_ptr + t * N + offsets, (cur * q_threshold).to(dtype), mask=mask
        )

    tl.store(q_final_ptr + offsets, q, mask=mask)
    tl.store(acc_q_final_ptr + offsets, acc_q, mask=mask)
    tl.store(cur_output_ptr + offsets, cur, mask=mask)
    tl.store(work_flags_ptr + pid_n, tl.max(work.to(tl.int32), axis=0))


def _select_stbif_kernel(T: int):
    if T <= _STBIF_STATIC_RANGE_MAX_T and use_static_range_for_triton_neuron_kernel(T):
        return _multi_step_stbif_kernel_static
    return _multi_step_stbif_kernel_dynamic


@register_op("sj::multi_step_stbif")
def multi_step_stbif(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_shape = x_seq.shape
    x_seq = x_seq.contiguous()
    q = q.contiguous()
    acc_q = acc_q.contiguous()
    T = x_seq.shape[0]
    N = x_seq[0].numel()
    dtype = x_seq.dtype
    if dtype not in type_dict:
        raise NotImplementedError(dtype)
    out_seq = torch.empty_like(x_seq)
    q_final = torch.empty_like(q)
    acc_q_final = torch.empty_like(acc_q)
    cur_output = torch.empty_like(q)

    x_seq_flat = x_seq.reshape(T, N)
    q_flat = q.reshape(N)
    acc_q_flat = acc_q.reshape(N)
    out_seq_flat = out_seq.reshape(T, N)
    q_final_flat = q_final.reshape(N)
    acc_q_final_flat = acc_q_final.reshape(N)
    cur_output_flat = cur_output.reshape(N)
    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    work_flags = torch.zeros(triton.cdiv(N, 32), device=x_seq.device, dtype=torch.int32)

    q_threshold_value = float(q_threshold.detach().item())
    pos_max_value = float(pos_max.detach().item())
    neg_min_value = float(neg_min.detach().item())
    with torch.cuda.device(x_seq.device):
        wrap_triton(_select_stbif_kernel(T))[grid](
            x_seq_flat,
            q_flat,
            acc_q_flat,
            out_seq_flat,
            q_final_flat,
            acc_q_final_flat,
            cur_output_flat,
            work_flags,
            q_threshold_value,
            pos_max_value,
            neg_min_value,
            T=T,
            N=N,
            dtype=type_dict[dtype],
        )
    return (
        out_seq.reshape(x_shape),
        q_final,
        acc_q_final,
        cur_output,
        work_flags.max(),
    )


@torch.library.register_fake("sj::multi_step_stbif")
def _multi_step_stbif_fake(
    x_seq: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
    return (
        x_seq.new_empty(x_seq.shape),
        q.new_empty(q.shape),
        acc_q.new_empty(acc_q.shape),
        q.new_empty(q.shape),
        torch.empty((), device=x_seq.device, dtype=torch.int32),
    )
