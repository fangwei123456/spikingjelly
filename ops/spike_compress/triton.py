import math
from typing import List

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .validation import _check_pack, _check_unpack


@triton.autotune(
    do_bench=triton.testing.do_bench,
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in [64, 128, 256]],
    key=[],
    restore_value=["s_seq_compressed_ptr"],
)
@triton.jit
def _bit_spike_compress_triton(
    s_seq_ptr,  # fp32, 0 or 1
    s_seq_compressed_ptr,
    n_elements,
    n_compressed_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    store_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    store_mask = store_offsets < n_compressed_elements

    s_seq_compressed = tl.zeros(
        [
            BLOCK_SIZE,
        ],
        dtype=tl.uint8,
    )

    for i in tl.static_range(8):
        load_offsets = i + store_offsets * 8
        load_mask = load_offsets < n_elements
        s_seq = tl.load(s_seq_ptr + load_offsets, mask=load_mask, other=0.0)
        s_seq = s_seq.to(tl.uint8)
        s_seq_compressed = s_seq_compressed | (s_seq << i)

    tl.store(s_seq_compressed_ptr + store_offsets, s_seq_compressed, mask=store_mask)


@triton.autotune(
    do_bench=triton.testing.do_bench,
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in [64, 128, 256]],
    key=[],
    restore_value=["s_seq_decompressed_ptr"],
)
@triton.jit
def _bit_spike_decompress_triton(
    s_seq_compressed_ptr,
    s_seq_decompressed_ptr,
    n_compressed_elements,
    n_decompressed_elements,
    BLOCK_SIZE: tl.constexpr,  # must be dividable by 8
):
    pid = tl.program_id(0)
    load_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    load_mask = load_offsets < n_compressed_elements

    s_seq_compressed = tl.load(
        s_seq_compressed_ptr + load_offsets,
        mask=load_mask,
        other=0,
    )

    for i in tl.static_range(8):
        store_offsets = i + load_offsets * 8
        store_mask = store_offsets < n_decompressed_elements
        tl.store(
            s_seq_decompressed_ptr + store_offsets,
            (s_seq_compressed >> i) & 1,
            mask=store_mask,
        )


@triton_op("sj_spike_compress::triton_forward", mutates_args=())
def _pack(x: torch.Tensor) -> torch.Tensor:
    _check_pack(x)
    torch._check(x.is_cuda, lambda: "Triton spike packing requires CUDA")
    x = x.bool().contiguous()
    packed = torch.empty(((x.numel() + 7) // 8,), device=x.device, dtype=torch.uint8)
    if packed.numel():
        with torch.cuda.device(x.device):
            wrap_triton(_bit_spike_compress_triton)[
                lambda meta: (triton.cdiv(packed.numel(), meta["BLOCK_SIZE"]),)
            ](x, packed, x.numel(), packed.numel())
    return packed


@triton_op("sj_spike_compress::triton_unpack", mutates_args=())
def _unpack(packed: torch.Tensor, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    _check_unpack(packed, shape)
    torch._check(packed.is_cuda, lambda: "Triton spike unpacking requires CUDA")
    packed = packed.contiguous()
    out = torch.empty(shape, device=packed.device, dtype=dtype)
    size = math.prod(shape)
    if size:
        with torch.cuda.device(packed.device):
            wrap_triton(_bit_spike_decompress_triton)[
                lambda meta: (triton.cdiv(packed.numel(), meta["BLOCK_SIZE"]),)
            ](packed, out, packed.numel(), size)
    return out
