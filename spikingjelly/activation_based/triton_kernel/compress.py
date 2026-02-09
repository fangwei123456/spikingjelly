import torch

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging
    from . import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.compress: {e}")
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()

from .triton_utils import contiguous_and_device_guard

__all__ = ["bit_spike_compress", "bit_spike_decompress"]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in [64, 128, 256]],
    key=[],
    restore_value=["s_seq_compressed_ptr"]
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

    tl.store(
        s_seq_compressed_ptr + store_offsets, s_seq_compressed, mask=store_mask
    )

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in [64, 128, 256]],
    key=[],
    restore_value=["s_seq_decompressed_ptr"]
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


@contiguous_and_device_guard
def bit_spike_compress(s_seq):
    # s_seq: float32, ndim=1
    s_seq = s_seq.reshape(-1)
    n_elements = s_seq.numel()
    n_compressed_elements = (n_elements + 7) // 8
    s_seq_compressed = torch.zeros(
        n_compressed_elements, dtype=torch.uint8, device=s_seq.device
    )
    grid = lambda meta: (triton.cdiv(n_compressed_elements, meta["BLOCK_SIZE"]),)

    with torch.cuda.device(s_seq.device):
        _bit_spike_compress_triton[grid](
            s_seq,
            s_seq_compressed,
            n_elements,
            n_compressed_elements,
        )
    return s_seq_compressed


@contiguous_and_device_guard
def bit_spike_decompress(s_seq_compressed, shape):
    # s_seq: uint8, ndim=1
    n_compressed_elements = s_seq_compressed.numel()
    n_decompressed_elements = shape.numel()
    s_seq_decompressed = torch.zeros(
        n_decompressed_elements, dtype=torch.uint8, device=s_seq_compressed.device
    )
    grid = lambda meta: (triton.cdiv(n_compressed_elements, meta["BLOCK_SIZE"]),)

    with torch.cuda.device(s_seq_compressed.device):
        _bit_spike_decompress_triton[grid](
            s_seq_compressed,
            s_seq_decompressed,
            n_compressed_elements,
            n_decompressed_elements,
        )
    return s_seq_decompressed.reshape(shape)