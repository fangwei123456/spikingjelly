from typing import Protocol

import torch
import torch.nn.functional as F

__all__ = [
    "SpikeCompressor",
    "NullSpikeCompressor",
    "BooleanSpikeCompressor",
    "Uint8SpikeCompressor",
    "BitSpikeCompressor",
    "SparseSpikeCompressor",
]


class SpikeCompressor(Protocol):
    r"""Compress and restore one tensor without storing per-call instance state.

    **中文：** 压缩器协议。``compress`` 返回值必须包含恢复 shape、dtype 和
    device 所需的信息；实现不得把单次调用信息写入实例。

    **English:** Compressor protocol. The value returned by ``compress`` must
    carry the metadata needed to restore shape, dtype, and device. Implementations
    must not store per-call metadata on the instance.
    """

    requires_strictly_binary: bool

    def compress(self, x: torch.Tensor) -> object: ...

    def decompress(self, packed: object) -> torch.Tensor: ...


class NullSpikeCompressor:
    requires_strictly_binary = False

    def __init__(self) -> None:
        r"""Keep a tensor unchanged. / 保持张量不变。"""

    def compress(self, x: torch.Tensor) -> object:
        return x.detach()

    def decompress(self, packed: object) -> torch.Tensor:
        if not isinstance(packed, torch.Tensor):
            raise TypeError("NullSpikeCompressor expects a tensor payload.")
        return packed


class BooleanSpikeCompressor:
    requires_strictly_binary = True

    def __init__(self) -> None:
        r"""Store strictly binary spikes as bool. / 将严格二值脉冲保存为 bool。"""

    def compress(self, x: torch.Tensor) -> object:
        return x.to(torch.bool), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        values, dtype, shape = packed
        return values.to(dtype=dtype).reshape(shape)


class Uint8SpikeCompressor:
    requires_strictly_binary = False

    def __init__(self) -> None:
        r"""Store integer-valued spikes as uint8. / 将整数脉冲保存为 uint8。"""

    def compress(self, x: torch.Tensor) -> object:
        return x.to(torch.uint8), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        values, dtype, shape = packed
        return values.to(dtype=dtype).reshape(shape)


class BitSpikeCompressor:
    requires_strictly_binary = True

    def __init__(self) -> None:
        r"""Pack eight strictly binary spikes into one byte.

        **中文：** 输入必须严格取值为 0 或 1。输出在 CPU 和 CUDA 上均由可被
        ``torch.compile`` 捕获的 PyTorch 张量运算生成。

        **English:** Inputs must contain only 0 and 1. CPU and CUDA use the same
        PyTorch tensor operations so the path can be captured by ``torch.compile``.
        """

    def compress(self, x: torch.Tensor) -> object:
        flat = x.to(torch.uint8).reshape(-1)
        padding = (-flat.numel()) % 8
        if padding:
            flat = F.pad(flat, (0, padding))
        shifts = torch.arange(8, dtype=torch.uint8, device=x.device)
        values = (flat.reshape(-1, 8) << shifts).sum(dim=1, dtype=torch.int16)
        return values.to(torch.uint8), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        values, dtype, shape = packed
        shifts = torch.arange(8, dtype=torch.uint8, device=values.device)
        flat = ((values.unsqueeze(1) >> shifts) & 1).reshape(-1)
        return flat[: torch.Size(shape).numel()].to(dtype=dtype).reshape(shape)


class SparseSpikeCompressor:
    requires_strictly_binary = True

    def __init__(self, dtype: torch.dtype = torch.int64) -> None:
        r"""Store indices of nonzero strictly binary spikes.

        **中文：** ``dtype`` 是保存索引的数据类型。

        **English:** ``dtype`` is the integer dtype used for stored indices.

        :param dtype: index dtype / 索引 dtype
        :type dtype: torch.dtype
        """
        self.dtype = dtype

    def compress(self, x: torch.Tensor) -> object:
        indices = torch.nonzero(x.reshape(-1), as_tuple=False).reshape(-1)
        return indices.to(self.dtype), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        indices, dtype, shape = packed
        flat = torch.zeros(
            torch.Size(shape).numel(), dtype=dtype, device=indices.device
        )
        return flat.scatter_(0, indices.to(torch.int64), 1).reshape(shape)
